
using FFTW
using GPUArraysCore
using Oceananigans.Grids: XYZRegularRG, XYRegularRG, XZRegularRG, YZRegularRG
using Dagger
using MPI

import Oceananigans.Solvers: poisson_eigenvalues, solve!
import Oceananigans.Architectures: architecture
import Oceananigans.Fields: interior

include("/home/taghipouranvari.s/new/MPI/Dagger.jl/src/fft.jl")

using .DaggerFFTs

struct DistributedDaggerFFTBasedPoissonSolver{P, F, L, λ, B, S, DA, DB, DC}
    plan :: P
    global_grid :: F
    local_grid :: L
    eigenvalues :: λ
    buffer :: B
    storage :: S
    # Add DArrays
    darray_a :: DA
    darray_b :: DB
    darray_c :: DC
end

architecture(solver::DistributedDaggerFFTBasedPoissonSolver) =
    architecture(solver.global_grid)

function DistributedDaggerFFTBasedPoissonSolver(global_grid, local_grid, planner_flag=FFTW.PATIENT)
    
    validate_poisson_solver_distributed_grid(global_grid)
    validate_poisson_solver_configuration(global_grid, local_grid)

    FT = Complex{eltype(local_grid)}

    storage = TransposableField(CenterField(local_grid), FT)

    arch = architecture(storage.xfield.grid)
    child_arch = child_architecture(arch)

    # Build _global_ eigenvalues
    topo = (TX, TY, TZ) = topology(global_grid)
    λx = dropdims(poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX()), dims=(2, 3))
    λy = dropdims(poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY()), dims=(1, 3))
    λz = dropdims(poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ()), dims=(1, 2))

    λx = partition_coordinate(λx, size(storage.xfield.grid, 1), arch, 1)
    λy = partition_coordinate(λy, size(storage.xfield.grid, 2), arch, 2)
    λz = partition_coordinate(λz, size(storage.xfield.grid, 3), arch, 3)

    λx = on_architecture(child_arch, λx)
    λy = on_architecture(child_arch, λy)
    λz = on_architecture(child_arch, λz)

    eigenvalues = (λx, λy, λz)

    plan = plan_distributed_transforms(global_grid, storage, planner_flag)

    x_buffer_needed = child_arch isa GPU && TX == Bounded
    z_buffer_needed = child_arch isa GPU && TZ == Bounded
    y_buffer_needed = child_arch isa GPU

    buffer_x = x_buffer_needed ? on_architecture(child_arch, zeros(FT, size(storage.xfield)...)) : nothing
    buffer_y = y_buffer_needed ? on_architecture(child_arch, zeros(FT, size(storage.yfield)...)) : nothing
    buffer_z = z_buffer_needed ? on_architecture(child_arch, zeros(FT, size(storage.zfield)...)) : nothing

    buffer = (; x = buffer_x, y = buffer_y, z = buffer_z)

    Nx, Ny, Nz = global_grid.Nx, global_grid.Ny, global_grid.Nz
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    if rank == 0
        dummy_array = zeros(ComplexF64, Nx, Ny, Nz)
        #hardcoded for 4 ranks
        darray_a = distribute(dummy_array, Blocks(Nx, Ny÷2, Nz÷2); root=0, comm=comm)
        darray_b = distribute(dummy_array, Blocks(Nx÷2, Ny, Nz÷2); root=0, comm=comm)
        darray_c = distribute(dummy_array, Blocks(Nx÷2, Ny÷2, Nz); root=0, comm=comm)
    else
        darray_a = distribute(nothing, Blocks(Nx, Ny÷2, Nz÷2); root=0, comm=comm)
        darray_b = distribute(nothing, Blocks(Nx÷2, Ny, Nz÷2); root=0, comm=comm)
        darray_c = distribute(nothing, Blocks(Nx÷2, Ny÷2, Nz); root=0, comm=comm)
    end

    return DistributedDaggerFFTBasedPoissonSolver(plan, global_grid, local_grid, eigenvalues, 
                                                  buffer, storage, darray_a, darray_b, darray_c)
end

# Modified solve! function using DaggerFFTs
function solve!(x, solver::DistributedDaggerFFTBasedPoissonSolver)
    storage = solver.storage
    arch = architecture(storage.xfield.grid)
    
    copy_storage_to_darray!(solver.darray_a, storage.zfield)
    
    fft(solver.darray_a, solver.darray_b, solver.darray_c, 
        (DaggerFFTs.FFT!(), DaggerFFTs.FFT!(), DaggerFFTs.FFT!()), 
        (1, 2, 3), DaggerFFTs.Pencil())

    copy_darray_to_storage!(storage.xfield, solver.darray_c)

    λ = solver.eigenvalues
    x̂ = b̂ = parent(storage.xfield)

    launch!(arch, storage.xfield.grid, :xyz, _solve_poisson_in_spectral_space!, x̂, b̂, λ[1], λ[2], λ[3])

    if arch.local_rank == 0
        @allowscalar x̂[1, 1, 1] = 0
    end

    copy_storage_to_darray!(solver.darray_c, storage.xfield)

    ifft(solver.darray_c, solver.darray_b, solver.darray_a,
         (DaggerFFTs.IFFT!(), DaggerFFTs.IFFT!(), DaggerFFTs.IFFT!()), 
         (1, 2, 3), DaggerFFTs.Pencil())

    copy_darray_to_storage!(storage.zfield, solver.darray_a)

    launch!(arch, solver.local_grid, :xyz,
            _copy_real_component!, x, parent(storage.zfield))

    return x
end

function copy_storage_to_darray!(darray, field)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    local_data = parent(field)
    
    for (idx, chunk) in enumerate(darray.chunks)
        if chunk.handle.rank == rank
            chunk_data = fetch(chunk)
            copy_with_bounds_check!(chunk_data, local_data)
            break
        end
    end
end

function copy_darray_to_storage!(field, darray)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    local_data = parent(field)
    
    for (idx, chunk) in enumerate(darray.chunks)
        if chunk.handle.rank == rank
            chunk_data = fetch(chunk)
            copy_with_bounds_check!(local_data, chunk_data)
            break
        end
    end
end

function copy_with_bounds_check!(dest, src)
    dest_size = size(dest)
    src_size = size(src)
    
    copy_size = min.(dest_size, src_size)
    
    dest_view = view(dest, ntuple(i -> 1:copy_size[i], length(copy_size))...)
    src_view = view(src, ntuple(i -> 1:copy_size[i], length(copy_size))...)
    
    copyto!(dest_view, src_view)
end

@kernel function _solve_poisson_in_spectral_space!(x̂, b̂, λx, λy, λz)
    i, j, k = @index(Global, NTuple)
    @inbounds x̂[i, j, k] = - b̂[i, j, k] / (λx[i] + λy[j] + λz[k])
end

@kernel function _copy_real_component!(ϕ, ϕc)
    i, j, k = @index(Global, NTuple)
    @inbounds ϕ[i, j, k] = real(ϕc[i, j, k])
end

validate_poisson_solver_distributed_grid(global_grid) =
        throw("Grids other than the RectilinearGrid are not supported in the Distributed NonhydrostaticModels")

function validate_poisson_solver_distributed_grid(global_grid::RectilinearGrid)
    TX, TY, TZ = topology(global_grid)

    if (TY == Bounded && TZ == Periodic) || (TX == Bounded && TY == Periodic) || (TX == Bounded && TZ == Periodic)
        throw("Distributed Poisson solvers do not support grids with topology ($TX, $TY, $TZ) at the moment.
               A Periodic z-direction requires also the y- and and x-directions to be Periodic, while a Periodic y-direction requires also
               the x-direction to be Periodic.")
    end

    if !(global_grid isa YZRegularRG) && !(global_grid isa XYRegularRG) && !(global_grid isa XZRegularRG)
        throw("The provided grid is stretched in directions $(stretched_dimensions(global_grid)).
               A distributed Poisson solver supports only RectilinearGrids that have variably-spaced cells in at most one direction.")
    end

    return nothing
end

function validate_poisson_solver_configuration(global_grid, local_grid)
    # We don't support distributing anything in z.
    Rx, Ry, Rz = architecture(local_grid).ranks
    Rz == 1 || throw("Non-singleton ranks in the vertical are not supported by distributed Poisson solvers.")

    # Limitation of the current implementation (see the docstring)
    if global_grid.Nz % Ry != 0
        throw("The number of ranks in the y-direction are $(Ry) with Nz = $(global_grid.Nz) cells in the z-direction.
               The distributed Poisson solver requires that the number of ranks in the y-direction divide Nz.")
    end

    if global_grid.Ny % Rx != 0
        throw("The number of ranks in the y-direction are $(Rx) with Ny = $(global_grid.Ny) cells in the y-direction.
               The distributed Poisson solver requires that the number of ranks in the x-direction divide Ny.")
    end

    return nothing
end