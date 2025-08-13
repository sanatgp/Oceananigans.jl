import FFTW
using Dagger
using MPI
using LinearAlgebra
using AbstractFFTs
using KernelAbstractions

using .DaggerFFTs
import .DaggerFFTs: FFT!, FFT, fft, Pencil, Slab, IFFT!, ifft

using GPUArraysCore
using Oceananigans.Grids: XYZRegularRG, XYRegularRG, XZRegularRG, YZRegularRG
import Oceananigans.Solvers: poisson_eigenvalues, solve!
import Oceananigans.Architectures: architecture, child_architecture
import Oceananigans.Fields: interior
using Oceananigans.Grids: topology, size, stretched_dimensions
using Oceananigans.Fields: CenterField
using Oceananigans.DistributedComputations: TransposableField, partition_coordinate, child_architecture
using Oceananigans.Utils: launch!

struct DistributedFFTBasedPoissonSolver{P, F, L, λ, D, S}
    plan :: P
    global_grid :: F
    local_grid :: L
    eigenvalues :: λ
    dagger_arrays :: D 
    storage :: S
end

architecture(solver::DistributedFFTBasedPoissonSolver) =
    architecture(solver.global_grid)

function DistributedFFTBasedPoissonSolver(global_grid, local_grid, planner_flag=FFTW.PATIENT)
    
    validate_poisson_solver_distributed_grid(global_grid)
    validate_poisson_solver_configuration(global_grid, local_grid)
    
    FT = Complex{eltype(local_grid)}
    
    if !MPI.Initialized()
        MPI.Init()
    end
    Dagger.accelerate!(:mpi)
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    sz = MPI.Comm_size(comm)
    
    Nx, Ny, Nz = size(global_grid)
    
    arch = architecture(local_grid)
    Rx, Ry, Rz = arch.ranks
    
    if rank == 0
        dummy_data = zeros(FT, Nx, Ny, Nz)
        
        # Create DArrays with appropriate block distributions for pencil decomposition
        # Match your DaggerFFT example: (N, 128, 128) -> (128, N, 128) -> (128, 128, N)
        if Rx > 1 && Ry == 1  # Decomposed in x only (slab)
            DA = distribute(dummy_data, Blocks(Nx, div(Ny, Rx), Nz); root=0, comm=comm)
            DB = distribute(dummy_data, Blocks(div(Nx, Rx), Ny, Nz); root=0, comm=comm)
            DC = distribute(dummy_data, Blocks(div(Nx, Rx), div(Ny, Rx), Nz); root=0, comm=comm)
        elseif Rx == 1 && Ry > 1  # Decomposed in y only (slab)
            DA = distribute(dummy_data, Blocks(Nx, div(Ny, Ry), Nz); root=0, comm=comm)
            DB = distribute(dummy_data, Blocks(div(Nx, Ry), Ny, Nz); root=0, comm=comm)
            DC = distribute(dummy_data, Blocks(div(Nx, Ry), div(Ny, Ry), Nz); root=0, comm=comm)
        else  # Pencil decomposition (Rx > 1 && Ry > 1)
            DA = distribute(dummy_data, Blocks(Nx, div(Ny, Rx), div(Nz, Ry)); root=0, comm=comm)
            DB = distribute(dummy_data, Blocks(div(Nx, Ry), Ny, div(Nz, Ry)); root=0, comm=comm)
            DC = distribute(dummy_data, Blocks(div(Nx, Ry), div(Ny, Rx), Nz); root=0, comm=comm)
        end
    else
        if Rx > 1 && Ry == 1
            DA = distribute(nothing, Blocks(Nx, div(Ny, Rx), Nz); root=0, comm=comm)
            DB = distribute(nothing, Blocks(div(Nx, Rx), Ny, Nz); root=0, comm=comm)
            DC = distribute(nothing, Blocks(div(Nx, Rx), div(Ny, Rx), Nz); root=0, comm=comm)
        elseif Rx == 1 && Ry > 1
            DA = distribute(nothing, Blocks(Nx, div(Ny, Ry), Nz); root=0, comm=comm)
            DB = distribute(nothing, Blocks(div(Nx, Ry), Ny, Nz); root=0, comm=comm)
            DC = distribute(nothing, Blocks(div(Nx, Ry), div(Ny, Ry), Nz); root=0, comm=comm)
        else
            DA = distribute(nothing, Blocks(Nx, div(Ny, Rx), div(Nz, Ry)); root=0, comm=comm)
            DB = distribute(nothing, Blocks(div(Nx, Ry), Ny, div(Nz, Ry)); root=0, comm=comm)
            DC = distribute(nothing, Blocks(div(Nx, Ry), div(Ny, Rx), Nz); root=0, comm=comm)
        end
    end
    
    dagger_arrays = (DA=DA, DB=DB, DC=DC)
    
    storage = TransposableField(CenterField(local_grid), FT)
    
    child_arch = child_architecture(arch)
    
    # Build global eigenvalues
    topo = (TX, TY, TZ) = topology(global_grid)
    λx = dropdims(poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX()), dims=(2, 3))
    λy = dropdims(poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY()), dims=(1, 3))
    λz = dropdims(poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ()), dims=(1, 2))
    
    # Don't partition eigenvalues - we need them globally for DaggerFFT
    eigenvalues = (λx, λy, λz)
    
    plan = nothing
    
    return DistributedFFTBasedPoissonSolver(plan, global_grid, local_grid, eigenvalues, dagger_arrays, storage)
end

function solve!(x, solver::DistributedFFTBasedPoissonSolver)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    sz = MPI.Comm_size(comm)
    
    # Get the DArrays
    DA = solver.dagger_arrays.DA
    DB = solver.dagger_arrays.DB
    DC = solver.dagger_arrays.DC
    
    # Get dimensions
    Nx, Ny, Nz = size(solver.global_grid)
    arch = architecture(solver.local_grid)
    
    # Get the right hand side from storage (it was copied there before calling solve!)
    b_local = parent(solver.storage.zfield)  # This contains the local part of RHS
    
    # Gather all local parts to create the global array using MPI
    local_size = length(b_local)
    local_data = vec(Array(b_local))  # Flatten and ensure it's on CPU
    
    all_sizes = MPI.Allgather(local_size, comm)
    
    displacements = zeros(Int, sz)
    for i in 2:sz
        displacements[i] = displacements[i-1] + all_sizes[i-1]
    end
    total_size = sum(all_sizes)
    
    global_b_flat = zeros(Complex{eltype(solver.local_grid)}, total_size)
    MPI.Allgatherv!(local_data, global_b_flat, all_sizes, displacements, comm)
    
    global_b = reshape(global_b_flat, Nx, Ny, Nz)
    
    # Copy global data to the appropriate chunks of DA
    # Each rank copies to its own chunks
    for (idx, chunk) in enumerate(DA.chunks)
        if chunk.handle.rank == rank
            chunk_data = fetch(chunk)
            subdomain = DA.subdomains[idx]
            
            # Extract the relevant portion from global_b
            i_range = subdomain.indexes[1]
            j_range = subdomain.indexes[2]
            k_range = subdomain.indexes[3]
            
            chunk_data .= global_b[i_range, j_range, k_range]
        end
    end
    
    MPI.Barrier(comm)
    
    transforms = (FFT!(), FFT!(), FFT!())
    dims = (1, 2, 3)
    
    fft(DA, DB, DC, transforms, dims, Pencil())
    
    # Solve Poisson equation in spectral space (distributed)
    # Each rank works on its own chunks
    λx, λy, λz = solver.eigenvalues
    
    for (idx, chunk) in enumerate(DC.chunks)
        if chunk.handle.rank == rank
            chunk_data = fetch(chunk)
            subdomain = DC.subdomains[idx]
            
            # Get indices for this chunk
            i_range = subdomain.indexes[1]
            j_range = subdomain.indexes[2]
            k_range = subdomain.indexes[3]
            
            # Solve in spectral space for this chunk
            for (k_local, k) in enumerate(k_range)
                for (j_local, j) in enumerate(j_range)
                    for (i_local, i) in enumerate(i_range)
                        if i == 1 && j == 1 && k == 1
                            chunk_data[i_local, j_local, k_local] = 0 
                        else
                            chunk_data[i_local, j_local, k_local] = 
                                -chunk_data[i_local, j_local, k_local] / 
                                (λx[i] + λy[j] + λz[k])
                        end
                    end
                end
            end
        end
    end
    
    MPI.Barrier(comm)
    
    inverse_transforms = (IFFT!(), IFFT!(), IFFT!())
    ifft(DC, DB, DA, inverse_transforms, dims, Pencil())
    
    # Extract solution from DA and distribute to local arrays
    # Gather the solution
    solution_flat = zeros(Complex{eltype(solver.local_grid)}, total_size)
    
    # Each rank extracts its portion from DA
    for (idx, chunk) in enumerate(DA.chunks)
        if chunk.handle.rank == rank
            chunk_data = fetch(chunk)
            subdomain = DA.subdomains[idx]
            
            # Copy this chunk's data to the appropriate position in solution
            i_range = subdomain.indexes[1]
            j_range = subdomain.indexes[2] 
            k_range = subdomain.indexes[3]
            
            for (k_local, k) in enumerate(k_range)
                for (j_local, j) in enumerate(j_range)
                    for (i_local, i) in enumerate(i_range)
                        linear_idx = (k-1)*Nx*Ny + (j-1)*Nx + i
                        solution_flat[linear_idx] = chunk_data[i_local, j_local, k_local]
                    end
                end
            end
        end
    end
    
    MPI.Allreduce!(solution_flat, MPI.SUM, comm)
    
    global_solution = reshape(solution_flat, Nx, Ny, Nz)
    
    # Copy the real part of the local portion to x
    local_Nx, local_Ny, local_Nz = size(solver.local_grid)
    x_reshaped = reshape(x, local_Nx, local_Ny, local_Nz)
    
    # Calculate local indices based on rank
    Rx, Ry, Rz = arch.ranks
    rank_x = rank % Rx
    rank_y = (rank ÷ Rx) % Ry
    
    i_start = rank_x * local_Nx + 1
    i_end = (rank_x + 1) * local_Nx
    j_start = rank_y * local_Ny + 1
    j_end = (rank_y + 1) * local_Ny
    
    x_reshaped .= real.(global_solution[i_start:i_end, j_start:j_end, 1:local_Nz])
    
    return x
end

validate_poisson_solver_distributed_grid(global_grid) =
    throw("Grids other than the RectilinearGrid are not supported in the Distributed NonhydrostaticModels")

function validate_poisson_solver_distributed_grid(global_grid::RectilinearGrid)
    TX, TY, TZ = topology(global_grid)
    
    if (TY == Bounded && TZ == Periodic) || (TX == Bounded && TY == Periodic) || (TX == Bounded && TZ == Periodic)
        throw("Distributed Poisson solvers do not support grids with topology ($TX, $TY, $TZ) at the moment.")
    end
    
    if !(global_grid isa YZRegularRG) && !(global_grid isa XYRegularRG) && !(global_grid isa XZRegularRG)
        throw("The provided grid is stretched in directions $(stretched_dimensions(global_grid)).")
    end
    
    return nothing
end

function validate_poisson_solver_configuration(global_grid, local_grid)
    Rx, Ry, Rz = architecture(local_grid).ranks
    Rz == 1 || throw("Non-singleton ranks in the vertical are not supported.")
    
    if global_grid.Nz % Ry != 0
        throw("The number of ranks in the y-direction are $(Ry) with Nz = $(global_grid.Nz) cells.")
    end
    
    if global_grid.Ny % Rx != 0
        throw("The number of ranks in the x-direction are $(Rx) with Ny = $(global_grid.Ny) cells.")
    end
    
    return nothing
end

@kernel function _solve_poisson_in_spectral_space!(x̂, b̂, λx, λy, λz)
    i, j, k = @index(Global, NTuple)
    @inbounds x̂[i, j, k] = - b̂[i, j, k] / (λx[i] + λy[j] + λz[k])
end

@kernel function _copy_real_component!(ϕ, ϕc)
    i, j, k = @index(Global, NTuple)
    @inbounds ϕ[i, j, k] = real(ϕc[i, j, k])
end