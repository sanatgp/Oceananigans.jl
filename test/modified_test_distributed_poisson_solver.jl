using MPI
MPI.Init()

using Random
Random.seed!(1234)

# Initialize Dagger with MPI
using Dagger
Dagger.accelerate!(:mpi)

include("dependencies_for_runtests.jl")
include("dependencies_for_poisson_solvers.jl")

include("../src/DistributedComputations/dagger_oceananigans_integration.jl")

using Oceananigans.DistributedComputations: reconstruct_global_grid, DistributedGrid, Partition, DistributedFourierTridiagonalPoissonSolver
using Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!

function random_divergent_source_term(grid::DistributedGrid)
    arch = architecture(grid)
    default_bcs = FieldBoundaryConditions()

    u_bcs = regularize_field_boundary_conditions(default_bcs, grid, :u)
    v_bcs = regularize_field_boundary_conditions(default_bcs, grid, :v)
    w_bcs = regularize_field_boundary_conditions(default_bcs, grid, :w)

    u_bcs = inject_halo_communication_boundary_conditions(u_bcs, arch.local_rank, arch.connectivity, topology(grid))
    v_bcs = inject_halo_communication_boundary_conditions(v_bcs, arch.local_rank, arch.connectivity, topology(grid))
    w_bcs = inject_halo_communication_boundary_conditions(w_bcs, arch.local_rank, arch.connectivity, topology(grid))

    Ru = XFaceField(grid, boundary_conditions=u_bcs)
    Rv = YFaceField(grid, boundary_conditions=v_bcs)
    Rw = ZFaceField(grid, boundary_conditions=w_bcs)
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, rand(size(Ru)...))
    set!(Rv, rand(size(Rv)...))
    set!(Rw, rand(size(Rw)...))

    fill_halo_regions!(Ru)
    fill_halo_regions!(Rv)
    fill_halo_regions!(Rw)

    # Compute the right hand side R = ∇⋅U
    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R)

    return R, U
end
function divergence_free_poisson_solution_dagger(grid_points, ranks, topo, child_arch)
    arch = Distributed(child_arch, partition=Partition(ranks...))
    local_grid = RectilinearGrid(arch, topology=topo, size=grid_points, extent=(2π, 2π, 2π))

    # The test will solve for ϕ, then compare R to ∇²ϕ.
    ϕ   = CenterField(local_grid)
    ∇²ϕ = CenterField(local_grid)
    R, U = random_divergent_source_term(local_grid)

    global_grid = reconstruct_global_grid(local_grid)
    
    # Keep using the original solver for now
    solver = DistributedFFTBasedPoissonSolver(global_grid, local_grid)

    # ADD TIMING
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    # Warmup
    solve_for_pressure!(ϕ, solver, 1, U)
    
    # Timed solve
    MPI.Barrier(comm)
    start_time = MPI.Wtime()
    solve_for_pressure!(ϕ, solver, 1, U)
    MPI.Barrier(comm)
    elapsed_time = MPI.Wtime() - start_time
    
    if rank == 0
        println("DaggerFFT: Grid $(grid_points), Ranks $(ranks), Topology $(topo), Time: $(elapsed_time) seconds")
    end

    # "Recompute" ∇²ϕ
    compute_∇²!(∇²ϕ, ϕ, arch, local_grid)

    return Array(interior(∇²ϕ)) ≈ Array(R)
end


function benchmark_dagger_fft_solver(grid_points, ranks, topo, child_arch, iterations=10)
    arch = Distributed(child_arch, partition=Partition(ranks...))
    local_grid = RectilinearGrid(arch, topology=topo, size=grid_points, extent=(2π, 2π, 2π))

    ϕ   = CenterField(local_grid)
    R, U = random_divergent_source_term(local_grid)

    global_grid = reconstruct_global_grid(local_grid)
    solver = DistributedDaggerFFTBasedPoissonSolver(global_grid, local_grid)

    # Warmup
    solve_for_pressure!(ϕ, solver, 1, U)
    solve_for_pressure!(ϕ, solver, 1, U)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    sz = MPI.Comm_size(comm)

    times = Float64[]
    
    for iter in 1:iterations
        MPI.Barrier(comm)
        start_time = MPI.Wtime()
        
        solve_for_pressure!(ϕ, solver, 1, U)
        
        MPI.Barrier(comm)
        elapsed_time = MPI.Wtime() - start_time
        push!(times, elapsed_time)
        
        if rank == 0 && iter == iterations
            N = maximum(grid_points)
            fftsize = Float64(prod(grid_points))
            avg_time = sum(times[end-4:end]) / 5  # Average of last 5 iterations
            floprate = 5.0 * fftsize * log2(fftsize) * 1e-9 / avg_time
            
            println("----------------------------------------------------------------------------- ")
            println("DaggerFFT Oceananigans Poisson Solver performance test")
            println("----------------------------------------------------------------------------- ")
            println("Grid size: $(grid_points)")
            println("MPI ranks: $(sz)")
            println("Topology: $(topo)")
            println("Average time per solve: $(avg_time) (s)")
            println("Estimated performance: $(floprate) GFlops/s")
            println("----------------------------------------------------------------------------- ")
        end
    end
    
    return times
end

@testset "Distributed DaggerFFT-based Poisson solver" begin
    for topology in ((Periodic, Periodic, Periodic),
                     (Periodic, Periodic, Bounded),
                     (Periodic, Bounded, Bounded),
                     (Bounded, Bounded, Bounded))

        @info "  Testing 3D distributed DaggerFFT-based Poisson solver with topology $topology and (4, 1, 1) ranks..."
        @test divergence_free_poisson_solution_dagger((44, 44, 8), (4, 1, 1), topology, child_arch)
        @test divergence_free_poisson_solution_dagger((16, 44, 8), (4, 1, 1), topology, child_arch)
        
        @info "  Testing 3D distributed DaggerFFT-based Poisson solver with topology $topology and (1, 4, 1) ranks..."
        @test divergence_free_poisson_solution_dagger((44, 44, 8), (1, 4, 1), topology, child_arch)
        @test divergence_free_poisson_solution_dagger((44, 16, 8), (1, 4, 1), topology, child_arch)
        @test divergence_free_poisson_solution_dagger((16, 44, 8), (1, 4, 1), topology, child_arch)
        
        @info "  Testing 3D distributed DaggerFFT-based Poisson solver with topology $topology and (2, 2, 1) ranks..."
        @test divergence_free_poisson_solution_dagger((22, 44, 8), (2, 2, 1), topology, child_arch)
        @test divergence_free_poisson_solution_dagger((44, 22, 8), (2, 2, 1), topology, child_arch)

        @info "  Testing 2D distributed DaggerFFT-based Poisson solver with topology $topology..."
        @test divergence_free_poisson_solution_dagger((44, 16, 1), (4, 1, 1), topology, child_arch)
        @test divergence_free_poisson_solution_dagger((16, 44, 1), (4, 1, 1), topology, child_arch)
    end
end

@testset "DaggerFFT Performance Benchmarks" begin
    topology = (Periodic, Periodic, Periodic)
    
    @info "Benchmarking DaggerFFT solver performance..."
    benchmark_dagger_fft_solver((64, 64, 64), (4, 1, 1), topology, child_arch, 10)
    benchmark_dagger_fft_solver((128, 128, 64), (4, 1, 1), topology, child_arch, 10)
    benchmark_dagger_fft_solver((64, 64, 64), (2, 2, 1), topology, child_arch, 10)
end