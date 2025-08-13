using MPI
MPI.Init()

# Make sure results are
# reproducible
using Random
Random.seed!(1234)

include("dependencies_for_runtests.jl")
include("dependencies_for_poisson_solvers.jl")

using Dagger
Dagger.accelerate!(:mpi)

# # Distributed Poisson Solver tests
#
# These tests are meant to be run on 4 ranks. This script may be run
# stand-alone (outside the test environment) via
#
# mpiexec -n 4 julia --project test_distributed_poisson_solver.jl
#
# provided that a few packages (like TimesDates.jl) are in your global environment.
#
# Another possibility is to use tmpi ():
#
# tmpi 4 julia --project
#
# then later:
#
# julia> include("test_distributed_poisson_solver.jl")
#
# When running the tests this way, uncomment the following line

# to initialize MPI.

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

function divergence_free_poisson_solution(grid_points, ranks, topo, child_arch)
    arch = Distributed(child_arch, partition=Partition(ranks...))
    local_grid = RectilinearGrid(arch, topology=topo, size=grid_points, extent=(2π, 2π, 2π))

    # The test will solve for ϕ, then compare R to ∇²ϕ.
    ϕ   = CenterField(local_grid)
    ∇²ϕ = CenterField(local_grid)
    R, U = random_divergent_source_term(local_grid)

    global_grid = reconstruct_global_grid(local_grid)
    solver = DistributedFFTBasedPoissonSolver(global_grid, local_grid)

    # ADD TIMING FOR PERFORMANCE COMPARISON
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    # Warmup run
    solve_for_pressure!(ϕ, solver, 1, U)
    
    # Multiple timed runs for better statistics
    num_runs = 5
    times = zeros(num_runs)
    
    for i in 1:num_runs
        MPI.Barrier(comm)
        start_time = MPI.Wtime()
        solve_for_pressure!(ϕ, solver, 1, U)
        MPI.Barrier(comm)
        times[i] = MPI.Wtime() - start_time
    end
    
    avg_time = sum(times) / num_runs
    min_time = minimum(times)
    max_time = maximum(times)
    
    if rank == 0
        println("=============================================================================")
        println("DAGGERFFT IMPLEMENTATION: Grid $(grid_points), Ranks $(ranks), Topology $(topo)")
        println("Average Time: $(avg_time) seconds")
        println("Min Time: $(min_time) seconds, Max Time: $(max_time) seconds")
        
        # Calculate performance metrics
        Nx, Ny, Nz = grid_points
        fftsize = Float64(Nx * Ny * Nz)
        # 2 FFTs (forward and inverse) in the solve
        floprate = 2.0 * 5.0 * fftsize * log2(fftsize) * 1e-9 / avg_time
        println("Performance: $(floprate) GFlops/s")
        println("=============================================================================")
    end

    # "Recompute" ∇²ϕ
    compute_∇²!(∇²ϕ, ϕ, arch, local_grid)

    return Array(interior(∇²ϕ)) ≈ Array(R)
end

function divergence_free_poisson_tridiagonal_solution(grid_points, ranks, stretched_direction, child_arch)
    arch = Distributed(child_arch, partition=Partition(ranks...))

    if stretched_direction == :x
        x = collect(range(0, 2π, length = grid_points[1]+1))
        y = z = (0, 2π)
    elseif stretched_direction == :y
        y = collect(range(0, 2π, length = grid_points[2]+1))
        x = z = (0, 2π)
    elseif stretched_direction == :z
        z = collect(range(0, 2π, length = grid_points[3]+1))
        x = y = (0, 2π)
    end

    local_grid = RectilinearGrid(arch;
                                 topology=(Bounded, Bounded, Bounded),
                                 size=grid_points,
                                 halo=(2, 2, 2),
                                 x, y, z)

    # The test will solve for ϕ, then compare R to ∇²ϕ.
    ϕ   = CenterField(local_grid)
    ∇²ϕ = CenterField(local_grid)
    R, U = random_divergent_source_term(local_grid)

    global_grid = reconstruct_global_grid(local_grid)
    solver = DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid)

    # Using Δt = 1.
    solve_for_pressure!(ϕ, solver, 1, U)

    # "Recompute" ∇²ϕ
    compute_∇²!(∇²ϕ, ϕ, arch, local_grid)

    return Array(interior(∇²ϕ)) ≈ Array(R)
end

@testset "Distributed FFT-based Poisson solver" begin
    for topology in ((Periodic, Periodic, Periodic),
                     (Periodic, Periodic, Bounded),
                     (Periodic, Bounded, Bounded),
                     (Bounded, Bounded, Bounded))

        @info "  Testing 3D distributed FFT-based Poisson solver with topology $topology and (4, 1, 1) ranks..."
        @test divergence_free_poisson_solution((44, 44, 8), (4, 1, 1), topology, child_arch)
        @test divergence_free_poisson_solution((16, 44, 8), (4, 1, 1), topology, child_arch)
        @info "  Testing 3D distributed FFT-based Poisson solver with topology $topology and (1, 4, 1) ranks..."
        @test divergence_free_poisson_solution((44, 44, 8), (1, 4, 1), topology, child_arch)
        @test divergence_free_poisson_solution((44, 16, 8), (1, 4, 1), topology, child_arch)
        @test divergence_free_poisson_solution((16, 44, 8), (1, 4, 1), topology, child_arch)
        @info "  Testing 3D distributed FFT-based Poisson solver with topology $topology and (2, 2, 1) ranks..."
        @test divergence_free_poisson_solution((22, 44, 8), (2, 2, 1), topology, child_arch)
        @test divergence_free_poisson_solution((44, 22, 8), (2, 2, 1), topology, child_arch)

        @info "  Testing 2D distributed FFT-based Poisson solver with topology $topology..."
        @test divergence_free_poisson_solution((44, 16, 1), (4, 1, 1), topology, child_arch)
        @test divergence_free_poisson_solution((16, 44, 1), (4, 1, 1), topology, child_arch)
    end

    for stretched_direction in (:z, )
        @info "  Testing 3D distributed Fourier Tridiagonal Poisson solver stretched in $stretched_direction with (4, 1, 1) ranks"
        @test divergence_free_poisson_tridiagonal_solution((44, 44, 8), (4, 1, 1), stretched_direction, child_arch)
        @test divergence_free_poisson_tridiagonal_solution((4,  44, 8), (4, 1, 1), stretched_direction, child_arch)
        @test divergence_free_poisson_tridiagonal_solution((16, 44, 8), (4, 1, 1), stretched_direction, child_arch)
        @info "  Testing 3D distributed Fourier Tridiagonal Poisson solver stretched in $stretched_direction with (1, 4, 1) ranks"
        @test divergence_free_poisson_tridiagonal_solution((44, 44, 8), (1, 4, 1), stretched_direction, child_arch)
        @test divergence_free_poisson_tridiagonal_solution((44,  4, 8), (1, 4, 1), stretched_direction, child_arch)
        @test divergence_free_poisson_tridiagonal_solution((16, 44, 8), (1, 4, 1), stretched_direction, child_arch)
        @info "  Testing 3D distributed Fourier Tridiagonal Poisson solver stretched in $stretched_direction with (2, 2, 1) ranks"
        @test divergence_free_poisson_tridiagonal_solution((22,  8, 8), (2, 2, 1), stretched_direction, child_arch)
        @test divergence_free_poisson_tridiagonal_solution(( 8, 22, 8), (2, 2, 1), stretched_direction, child_arch)
    end
end