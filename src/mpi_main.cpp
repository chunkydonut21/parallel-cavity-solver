#include "mpi_solver.hpp"

#include <mpi.h>

#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage() {
    std::cout << "cavity_mpi [options] (launch with mpiexec -n RANKS)\n"
              << "  --grid N                 Points per side (default 65)\n"
              << "  --re RE                  Reynolds number (default 100)\n"
              << "  --steps N                Maximum timesteps (default 4000)\n"
              << "  --dt VALUE               Fixed timestep; zero selects auto dt\n"
              << "  --poisson-tol VALUE      Maximum absolute Poisson residual (default 1e-6)\n"
              << "  --poisson-max-iters N    Maximum Jacobi sweeps per solve (default 15000)\n"
              << "  --steady-tol VALUE       Stop at this max |delta omega|/dt\n"
              << "  --min-steps N            Minimum steps before steady stopping (default 100)\n"
              << "  --output PATH            Flow CSV (default results/cavity.csv)\n"
              << "  --metrics PATH           Run metadata JSON (default results/cavity_metrics.json)\n";
}

std::string next_value(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    }
    return argv[++index];
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        cavity::Config config;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help" || option == "-h") {
                if (rank == 0) usage();
                MPI_Finalize();
                return 0;
            } else if (option == "--grid") {
                config.grid_size = std::stoul(next_value(i, argc, argv));
            } else if (option == "--re") {
                config.reynolds = std::stod(next_value(i, argc, argv));
            } else if (option == "--steps") {
                config.max_steps = std::stoi(next_value(i, argc, argv));
            } else if (option == "--dt") {
                config.dt = std::stod(next_value(i, argc, argv));
            } else if (option == "--poisson-tol") {
                config.poisson_tolerance = std::stod(next_value(i, argc, argv));
            } else if (option == "--poisson-max-iters") {
                config.poisson_max_iterations = std::stoi(next_value(i, argc, argv));
            } else if (option == "--steady-tol") {
                config.steady_tolerance = std::stod(next_value(i, argc, argv));
            } else if (option == "--min-steps") {
                config.min_steps = std::stoi(next_value(i, argc, argv));
            } else if (option == "--output") {
                config.output = next_value(i, argc, argv);
            } else if (option == "--metrics") {
                config.metrics_output = next_value(i, argc, argv);
            } else {
                throw std::invalid_argument("unknown option: " + option);
            }
        }

        cavity::MpiSolver solver(config, MPI_COMM_WORLD);
        const auto summary = solver.run();
        solver.save_csv(config.output);
        solver.save_metrics(config.metrics_output, summary);
        if (rank == 0) {
            std::cout << std::setprecision(8)
                      << "grid=" << config.grid_size << " re=" << config.reynolds
                      << " steps=" << summary.steps << " dt=" << summary.dt << '\n'
                      << "runtime_seconds=" << summary.runtime_seconds
                      << " mean_poisson_iterations=" << summary.mean_poisson_iterations
                      << " final_poisson_residual=" << summary.last_poisson_residual << '\n'
                      << "poisson_seconds=" << summary.poisson_seconds
                      << " transport_seconds=" << summary.transport_seconds << '\n'
                      << "last_vorticity_change_rate=" << summary.last_vorticity_change_rate
                      << " steady_converged=" << std::boolalpha << summary.steady_converged << '\n'
                      << "csv=" << config.output << " metrics=" << config.metrics_output << '\n';
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank << " error: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
