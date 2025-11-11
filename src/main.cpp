#include "cavity.hpp"

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef CAVITY_USE_OPENMP
#include <omp.h>
#endif

namespace {

void usage() {
    std::cout
#ifdef CAVITY_USE_OPENMP
              << "cavity_openmp [options]\n"
              << "  --threads N              OpenMP threads (default: runtime setting)\n"
#else
              << "cavity_serial [options]\n"
#endif
              << "  --grid N                 Points per side (default 65)\n"
              << "  --re RE                  Reynolds number (default 100)\n"
              << "  --steps N                Maximum timesteps (default 4000)\n"
              << "  --dt VALUE               Fixed timestep; zero selects conservative auto dt\n"
              << "  --poisson-tol VALUE      Maximum absolute Poisson residual (default 1e-6)\n"
              << "  --poisson-max-iters N    Maximum Jacobi sweeps per solve (default 15000)\n"
              << "  --steady-tol VALUE       Stop when max |delta omega|/dt is below VALUE\n"
              << "  --min-steps N            Minimum steps before steady stopping (default 100)\n"
              << "  --output PATH            Flow CSV (default results/cavity.csv)\n"
              << "  --metrics PATH           Run metadata JSON (default results/cavity_metrics.json)\n";
}

std::string next_value(int& index, int argc, char** argv) {
    if (index + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    return argv[++index];
}

} // namespace

int main(int argc, char** argv) {
    try {
#ifdef CAVITY_USE_OPENMP
        omp_set_dynamic(0);
#endif
        cavity::Config config;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help" || option == "-h") {
                usage();
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
#ifdef CAVITY_USE_OPENMP
            } else if (option == "--threads") {
                const int threads = std::stoi(next_value(i, argc, argv));
                if (threads < 1) throw std::invalid_argument("--threads must be positive");
                omp_set_num_threads(threads);
#endif
            } else {
                throw std::invalid_argument("unknown option: " + option);
            }
        }
        cavity::Solver solver(config);
        const auto summary = solver.run();
        solver.save_csv(config.output);
        solver.save_metrics(config.metrics_output, summary);
        std::cout << std::setprecision(8)
                  << "grid=" << config.grid_size << " re=" << config.reynolds
                  << " steps=" << summary.steps << " dt=" << summary.dt
                  << " simulated_time=" << summary.physical_time << '\n'
                  << "runtime_seconds=" << summary.runtime_seconds
                  << " mean_poisson_iterations=" << summary.mean_poisson_iterations
                  << " final_poisson_residual=" << summary.last_poisson_residual << '\n'
                  << "poisson_seconds=" << summary.poisson_seconds
                  << " transport_seconds=" << summary.transport_seconds << '\n'
                  << "last_vorticity_change_rate=" << summary.last_vorticity_change_rate
                  << " steady_converged=" << std::boolalpha << summary.steady_converged << '\n'
                  << "csv=" << config.output << " metrics=" << config.metrics_output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
