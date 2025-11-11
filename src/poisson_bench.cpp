#include "cavity.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::size_t grid = 65;
    double tolerance = 1e-6;
    int max_iterations = 50000;
    int repeats = 5;
    int batch = 1;
    std::string output = "results/poisson_serial.csv";
};

std::string value(int& index, int argc, char** argv) {
    if (index + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    return argv[++index];
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--grid") {
            options.grid = std::stoul(value(i, argc, argv));
        } else if (key == "--tol") {
            options.tolerance = std::stod(value(i, argc, argv));
        } else if (key == "--max-iters") {
            options.max_iterations = std::stoi(value(i, argc, argv));
        } else if (key == "--repeats") {
            options.repeats = std::stoi(value(i, argc, argv));
        } else if (key == "--batch") {
            options.batch = std::stoi(value(i, argc, argv));
        } else if (key == "--output") {
            options.output = value(i, argc, argv);
        } else if (key == "--help" || key == "-h") {
            std::cout << "poisson_serial_bench [--grid N] [--tol VALUE] [--max-iters N] "
                         "[--repeats N] [--batch N] [--output PATH]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + key);
        }
    }
    if (options.grid < 5 || options.tolerance <= 0.0 ||
        options.max_iterations < 1 || options.repeats < 1 || options.batch < 1) {
        throw std::invalid_argument("invalid benchmark configuration");
    }
    return options;
}

void make_problem(cavity::Grid& exact, cavity::Grid& omega, double h) {
    const double pi = std::acos(-1.0);
    const auto n = exact.size();
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            exact(i, j) = std::sin(pi * i * h) * std::sin(pi * j * h);
        }
    }
    for (std::size_t j = 1; j + 1 < n; ++j) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const double lap = (exact(i + 1, j) + exact(i - 1, j) +
                                exact(i, j + 1) + exact(i, j - 1) - 4.0 * exact(i, j)) / (h * h);
            omega(i, j) = -lap;
        }
    }
}

double max_error(const cavity::Grid& actual, const cavity::Grid& exact) {
    double error = 0.0;
    const auto n = actual.size();
    for (std::size_t j = 1; j + 1 < n; ++j) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            error = std::max(error, std::abs(actual(i, j) - exact(i, j)));
        }
    }
    return error;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        const double h = 1.0 / static_cast<double>(options.grid - 1);
        cavity::Grid exact(options.grid), omega(options.grid);
        make_problem(exact, omega, h);
        const auto parent = std::filesystem::path(options.output).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        std::ofstream out(options.output);
        if (!out) throw std::runtime_error("cannot write benchmark CSV");
        out << "implementation,grid_size,tolerance,repeat,batch,iterations,residual,max_error,seconds,cell_updates_per_second\n";
        out << std::setprecision(17);
        std::vector<double> timings;
        for (int repeat = 1; repeat <= options.repeats; ++repeat) {
            cavity::Grid psi(options.grid), scratch(options.grid);
            cavity::PoissonResult result;
            const auto start = std::chrono::steady_clock::now();
            for (int run = 0; run < options.batch; ++run) {
                // Each solve starts from zero, so batches measure the same work.
                psi.fill(0.0);
                scratch.fill(0.0);
                result = cavity::solve_poisson(omega, psi, scratch, h,
                                                options.tolerance, options.max_iterations);
            }
            const auto end = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(end - start).count() / options.batch;
            const double error = max_error(psi, exact);
            const double updates = static_cast<double>(result.iterations) *
                                   static_cast<double>(options.grid - 2) * static_cast<double>(options.grid - 2);
            out << "serial," << options.grid << ',' << options.tolerance << ',' << repeat << ','
                << options.batch << ','
                << result.iterations << ',' << result.residual << ',' << error << ','
                << seconds << ',' << updates / seconds << '\n';
            timings.push_back(seconds);
        }
        if (!out) throw std::runtime_error("benchmark CSV write failed");
        std::sort(timings.begin(), timings.end());
        std::cout << std::setprecision(8) << "grid=" << options.grid
                  << " repeats=" << options.repeats
                  << " batch=" << options.batch
                  << " median_seconds=" << timings[timings.size() / 2]
                  << " csv=" << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
