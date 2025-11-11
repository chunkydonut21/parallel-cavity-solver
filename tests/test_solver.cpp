#include "cavity.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_poisson_manufactured_solution() {
    constexpr std::size_t n = 17;
    const double h = 1.0 / static_cast<double>(n - 1);
    const double pi = std::acos(-1.0);
    cavity::Grid exact(n), omega(n), psi(n), scratch(n);
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
    const auto result = cavity::solve_poisson(omega, psi, scratch, h, 1e-8, 10000);
    double max_error = 0.0;
    for (std::size_t j = 1; j + 1 < n; ++j) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            max_error = std::max(max_error, std::abs(psi(i, j) - exact(i, j)));
        }
    }
    require(result.residual <= 1e-8, "Poisson residual too large");
    require(max_error < 1e-9, "manufactured Poisson solution mismatch");
}

void test_short_cavity_run() {
    cavity::Config config;
    config.grid_size = 17;
    config.max_steps = 20;
    config.poisson_tolerance = 1e-5;
    cavity::Solver solver(config);
    const auto result = solver.run();
    require(result.steps == 20, "short run did not finish");
    require(result.last_poisson_residual <= config.poisson_tolerance, "final Poisson solve failed");
    require(std::isfinite(result.runtime_seconds), "runtime is not finite");
    const auto& u = solver.horizontal_velocity();
    const auto& v = solver.vertical_velocity();
    for (std::size_t i = 1; i + 1 < config.grid_size; ++i) {
        require(u(i, config.grid_size - 1) == config.lid_velocity, "top lid velocity mismatch");
        require(u(i, 0) == 0.0 && v(i, 0) == 0.0, "bottom wall velocity mismatch");
    }
    for (std::size_t j = 0; j < config.grid_size; ++j) {
        require(u(0, j) == 0.0 && v(0, j) == 0.0, "left wall velocity mismatch");
        require(u(config.grid_size - 1, j) == 0.0, "right wall velocity mismatch");
    }
}

} // namespace

int main() {
    try {
        test_poisson_manufactured_solution();
        test_short_cavity_run();
        std::cout << "serial solver checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
