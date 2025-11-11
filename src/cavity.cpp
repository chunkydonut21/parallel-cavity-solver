#include "cavity.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#ifdef CAVITY_USE_OPENMP
#include <omp.h>
#endif

namespace cavity {

Grid::Grid(std::size_t n, double initial) : n_(n), data_(n * n, initial) {
    if (n < 3) throw std::invalid_argument("grid must contain at least 3 points per side");
}

void Grid::swap(Grid& other) {
    if (n_ != other.n_) throw std::invalid_argument("cannot swap grids of different sizes");
    data_.swap(other.data_);
}

void Grid::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

static double max_poisson_residual(const Grid& omega, const Grid& psi, double h) {
    const int n = static_cast<int>(psi.size());
    const double inv_h2 = 1.0 / (h * h);
    double result = 0.0;
#ifdef CAVITY_USE_OPENMP
    std::vector<double> row_maxima(n, 0.0);
    // Rows only read psi and omega, and write separate maxima.
#pragma omp parallel for schedule(static)
#endif
    for (int j = 1; j < n - 1; ++j) {
        double row_max = 0.0;
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const double lap = (psi(i + 1, j) + psi(i - 1, j) +
                                psi(i, j + 1) + psi(i, j - 1) - 4.0 * psi(i, j)) * inv_h2;
            row_max = std::max(row_max, std::abs(lap + omega(i, j)));
        }
#ifdef CAVITY_USE_OPENMP
        row_maxima[j] = row_max;
#else
        result = std::max(result, row_max);
#endif
    }
#ifdef CAVITY_USE_OPENMP
    for (double value : row_maxima) result = std::max(result, value);
#endif
    return result;
}

PoissonResult solve_poisson(const Grid& omega, Grid& psi, Grid& scratch,
                            double h, double tolerance, int max_iterations) {
    if (omega.size() != psi.size() || psi.size() != scratch.size() || h <= 0.0 ||
        tolerance <= 0.0 || max_iterations < 1) {
        throw std::invalid_argument("invalid Poisson solver inputs");
    }
    const int n = static_cast<int>(psi.size());
    const double h2 = h * h;
    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        // Two grids make the Jacobi update independent at every interior cell.
        // Jacobi reads the old grid and writes each scratch cell once.
#ifdef CAVITY_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 1; j < n - 1; ++j) {
            for (std::size_t i = 1; i + 1 < n; ++i) {
                scratch(i, j) = 0.25 * (psi(i + 1, j) + psi(i - 1, j) +
                                         psi(i, j + 1) + psi(i, j - 1) + h2 * omega(i, j));
            }
        }
        psi.swap(scratch);
        // A residual measures the equation error; a small update alone can be misleading.
        if (iteration == 1 || iteration % 20 == 0 || iteration == max_iterations) {
            const double residual = max_poisson_residual(omega, psi, h);
            if (residual <= tolerance) return {iteration, residual};
            if (iteration == max_iterations) {
                throw std::runtime_error("Poisson solver residual " + std::to_string(residual) +
                                         " after " + std::to_string(max_iterations) +
                                         " sweeps; increase the iteration cap or relax the tolerance");
            }
        }
    }
    throw std::logic_error("unreachable Poisson solver state");
}

Solver::Solver(const Config& config)
    : config_(config),
      h_(config_.length / static_cast<double>(config_.grid_size - 1)),
      viscosity_(config_.lid_velocity * config_.length / config_.reynolds),
      omega_(config_.grid_size), omega_next_(config_.grid_size),
      psi_(config_.grid_size), psi_scratch_(config_.grid_size),
      u_(config_.grid_size), v_(config_.grid_size) {
    if (config_.grid_size < 5 || config_.reynolds <= 0.0 || config_.lid_velocity <= 0.0 ||
        config_.length <= 0.0 || config_.max_steps < 1 || config_.min_steps < 1 ||
        config_.steady_tolerance < 0.0 || config_.dt < 0.0) {
        throw std::invalid_argument("invalid cavity configuration");
    }
}

double Solver::choose_timestep() const {
    if (config_.dt > 0.0) {
        return config_.dt;
    }

    const double advection_limit = h_ / config_.lid_velocity;
    const double diffusion_limit = h_ * h_ / (4.0 * viscosity_);
    return 0.2 * std::min(advection_limit, diffusion_limit);
}

void Solver::apply_vorticity_boundary() {
    const auto n = config_.grid_size;
    const double factor = -2.0 / (h_ * h_);
    for (std::size_t i = 1; i + 1 < n; ++i) {
        omega_(i, 0) = factor * psi_(i, 1);
        omega_(i, n - 1) = factor * psi_(i, n - 2) - 2.0 * config_.lid_velocity / h_;
    }
    for (std::size_t j = 1; j + 1 < n; ++j) {
        omega_(0, j) = factor * psi_(1, j);
        omega_(n - 1, j) = factor * psi_(n - 2, j);
    }
    // Corners are assigned the mean of their two neighboring wall values.
    omega_(0, 0) = 0.5 * (omega_(1, 0) + omega_(0, 1));
    omega_(n - 1, 0) = 0.5 * (omega_(n - 2, 0) + omega_(n - 1, 1));
    omega_(0, n - 1) = 0.5 * (omega_(1, n - 1) + omega_(0, n - 2));
    omega_(n - 1, n - 1) = 0.5 * (omega_(n - 2, n - 1) + omega_(n - 1, n - 2));
}

void Solver::compute_velocity() {
    const int n = static_cast<int>(config_.grid_size);
    const double factor = 0.5 / h_;
    // Interior velocities read psi and write separate u/v cells.
#ifdef CAVITY_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 1; j < n - 1; ++j) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            u_(i, j) = (psi_(i, j + 1) - psi_(i, j - 1)) * factor;
            v_(i, j) = -(psi_(i + 1, j) - psi_(i - 1, j)) * factor;
        }
    }
    // The lid is moving except at the two corners, which belong to side walls.
    for (std::size_t i = 1; i + 1 < n; ++i) {
        u_(i, n - 1) = config_.lid_velocity;
    }
}

double Solver::advance_vorticity(double dt) {
    const int n = static_cast<int>(config_.grid_size);
    const double inv_h = 1.0 / h_;
    const double inv_h2 = inv_h * inv_h;
    double largest_change = 0.0;
    int non_finite = 0;
#ifdef CAVITY_USE_OPENMP
    std::vector<double> row_changes(n, 0.0);
    std::vector<int> row_errors(n, 0);
#endif
    // Every update reads old omega and writes a distinct cell in omega_next_.
#ifdef CAVITY_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 1; j < n - 1; ++j) {
        double row_change = 0.0;
        int row_error = 0;
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const double u = u_(i, j);
            const double v = v_(i, j);
            // Use the value upstream of the flow at this cell.
            double dx;
            if (u >= 0.0) {
                dx = (omega_(i, j) - omega_(i - 1, j)) * inv_h;
            } else {
                dx = (omega_(i + 1, j) - omega_(i, j)) * inv_h;
            }

            double dy;
            if (v >= 0.0) {
                dy = (omega_(i, j) - omega_(i, j - 1)) * inv_h;
            } else {
                dy = (omega_(i, j + 1) - omega_(i, j)) * inv_h;
            }
            const double lap = (omega_(i + 1, j) + omega_(i - 1, j) +
                                omega_(i, j + 1) + omega_(i, j - 1) - 4.0 * omega_(i, j)) * inv_h2;
            const double updated = omega_(i, j) + dt * (-u * dx - v * dy + viscosity_ * lap);
            if (!std::isfinite(updated)) row_error = 1;
            omega_next_(i, j) = updated;
            row_change = std::max(row_change, std::abs(updated - omega_(i, j)));
        }
#ifdef CAVITY_USE_OPENMP
        row_changes[j] = row_change;
        row_errors[j] = row_error;
#else
        largest_change = std::max(largest_change, row_change);
        non_finite = std::max(non_finite, row_error);
#endif
    }
#ifdef CAVITY_USE_OPENMP
    for (int j = 1; j < n - 1; ++j) {
        largest_change = std::max(largest_change, row_changes[j]);
        non_finite = std::max(non_finite, row_errors[j]);
    }
#endif
    if (non_finite) throw std::runtime_error("non-finite vorticity; reduce --dt");
    omega_.swap(omega_next_);
    return largest_change / dt;
}

RunSummary Solver::run() {
    RunSummary result;
    result.dt = choose_timestep();
    double total_poisson_iterations = 0.0;
    const auto simulation_start = std::chrono::steady_clock::now();

    for (int step = 1; step <= config_.max_steps; ++step) {
        const auto poisson_start = std::chrono::steady_clock::now();
        const PoissonResult poisson = solve_poisson(
            omega_, psi_, psi_scratch_, h_,
            config_.poisson_tolerance, config_.poisson_max_iterations);
        const auto poisson_end = std::chrono::steady_clock::now();

        compute_velocity();
        const auto velocity_end = std::chrono::steady_clock::now();

        apply_vorticity_boundary();
        const auto boundary_end = std::chrono::steady_clock::now();

        result.last_vorticity_change_rate = advance_vorticity(result.dt);
        const auto transport_end = std::chrono::steady_clock::now();

        result.poisson_seconds += std::chrono::duration<double>(poisson_end - poisson_start).count();
        result.velocity_seconds += std::chrono::duration<double>(velocity_end - poisson_end).count();
        result.boundary_seconds += std::chrono::duration<double>(boundary_end - velocity_end).count();
        result.transport_seconds += std::chrono::duration<double>(transport_end - boundary_end).count();
        total_poisson_iterations += poisson.iterations;
        result.steps = step;

        if (config_.steady_tolerance > 0.0 && step >= config_.min_steps &&
            result.last_vorticity_change_rate <= config_.steady_tolerance) {
            result.steady_converged = true;
            break;
        }
    }

    // Recompute fields for the final omega, so CSV columns describe one state.
    const auto poisson_start = std::chrono::steady_clock::now();
    const PoissonResult final_poisson = solve_poisson(
        omega_, psi_, psi_scratch_, h_,
        config_.poisson_tolerance, config_.poisson_max_iterations);
    const auto poisson_end = std::chrono::steady_clock::now();

    compute_velocity();
    const auto velocity_end = std::chrono::steady_clock::now();

    apply_vorticity_boundary();
    const auto simulation_end = std::chrono::steady_clock::now();

    result.poisson_seconds += std::chrono::duration<double>(poisson_end - poisson_start).count();
    result.velocity_seconds += std::chrono::duration<double>(velocity_end - poisson_end).count();
    result.boundary_seconds += std::chrono::duration<double>(simulation_end - velocity_end).count();
    result.runtime_seconds = std::chrono::duration<double>(simulation_end - simulation_start).count();
    result.physical_time = result.steps * result.dt;
    result.last_poisson_residual = final_poisson.residual;
    result.mean_poisson_iterations = total_poisson_iterations / result.steps;
    return result;
}

static void ensure_parent(const std::string& path) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
}

void Solver::save_csv(const std::string& path) const {
    ensure_parent(path);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write CSV: " + path);
    out << "x,y,u,v,speed,omega,psi\n" << std::setprecision(17);
    for (std::size_t j = 0; j < config_.grid_size; ++j) {
        for (std::size_t i = 0; i < config_.grid_size; ++i) {
            out << i * h_ << ',' << j * h_ << ',' << u_(i, j) << ',' << v_(i, j) << ','
                << std::hypot(u_(i, j), v_(i, j)) << ',' << omega_(i, j) << ',' << psi_(i, j) << '\n';
        }
    }
    if (!out) throw std::runtime_error("CSV write failed: " + path);
}

void Solver::save_metrics(const std::string& path, const RunSummary& s) const {
    ensure_parent(path);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write metrics: " + path);
    out << std::setprecision(17)
#ifdef CAVITY_USE_OPENMP
        << "{\n  \"implementation\": \"openmp\",\n"
        << "  \"threads\": " << omp_get_max_threads() << ",\n"
#else
        << "{\n  \"implementation\": \"serial\",\n"
        << "  \"threads\": 1,\n"
#endif
        << "  \"grid_size\": " << config_.grid_size << ",\n"
        << "  \"reynolds\": " << config_.reynolds << ",\n"
        << "  \"lid_velocity\": " << config_.lid_velocity << ",\n"
        << "  \"steps\": " << s.steps << ",\n"
        << "  \"dt\": " << s.dt << ",\n"
        << "  \"physical_time\": " << s.physical_time << ",\n"
        << "  \"runtime_seconds\": " << s.runtime_seconds << ",\n"
        << "  \"poisson_seconds\": " << s.poisson_seconds << ",\n"
        << "  \"velocity_seconds\": " << s.velocity_seconds << ",\n"
        << "  \"boundary_seconds\": " << s.boundary_seconds << ",\n"
        << "  \"transport_seconds\": " << s.transport_seconds << ",\n"
        << "  \"poisson_tolerance\": " << config_.poisson_tolerance << ",\n"
        << "  \"last_poisson_residual\": " << s.last_poisson_residual << ",\n"
        << "  \"mean_poisson_iterations\": " << s.mean_poisson_iterations << ",\n"
        << "  \"last_vorticity_change_rate\": " << s.last_vorticity_change_rate << ",\n"
        << "  \"steady_converged\": " << (s.steady_converged ? "true" : "false") << "\n}\n";
    if (!out) throw std::runtime_error("metrics write failed: " + path);
}

} // namespace cavity
