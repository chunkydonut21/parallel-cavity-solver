#include "mpi_solver.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace cavity {

LocalGrid::LocalGrid(int columns, int owned_rows)
    : columns_(columns), values_((owned_rows + 2) * columns, 0.0) {}

void LocalGrid::swap(LocalGrid& other) {
    values_.swap(other.values_);
}

MpiSolver::MpiSolver(const Config& config, MPI_Comm comm)
    : config_(config), comm_(comm) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &ranks_);
    // MPI_Gatherv counts and offsets use int in this implementation.
    if (config_.grid_size < 5 || config_.grid_size > 46340 ||
        config_.reynolds <= 0.0 || config_.lid_velocity <= 0.0 ||
        config_.length <= 0.0 || config_.max_steps < 1 || config_.min_steps < 1 ||
        config_.poisson_tolerance <= 0.0 || config_.poisson_max_iterations < 1 ||
        config_.steady_tolerance < 0.0 || config_.dt < 0.0) {
        throw std::invalid_argument("invalid cavity configuration");
    }

    n_ = static_cast<int>(config_.grid_size);
    const int interior_rows = n_ - 2;
    if (ranks_ > interior_rows) {
        throw std::invalid_argument("MPI ranks cannot exceed the number of interior rows");
    }
    const int base = interior_rows / ranks_;
    const int extra = interior_rows % ranks_;
    rows_ = base + (rank_ < extra ? 1 : 0);
    h_ = config_.length / static_cast<double>(n_ - 1);
    viscosity_ = config_.lid_velocity * config_.length / config_.reynolds;

    omega_ = LocalGrid(n_, rows_);
    omega_next_ = LocalGrid(n_, rows_);
    psi_ = LocalGrid(n_, rows_);
    psi_scratch_ = LocalGrid(n_, rows_);
    u_ = LocalGrid(n_, rows_);
    v_ = LocalGrid(n_, rows_);
}

void MpiSolver::exchange_rows(LocalGrid& grid) const {
    const int below = rank_ == 0 ? MPI_PROC_NULL : rank_ - 1;
    const int above = rank_ == ranks_ - 1 ? MPI_PROC_NULL : rank_ + 1;

    // First exchange the rows needed above each strip, then those below it.
    MPI_Sendrecv(grid.row(1), n_, MPI_DOUBLE, below, 10,
                 grid.row(rows_ + 1), n_, MPI_DOUBLE, above, 10,
                 comm_, MPI_STATUS_IGNORE);
    MPI_Sendrecv(grid.row(rows_), n_, MPI_DOUBLE, above, 11,
                 grid.row(0), n_, MPI_DOUBLE, below, 11,
                 comm_, MPI_STATUS_IGNORE);
}

PoissonResult MpiSolver::solve_streamfunction() {
    const double h2 = h_ * h_;
    const double inv_h2 = 1.0 / h2;
    for (int iteration = 1; iteration <= config_.poisson_max_iterations; ++iteration) {
        exchange_rows(psi_);
        for (int j = 1; j <= rows_; ++j) {
            for (int i = 1; i < n_ - 1; ++i) {
                psi_scratch_(i, j) = 0.25 * (
                    psi_(i + 1, j) + psi_(i - 1, j) +
                    psi_(i, j + 1) + psi_(i, j - 1) + h2 * omega_(i, j));
            }
        }
        psi_.swap(psi_scratch_);

        if (iteration == 1 || iteration % 20 == 0 ||
            iteration == config_.poisson_max_iterations) {
            // The new psi needs current ghost rows before its residual is checked.
            exchange_rows(psi_);
            double local_residual = 0.0;
            for (int j = 1; j <= rows_; ++j) {
                for (int i = 1; i < n_ - 1; ++i) {
                    const double lap = (psi_(i + 1, j) + psi_(i - 1, j) +
                                        psi_(i, j + 1) + psi_(i, j - 1) -
                                        4.0 * psi_(i, j)) * inv_h2;
                    local_residual = std::max(local_residual,
                                              std::abs(lap + omega_(i, j)));
                }
            }
            double residual = 0.0;
            MPI_Allreduce(&local_residual, &residual, 1, MPI_DOUBLE, MPI_MAX, comm_);
            if (residual <= config_.poisson_tolerance) return {iteration, residual};
            if (iteration == config_.poisson_max_iterations) {
                throw std::runtime_error("Poisson solver did not reach its residual target");
            }
        }
    }
    throw std::logic_error("unreachable Poisson solver state");
}

void MpiSolver::compute_velocity() {
    const double factor = 0.5 / h_;
    for (int j = 1; j <= rows_; ++j) {
        for (int i = 1; i < n_ - 1; ++i) {
            u_(i, j) = (psi_(i, j + 1) - psi_(i, j - 1)) * factor;
            v_(i, j) = -(psi_(i + 1, j) - psi_(i - 1, j)) * factor;
        }
    }
    if (rank_ == ranks_ - 1) {
        for (int i = 1; i < n_ - 1; ++i) {
            u_(i, rows_ + 1) = config_.lid_velocity;
        }
    }
}

void MpiSolver::apply_vorticity_boundary() {
    const double factor = -2.0 / (h_ * h_);
    for (int j = 1; j <= rows_; ++j) {
        omega_(0, j) = factor * psi_(1, j);
        omega_(n_ - 1, j) = factor * psi_(n_ - 2, j);
    }
    if (rank_ == 0) {
        for (int i = 1; i < n_ - 1; ++i) {
            omega_(i, 0) = factor * psi_(i, 1);
        }
        omega_(0, 0) = 0.5 * (omega_(1, 0) + omega_(0, 1));
        omega_(n_ - 1, 0) = 0.5 * (omega_(n_ - 2, 0) + omega_(n_ - 1, 1));
    }
    if (rank_ == ranks_ - 1) {
        for (int i = 1; i < n_ - 1; ++i) {
            omega_(i, rows_ + 1) = factor * psi_(i, rows_)
                                   - 2.0 * config_.lid_velocity / h_;
        }
        omega_(0, rows_ + 1) = 0.5 * (omega_(1, rows_ + 1) + omega_(0, rows_));
        omega_(n_ - 1, rows_ + 1) = 0.5 * (
            omega_(n_ - 2, rows_ + 1) + omega_(n_ - 1, rows_));
    }
}

double MpiSolver::advance_vorticity(double dt) {
    const double inv_h = 1.0 / h_;
    const double inv_h2 = inv_h * inv_h;
    double local_change = 0.0;
    double local_bad = 0.0;
    for (int j = 1; j <= rows_; ++j) {
        for (int i = 1; i < n_ - 1; ++i) {
            const double u = u_(i, j);
            const double v = v_(i, j);
            double dx;
            if (u >= 0.0) dx = (omega_(i, j) - omega_(i - 1, j)) * inv_h;
            else dx = (omega_(i + 1, j) - omega_(i, j)) * inv_h;

            double dy;
            if (v >= 0.0) dy = (omega_(i, j) - omega_(i, j - 1)) * inv_h;
            else dy = (omega_(i, j + 1) - omega_(i, j)) * inv_h;

            const double lap = (omega_(i + 1, j) + omega_(i - 1, j) +
                                omega_(i, j + 1) + omega_(i, j - 1) -
                                4.0 * omega_(i, j)) * inv_h2;
            const double updated = omega_(i, j) + dt * (-u * dx - v * dy + viscosity_ * lap);
            if (!std::isfinite(updated)) local_bad = 1.0;
            omega_next_(i, j) = updated;
            local_change = std::max(local_change, std::abs(updated - omega_(i, j)));
        }
    }
    double local_values[2] = {local_change, local_bad};
    double global_values[2] = {0.0, 0.0};
    MPI_Allreduce(local_values, global_values, 2, MPI_DOUBLE, MPI_MAX, comm_);
    if (global_values[1] != 0.0) {
        throw std::runtime_error("non-finite vorticity; reduce --dt");
    }
    omega_.swap(omega_next_);
    return global_values[0] / dt;
}

double MpiSolver::choose_timestep() const {
    if (config_.dt > 0.0) return config_.dt;
    const double advection_limit = h_ / config_.lid_velocity;
    const double diffusion_limit = h_ * h_ / (4.0 * viscosity_);
    return 0.2 * std::min(advection_limit, diffusion_limit);
}

RunSummary MpiSolver::run() {
    RunSummary result;
    result.dt = choose_timestep();
    double total_poisson_iterations = 0.0;
    MPI_Barrier(comm_);
    const double simulation_start = MPI_Wtime();

    for (int step = 1; step <= config_.max_steps; ++step) {
        const double poisson_start = MPI_Wtime();
        const PoissonResult poisson = solve_streamfunction();
        const double poisson_end = MPI_Wtime();

        compute_velocity();
        const double velocity_end = MPI_Wtime();

        apply_vorticity_boundary();
        const double boundary_end = MPI_Wtime();

        exchange_rows(omega_);
        result.last_vorticity_change_rate = advance_vorticity(result.dt);
        const double transport_end = MPI_Wtime();

        result.poisson_seconds += poisson_end - poisson_start;
        result.velocity_seconds += velocity_end - poisson_end;
        result.boundary_seconds += boundary_end - velocity_end;
        result.transport_seconds += transport_end - boundary_end;
        total_poisson_iterations += poisson.iterations;
        result.steps = step;

        if (config_.steady_tolerance > 0.0 && step >= config_.min_steps &&
            result.last_vorticity_change_rate <= config_.steady_tolerance) {
            result.steady_converged = true;
            break;
        }
    }

    const double poisson_start = MPI_Wtime();
    const PoissonResult final_poisson = solve_streamfunction();
    const double poisson_end = MPI_Wtime();
    compute_velocity();
    const double velocity_end = MPI_Wtime();
    apply_vorticity_boundary();
    const double simulation_end = MPI_Wtime();

    result.poisson_seconds += poisson_end - poisson_start;
    result.velocity_seconds += velocity_end - poisson_end;
    result.boundary_seconds += simulation_end - velocity_end;
    result.runtime_seconds = simulation_end - simulation_start;
    result.physical_time = result.steps * result.dt;
    result.last_poisson_residual = final_poisson.residual;
    result.mean_poisson_iterations = total_poisson_iterations / result.steps;

    // Report the slowest rank: it determines when the whole run finishes.
    double local_times[5] = {result.runtime_seconds, result.poisson_seconds,
                             result.velocity_seconds, result.boundary_seconds,
                             result.transport_seconds};
    double global_times[5] = {};
    MPI_Allreduce(local_times, global_times, 5, MPI_DOUBLE, MPI_MAX, comm_);
    result.runtime_seconds = global_times[0];
    result.poisson_seconds = global_times[1];
    result.velocity_seconds = global_times[2];
    result.boundary_seconds = global_times[3];
    result.transport_seconds = global_times[4];
    return result;
}

std::vector<double> MpiSolver::gather(const LocalGrid& grid) const {
    std::vector<int> counts(ranks_), offsets(ranks_);
    const int base = (n_ - 2) / ranks_;
    const int extra = (n_ - 2) % ranks_;
    for (int r = 0; r < ranks_; ++r) {
        const int owned = base + (r < extra ? 1 : 0);
        const int first = 1 + r * base + std::min(r, extra);
        const int sent_rows = owned + (r == 0 ? 1 : 0) + (r == ranks_ - 1 ? 1 : 0);
        counts[r] = sent_rows * n_;
        offsets[r] = (r == 0 ? 0 : first) * n_;
    }
    const int first_local_row = rank_ == 0 ? 0 : 1;
    std::vector<double> global;
    if (rank_ == 0) global.resize(n_ * n_);
    MPI_Gatherv(grid.row(first_local_row), counts[rank_], MPI_DOUBLE,
                global.data(), counts.data(), offsets.data(), MPI_DOUBLE,
                0, comm_);
    return global;
}

static void ensure_parent(const std::string& path) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
}

void MpiSolver::save_csv(const std::string& path) const {
    // Gather only for output. The numerical work remains distributed.
    const auto u = gather(u_);
    const auto v = gather(v_);
    const auto omega = gather(omega_);
    const auto psi = gather(psi_);
    if (rank_ != 0) return;

    ensure_parent(path);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write CSV: " + path);
    out << "x,y,u,v,speed,omega,psi\n" << std::setprecision(17);
    for (int j = 0; j < n_; ++j) {
        for (int i = 0; i < n_; ++i) {
            const int index = j * n_ + i;
            out << i * h_ << ',' << j * h_ << ',' << u[index] << ',' << v[index] << ','
                << std::hypot(u[index], v[index]) << ',' << omega[index] << ','
                << psi[index] << '\n';
        }
    }
    if (!out) throw std::runtime_error("CSV write failed: " + path);
}

void MpiSolver::save_metrics(const std::string& path, const RunSummary& s) const {
    if (rank_ != 0) return;
    ensure_parent(path);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write metrics: " + path);
    out << std::setprecision(17)
        << "{\n  \"implementation\": \"mpi\",\n"
        << "  \"ranks\": " << ranks_ << ",\n"
        << "  \"grid_size\": " << n_ << ",\n"
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
