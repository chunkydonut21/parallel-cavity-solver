#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cavity {

class Grid {
public:
    explicit Grid(std::size_t n, double initial = 0.0);
    std::size_t size() const { return n_; }
    double& operator()(std::size_t i, std::size_t j) { return data_[j * n_ + i]; }
    double operator()(std::size_t i, std::size_t j) const { return data_[j * n_ + i]; }
    void swap(Grid& other);
    void fill(double value);

private:
    std::size_t n_;
    std::vector<double> data_;
};

struct PoissonResult {
    int iterations = 0;
    double residual = 0.0;
};

PoissonResult solve_poisson(const Grid& omega, Grid& psi, Grid& scratch,
                            double h, double tolerance, int max_iterations);

struct Config {
    std::size_t grid_size = 65;
    double reynolds = 100.0;
    double lid_velocity = 1.0;
    double length = 1.0;
    int max_steps = 4000;
    double dt = 0.0; // Zero selects a conservative fixed timestep.
    double poisson_tolerance = 1e-6;
    int poisson_max_iterations = 15000;
    double steady_tolerance = 0.0; // Zero runs exactly max_steps.
    int min_steps = 100;
    std::string output = "results/cavity.csv";
    std::string metrics_output = "results/cavity_metrics.json";
};

struct RunSummary {
    int steps = 0;
    double dt = 0.0;
    double physical_time = 0.0;
    double runtime_seconds = 0.0;
    double poisson_seconds = 0.0;
    double velocity_seconds = 0.0;
    double boundary_seconds = 0.0;
    double transport_seconds = 0.0;
    double last_vorticity_change_rate = 0.0;
    double last_poisson_residual = 0.0;
    double mean_poisson_iterations = 0.0;
    bool steady_converged = false;
};

class Solver {
public:
    explicit Solver(const Config& config);
    RunSummary run();
    void save_csv(const std::string& path) const;
    void save_metrics(const std::string& path, const RunSummary& summary) const;
    const Grid& vorticity() const { return omega_; }
    const Grid& streamfunction() const { return psi_; }
    const Grid& horizontal_velocity() const { return u_; }
    const Grid& vertical_velocity() const { return v_; }

private:
    void apply_vorticity_boundary();
    void compute_velocity();
    double advance_vorticity(double dt);
    double choose_timestep() const;

    Config config_;
    double h_;
    double viscosity_;
    Grid omega_;
    Grid omega_next_;
    Grid psi_;
    Grid psi_scratch_;
    Grid u_;
    Grid v_;
};

} // namespace cavity
