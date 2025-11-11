#pragma once

#include "cavity.hpp"

#include <mpi.h>

#include <string>
#include <vector>

namespace cavity {

// Interior rows plus one ghost row on each side of the strip.
class LocalGrid {
public:
    LocalGrid() = default;
    LocalGrid(int columns, int owned_rows);

    double& operator()(int i, int j) { return values_[j * columns_ + i]; }
    double operator()(int i, int j) const { return values_[j * columns_ + i]; }
    double* row(int j) { return values_.data() + j * columns_; }
    const double* row(int j) const { return values_.data() + j * columns_; }
    void swap(LocalGrid& other);

private:
    int columns_ = 0;
    std::vector<double> values_;
};

class MpiSolver {
public:
    MpiSolver(const Config& config, MPI_Comm comm);
    RunSummary run();
    void save_csv(const std::string& path) const; // Called by every rank.
    void save_metrics(const std::string& path, const RunSummary& summary) const;
    int rank() const { return rank_; }

private:
    void exchange_rows(LocalGrid& grid) const;
    PoissonResult solve_streamfunction();
    void compute_velocity();
    void apply_vorticity_boundary();
    double advance_vorticity(double dt);
    double choose_timestep() const;
    std::vector<double> gather(const LocalGrid& grid) const;

    Config config_;
    MPI_Comm comm_;
    int rank_ = 0;
    int ranks_ = 1;
    int n_ = 0;
    int rows_ = 0;
    double h_ = 0.0;
    double viscosity_ = 0.0;
    LocalGrid omega_, omega_next_, psi_, psi_scratch_, u_, v_;
};

} // namespace cavity
