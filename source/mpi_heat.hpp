#define USE_MATH_DEFINES
#include <mpi.h>
#include <vector>
#include <math.h>
#include <cassert>
#include <cstring>
#define L 1.0


void print_vector(const std::vector<double>& v) {
    for (auto& el : v)
        std::printf("%.3f ", el);
    std::cout << std::endl;
}

double vector_err(const std::vector<double>& v1, const std::vector<double>& v2) {
    assert(v1.size() == v2.size());
    double err = 0.0;
    for (uint_fast32_t i = 0, i_end = v1.size(); i < i_end; ++i)
        err = std::max(err, std::abs(v1[i] - v2[i]));
    return err;
}

// exact solution on one dimensional rod with length L = 1.0
double exact_solution(double x, double t, double k, double u_0) {
    constexpr uint8_t max_iter = 100;
    using namespace std;
    double sum = 0.0;
    double dif = 1.0;
    uint8_t m = 0;
    for (uint8_t m = 0; dif > 1e-10 && m < max_iter; ++m) {
        dif = exp(-k * M_PI * M_PI * (2 * m + 1) * (2 * m + 1) * t / (L * L));
        dif /= (2 * m + 1);
        dif *= sin(M_PI * (2 * m + 1) * x / L);
        sum += dif;
    }

    return 4 * u_0 * sum / M_PI;
}

std::vector<double> exact_solution_vector(const double t, const double k, const double u_0, const uint_fast32_t n_h) {
    std::vector<double> res(n_h + 1);
    const double h = L / n_h;
    for (uint_fast32_t i = 0, i_end = n_h + 1; i < i_end; ++i) {
        res[i] = exact_solution(i * h, t, k, u_0);
    }
    return res;
}

// get intervals of indicies for all processes 
std::vector<int> proc_intervals(
    const uint_fast32_t N,
    const uint_fast32_t n_proc
) {
    uint_fast32_t n = N / n_proc;
    std::vector<int> res(n_proc + 1, n);
    for (uint_fast32_t i = 1, i_end = N % n_proc + 1; i < i_end; ++i) {
        res[i]++;
    }
    res[0] = 0;
    for (uint_fast32_t i = 1; i <= n_proc; ++i) {
        res[i] += res[i - 1];
    }
    assert(res[n_proc] == N);
    return res;
}

// one heatstep
inline void heat_step(const double* u, double* u_next, const double kt_h2, const uint_fast32_t N) {
    for (uint_fast32_t i = 1, i_end = N - 1; i < i_end; i++) {
        u_next[i] = u[i] + kt_h2 * (u[i - 1] - 2 * u[i] + u[i + 1]);
    }
}

enum alg_type {
    simple,
    async,
    gpu,
    last
};


template<alg_type alg>
std::vector<double> solve_heat_mpi(
    const double T,
    const double k,
    const uint_fast64_t n_h,
    const uint_fast64_t n_t,
    const std::vector<double>& u_0
) {
    const uint_fast64_t N = n_h + 1;
    const double h = L / n_h;
    const double tau = T / n_t;
    const double kt_h2 = k * tau / (h * h);

    // printf("tau = %f, kt_h2 = %f\n", tau, kt_h2);

    std::vector<double> res;

    { // MPI section
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        const std::vector<int> intervals = proc_intervals(N, world_size);
        std::vector<int> sizes(world_size);
        for (int i = 0; i < world_size; i++) {
            sizes[i] = intervals[i + 1] - intervals[i];
        }

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        const bool is_left = rank == 0;
        const bool is_right = rank == world_size - 1;

        const uint_fast64_t start = intervals[rank];
        const uint_fast64_t end = intervals[rank + 1];
        const uint_fast64_t size = end - start + (!is_left) + (!is_right);

        // std::printf("rank %d, start %lu, end %lu, size %lu\n", rank, start, end, size);

        std::vector<double> u(size);

        MPI_Scatterv(u_0.data(), sizes.data(), intervals.data(), MPI_DOUBLE, u.data() + 1 - is_left, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::vector<double> u_next(size);
        if (is_left) {
            u[0] = 0.0;
            u_next[0] = 0.0;
        }
        if (is_right) {
            u[size - 1] = 0.0;
            u_next[size - 1] = 0.0;
        }
        const bool is_odd = rank % 2 == 1;

        int left_rank = rank - 1;
        int right_rank = rank + 1;
        if (is_left)
            left_rank = MPI_PROC_NULL;
        if (is_right)
            right_rank = MPI_PROC_NULL;

        MPI_Request req[4];
        for (uint_fast64_t i = 0; i < n_t; i++) {
            // exchange values with neighbors
            if constexpr (alg == simple) {
                MPI_Sendrecv(&u[1], 1, MPI_DOUBLE, left_rank, 0, &u[size - 1], 1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(&u[size - 2], 1, MPI_DOUBLE, right_rank, 0, &u[0], 1, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                heat_step(u.data(), u_next.data(), kt_h2, size);
            }
            else if constexpr (alg == async) {
                // printf("aaaa\n");
                u_next[1] = u[1] + kt_h2 * (u[0] - 2 * u[1] + u[2]);
                u_next[size - 2] = u[size - 2] + kt_h2 * (u[size - 3] - 2 * u[size - 2] + u[size - 1]);
                MPI_Isend(&u_next[1], 1, MPI_DOUBLE, left_rank, rank, MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&u_next[size - 2], 1, MPI_DOUBLE, right_rank, rank, MPI_COMM_WORLD, &req[1]);

                MPI_Irecv(&u_next[0], 1, MPI_DOUBLE, left_rank, left_rank, MPI_COMM_WORLD, &req[2]);
                MPI_Irecv(&u_next[size - 1], 1, MPI_DOUBLE, right_rank, right_rank, MPI_COMM_WORLD, &req[3]);

                heat_step(&u[1], &u_next[1], kt_h2, size - 2);

                MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
                // MPI_Recv(&u_next[0], 1, MPI_DOUBLE, left_rank, left_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // MPI_Recv(&u_next[size - 1], 1, MPI_DOUBLE, right_rank, right_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            u.swap(u_next);
        }

        // collect results from all processes
        if (rank == 0)
            res.resize(N);

        MPI_Gatherv(u.data() + 1 - is_left, sizes[rank], MPI_DOUBLE, res.data(), sizes.data(), intervals.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    return res;
}