#include <iostream>
#include <string>

#include "./source/mpi_heat.hpp"


int main(int argc, char** argv) {



    // check mpi
    const double k = 1.0;
    const double u_0 = 1.0;
    double T = 1e-4;
    uint_fast64_t n_t = 100000L;
    uint_fast64_t n_h = 100L;
    double time_simple = 0.0;
    double time_async = 0.0;

    switch (argc)
    {
    case 4:
        T = std::stod(argv[3]);
    case 3:
        n_t = std::stoi(argv[2]);
    case 2:
        n_h = std::stoi(argv[1]) - 1;

    default:
        break;
    }


    MPI_Init(NULL, NULL);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> u_init;
    if (rank == 0) {
        u_init = std::vector<double>(n_h + 1, u_0);
    }

    time_simple = MPI_Wtime();
    const std::vector<double>& res_simple = solve_heat_mpi<simple>(
        T,
        k,
        n_h,
        n_t,
        u_init
    );
    time_simple = MPI_Wtime() - time_simple;
    time_async = MPI_Wtime();
    const std::vector<double>& res_async = solve_heat_mpi<async>(
        T,
        k,
        n_h,
        n_t,
        u_init
    );
    time_async = MPI_Wtime() - time_async;


    if (rank == 0) {
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        const std::vector<double>& exact_solution = exact_solution_vector(T, k, u_0, n_h);

        double simple_er = vector_err(exact_solution, res_simple);

        double async_er = vector_err(exact_solution, res_async);

        printf(" %10s | %10s | %10s | %10s | %10s | %10s \n", "N_tasks", "N_points", "N_t", "name", "time_sec", "max_err");
        printf(" %10i | %10lu | %10lu | %10s | %10lf | %10lf \n", world_size, n_h + 1, n_t, "simple", time_simple, simple_er);
        printf(" %10i | %10lu | %10lu | %10s | %10lf | %10lf \n", world_size, n_h + 1, n_t, "async", time_async, async_er);
    }
    MPI_Finalize();
    return 0;
}