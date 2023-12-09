#include <iostream>
#include <string>

#include "./source/mpi_heat.hpp"


int main(int argc, char** argv) {



    // check mpi
    const double k = 1.0;
    const double T = 1e-4;
    const double u_0 = 1.0;
    const uint_fast64_t n_t = T / 1e-9;
    const uint_fast64_t n_h = 10000;

    MPI_Init(NULL, NULL);


    printf("\nCheck\n");

    const std::vector<double>& res_simple = solve_heat_mpi<simple>(
        T,
        k,
        n_h,
        n_t,
        1.0
    );
    printf("\nCheck\n");
    const std::vector<double>& res_async = solve_heat_mpi<async>(
        T,
        k,
        n_h,
        n_t,
        1.0
    );
    printf("\nCheck\n");

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("\nCheck\n");

        const std::vector<double>& exact_solution = exact_solution_vector(T, k, u_0, n_h);

        printf("\nResult simple err: %f\n", vector_err(exact_solution, res_simple));

        printf("\nResult async err: %f\n", vector_err(exact_solution, res_async));

        // print_vector(exact_solution);
        // printf("\n");
        // print_vector(res_simple);
    }
    MPI_Finalize();
    return 0;
}