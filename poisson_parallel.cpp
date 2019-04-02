/* Write a program that solves the Poisson problem with P processes using the fast diagolization method described in the lecture notes/*

/**
 * Rewriting the C program given.
 * C program to solve the two-dimensional Poisson equation on
 * a unit square using one-dimensional eigenvalue decompositions
 * and fast sine transforms.
 *
 * Einar M. Rønquist
 * NTNU, October 2000
 * Revised, October 2001
 * Revised by Eivind Fonn, February 2015
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cmath>


#define PI 3.14159265358979323846
#define true 1
#define false 0

typedef double double;
typedef int bool;

using namespace std;

// Function prototypes
double *mk_1D_array(size_t n, bool zero);
double **mk_2D_array(size_t n1, size_t n2, bool zero);
void transpose(double **bt, double **b, size_t m);
double rhs(double x, double y);

extern "C" {
    // Functions implemented in FORTRAN in fst.f and called from C.
    // The trailing underscore comes from a convention for symbol names, called name
    // mangling: if can differ with compilers.
    void fst_(double *v, int *n, double *w, int *nn);
    void fstinv_(double *v, int *n, double *w, int *nn);

}



int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Usage:\n");
        printf("  poisson n\n\n");
        printf("Arguments:\n");
        printf("  n: the problem size (must be a power of 2)\n");
        return 1;
    }

    /*
     *  The equation is solved on a 2D structured grid and homogeneous Dirichlet
     *  conditions are applied on the boundary:
     *  - the number of grid points in each direction is n+1,
     *  - the number of degrees of freedom in each direction is m = n-1,
     *  - the mesh size is constant h = 1/n.
     */
    int n = atoi(argv[1]);
    if ((n & (n-1)) != 0) {
      printf("n must be a power-of-two\n");
      return 2;
    }

    int m = n - 1;
    double h = 1.0 / n;

    // The MPI section starts here 
    int rank, P, t // myrank, number of mpi mprocesses and number of threads
    double time_start;
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // Assure that n > P

    if (n-1<P) {
        cout<< "n-1 < P! This leads to some processes receiving zero rows."<<endl;
        return 0;
    }

    if (rank == 0) {
        // Start timing
        time_start = MPI_Wtime();
    }

    // Distribute a block of rows to each process
    // Need to have control of global and local positions (?)

    int *block_size = (int*) calloc(P,sizeof(int)); // block size
    int *block_size_sum = (int*) calloc(P, sizof(int)); // sum of all block sizes i = 0, ..., i=i-1 control mechanism
    int *counts = (int*) calloc(P,sizeof(int)); // TODO: Explain
    int *displs = (int*) calloc(P,sizeof(int)); // TODO: Explain


    block_size[0] = ceil(m/P);
    block_size_sum[0] = 0;
    int rows_taken = 0;

    // Distribute using openmp
    #pragma omp parallel for
        for (size_t i=1; i<P; i++){
            rows_taken += block_size[i-1];
            block_size = ceil((m - rows_taken )/(P-1));
            block_size_sum[i] = rows_taken;
            }



    displs[0] = 0;
    counts[0] = block_size[rank]*block_size[0];
    int sum_counts = 0;

    #pragma omp parallel for
        for (size_t i=1; i<P; i++){
            sum_counts += counts[i-1];
            counts[i] = block_size[rank]*block_size[i]; // Num elements in block i = my number of rows*n_cols ????
            displs[i] = sum_counts; // Global position?
        }

    // Check that all m rows are distributed:
    if (rank == 0){
        cout<< "Check that all m rows are distributed:"<<endl;
        int rows_dist = accumulate(block_size.begin(), block_size.end(),0); // 0 is the initial value of the sum
        if (rows_dist != n){
            cout<<"rows distributed = "<<rows_dist<<" != m:"<<m<<endl;
        }
    }



    /*
     * Grid points are generated with constant mesh size on both x- and y-axis.
     */
    double *grid = mk_1D_array(n+1, false);
    #pragma omp parallel for
        for (size_t i = 0; i < n+1; i++) {
            grid[i] = i * h;
        }


    // Let the last processor check the generated mesh
    if (rank == P-1){
        if (block_size_sum[rank] + block_size[rank] < m){
            cout<<"Something is wrong. The whole domain is not distributed."<<endl;
        }
        if(block_size_sum[rank] + block_size[rank] > m){
            cout<<"Something is wrong. The domian is exceeded."<<endl;
        }
    }





    /*
     * The diagonal of the eigenvalue matrix of T is set with the eigenvalues
     * defined Chapter 9. page 93 of the Lecture Notes.
     * Note that the indexing starts from zero here, thus i+1.
     */
    double *diag = mk_1D_array(m, false);
    #pragma omp parallel for
        for (size_t i = 0; i < m; i++) {
            diag[i] = 2.0 * (1.0 - cos((i+1) * PI / n));
        }

    /*
     * Allocate the matrices b and bt which will be used for storing value of
     * G, \tilde G^T, \tilde U^T, U as described in Chapter 9. page 101.
     */
    double **b = mk_2D_array(block_size[rank], m, false);
    double **bt = mk_2D_array(block_size[rank], m, false);

    /*
     * This vector will holds coefficients of the Discrete Sine Transform (DST)
     * but also of the Fast Fourier Transform used in the FORTRAN code.
     * The storage size is set to nn = 4 * n, look at Chapter 9. pages 98-100:
     * - Fourier coefficients are complex so storage is used for the double part
     *   and the imaginary part.
     * - Fourier coefficients are defined for j = [[ - (n-1), + (n-1) ]] while 
     *   DST coefficients are defined for j [[ 0, n-1 ]].
     * As explained in the Lecture notes coefficients for positive j are stored
     * first.
     * The array is allocated once and passed as arguments to avoid doings 
     * doublelocations at each function call.
     */
    int nn = 4 * n;

    /*
     * Initialize the right hand side data for a given rhs function.
     * 
     */

    // Collapse the loop into one large iteration space    
    // Do I need a pragma omp parallel outside?

    #pragma omp for collapse(2)
        for (size_t i = 0; i < block_sizej[rank]; i++) {
            for (size_t j = 0; j < m; j++) {
                b[i][j] = h * h * rhs(grid[block_size_sum[rank]+i], grid[j+1]);
            }
        }



    /*
     * Compute \tilde G^T = S^-1 * (S * G)^T (Chapter 9. page 101 step 1)
     * Instead of using two matrix-matrix products the Discrete Sine Transform
     * (DST) is used.
     * The DST code is implemented in FORTRAN in fst.f and can be called from C.
     * The array zz is used as storage for DST coefficients and internally for 
     * FFT coefficients in fst_ and fstinv_.
     * In functions fst_ and fst_inv_ coefficients are written back to the input 
     * array (first argument) so that the initial values are overwritten.
     */


    double *block_vec_b = mk_1D_array(block_size[rank]*m, false);
    double *block_vec_b_pre_transp = mk_1D_array(block_size[rank]*m, false);
    double *block_vec_bt= mk_1D_array(block_size[rank]*m, false);


    double u_max = 0.0;

    // Create a parallel section that lasts until the end of main
    #pragma omp parallel{
    double *z = mk_1D_array(nn, false);

    // Find fst of b
    for (size_t i = 0; i < block_size[rank]; i++) {
        fst_(b[i], &n, z, &nn);
    }
    
    // Reorder b from row-by-row to subrow-by-subrow
    #pragma omp for collapse(2)
    for (size_t i = 0; i < block_size[rank]; i++){
        for (size_t j = 0; j<P; j++){
            for (size_t k = block_size_sum[j]; k<block_size_sum[j] + block_size[j]; k++){
                block_vec_b[ displs[j] + i*block_size[j] + k] = b[i][k];
            }
        }
    }

    // Broadcast data
    #pragma omp master 
    {
        MPI_Alltoallv(block_vec_b,counts,displs,MPI_DOUBLE, block_vec_b_pre_transp, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    }

    // Transpose block wise
    for (size_t i = 0; i < P; i++){
        int M = counts[i]/block_size[i]; // Rows of orig block
        #pragma omp for collapse(2)
        for (size_t j = 0; j<block_size[i]; j++){
            for (size_t k =0; k<M; k++){
                block_vec_bt[ displs[i] + k*block_size[i] + j] = block_vec_b_pre_transp[displs[i] + M*j + k];
            }
        }
    }

    // Reorder back from vector to matrix bt
    #pragma omp for collapse(2)
    for (size_t i = 0; i < block_size[rank]; i++){
        for (size_t j = 0; j<P; j++){
            for (size_t k = block_size_sum[j]; k<block_size_sum[j] + block_size[j]; k++){
                bt[i][k] = block_vec_bt[ displs[j] + i*block_size[j] + k];
            }
        }
    }    


    //transpose(bt, b, m);
    // Apply fstinv on bt
    for (size_t i = 0; i < block_size[rank]; i++) {
        fstinv_(bt[i], &n, z, &nn);
    }

    /*
     * Solve Lambda * \tilde U = \tilde G (Chapter 9. page 101 step 2)
     */
    #pragma omp for collapse(2)
    for (size_t i = 0; i < block_size[rank]; i++) {
        for (size_t j = 0; j < m; j++) {
            bt[i][j] = bt[i][j] / (diag[block_size_sum[rank] + i] + diag[j]);
        }
    }

    /*
     * Compute U = S^-1 * (S * Utilde^T) (Chapter 9. page 101 step 3)
     */
    
    // Apply fst on bt
    for (size_t i = 0; i < block_size[rank]; i++) {
        fst_(bt[i], &n, z, &nn);
    }

    //Do the transpose procedure again.
    //transpose(b, bt, m);


    // Reorder bt from row-by-row to subrow-by-subrow
    #pragma omp for collapse(2)
    for (size_t i = 0; i < block_size[rank]; i++){
        for (size_t j = 0; j<P; j++){
            for (size_t k = block_size_sum[j]; k<block_size_sum[j] + block_size[j]; k++){
                block_vec_bt[ displs[j] + i*block_size[j] + k] = bt[i][k];
            }
        }
    }

    // Broadcast data
    #pragma omp master 
    {
        MPI_Alltoallv(block_vec_bt,counts,displs,MPI_DOUBLE, block_vec_b_pre_transp, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    // Transpose block wise
    for (size_t i = 0; i < P; i++){
        int M = counts[i]/block_size[i]; // Rows of orig block
        #pragma omp for collapse(2)
        for (size_t j = 0; j<block_size[i]; j++){
            for (size_t k =0; k<M; k++){
                block_vec_b[ displs[i] + k*block_size[i] + j] = block_vec_b_pre_transp[displs[i] + M*j + k];
            }
        }
    }

    // Reorder back from vector to matrix
    #pragma omp for collapse(2)
    for (size_t i = 0; i < block_size[rank]; i++){
        for (size_t j = 0; j<P; j++){
            for (size_t k = block_size_sum[j]; k<block_size_sum[j] + block_size[j]; k++){
                b[i][k] = block_vec_b[ displs[j] + i*block_size[j] + k];
            }
        }
    }    
    // Transpose finished

    // Apply fstinv on b
    for (size_t i = 0; i < block_size[rank]; i++) {
        fstinv_(b[i], &n, z, &nn);
    }

    /*
     * Compute maximal value of solution for convergence analysis in L_\infty
     * norm.
     */
    #pragma omp for reduction(max:u_max)collapse(2)
    for (size_t i = 0; i < block_size[rank]; i++) {
        for (size_t j = 0; j < m; j++) {
            // Write the if sentence more readable
            if (u_max <= fabs(b[i][j])){
                u_max = fabs(b[i][j]);
            }
            //u_max = u_max > fabs(b[i][j]) ? u_max : fabs(b[i][j]);
        }
    }


    MPI_Reduce(&u_max, &u_max_all, 1 MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);




    if (rank == 0){
        #pragma omp master
        {
            double time_used = MPI_Wtime() - time_start;
            printf("for n = %i, P=%i and t = %i we get: \n time: %e\n u_max: %e\n", n, P, t, time_used, u_max_all);
        }
    }

    //printf("u_max = %e\n", u_max);

    } // end pragma



    // Release memory
    free(displs);
    free(counts);
    free(block_size);
    free(block_size_sum);
    free(block_vec_b);
    free(block_vec_b_pre_transp);
    free(block_vec_bt);
    

    MPI_Finalize();

    return 0;
} // end main

/*
 * This function is used for initializing the right-hand side of the equation.
 * Other functions can be defined to swtich between problem definitions.
 */

double rhs(double x, double y) {
    return 1;
}

/*
 * Write the transpose of b a matrix of R^(m*m) in bt.
 * In parallel the function MPI_Alltoallv is used to map directly the entries
 * stored in the array to the block structure, using displacement arrays.
 */

void transpose(double **bt, double **b, size_t m)
{
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            bt[i][j] = b[j][i];
        }
    }
}

/*
 * The allocation of a vectore of size n is done with just allocating an array.
 * The only thing to notice here is the use of calloc to zero the array.
 */

double *mk_1D_array(size_t n, bool zero)
{
    if (zero) {
        return (double *)calloc(n, sizeof(double));
    }
    return (double *)malloc(n * sizeof(double));
}

/*
 * The allocation of the two-dimensional array used for storing matrices is done
 * in the following way for a matrix in R^(n1*n2):
 * 1. an array of pointers is allocated, one pointer for each row,
 * 2. a 'flat' array of size n1*n2 is allocated to ensure that the memory space
 *   is contigusous,
 * 3. pointers are set for each row to the address of first element.
 */

double **mk_2D_array(size_t n1, size_t n2, bool zero)
{
    // 1
    double **ret = (double **)malloc(n1 * sizeof(double *));

    // 2
    if (zero) {
        ret[0] = (double *)calloc(n1 * n2, sizeof(double));
    }
    else {
        ret[0] = (double *)malloc(n1 * n2 * sizeof(double));
    }
    
    // 3
    for (size_t i = 1; i < n1; i++) {
        ret[i] = ret[i-1] + n2;
    }
    return ret;
}

