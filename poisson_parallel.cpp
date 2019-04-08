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
//#include <numeric>
#include <valarray>
#include <fstream>


#define PI 3.14159265358979323846
#define true 1
#define false 0

using namespace std;

// Function prototypes
double *mk_1D_array(size_t n, bool zero);
double **mk_2D_array(size_t n1, size_t n2, bool zero);
void transpose(double **bt, double **b, size_t m);
double rhs(double x, double y);
double rhs_alternative(double x, double y);
double u_analytical(double x, double y);


extern "C" {
    // Functions implemented in FORTRAN in fst.f and called from C.
    // The trailing underscore comes from a convention for symbol names, called name
    // mangling: if can differ with compilers.
    void fst_(double *v, int *n, double *w, int *nn);
    void fstinv_(double *v, int *n, double *w, int *nn);

}



int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage:\n");
        printf("  poisson n\n\n");
        printf("Arguments:\n");
        printf("  n: the problem size (must be a power of 2)\n");
        printf(" nthreads: number of openmp threads\n");
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
    int t = atoi(argv[2]); //number of threads
    if (t<=0){
        printf("t is not a positive integer\n");
        return 3;
    }


    int m = n - 1;
    double h = 1.0 / n;

    int rank, P; // rank, number of mpi mprocesses
    double time_start;
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&P);

    // Assure that m> P

    if (n-1<P) {
        cout<< "n-1 < P! This leads to some processes receiving zero rows."<<endl;
        return 4;
    }

    if (rank == 0) {
        // Start timing
        time_start = MPI_Wtime();
    }

    // Distribute a block of rows to each process
    int *block_size = (int*) calloc(P,sizeof(int)); // block sizes, i.e. num rows
    int *block_size_sum = (int*) calloc(P, sizeof(int)); // sum of all block sizes i = 0, ..., i=i-1 
    int *counts = (int*) calloc(P,sizeof(int)); // n elements in block
    int *displs = (int*) calloc(P,sizeof(int)); // needed for block sending, displacement from start of sent vector

    // Calculate block sizes and blocksizesums
    // Let's let the first block size be rounded down
    block_size[0] = m/P;
    block_size_sum[0] = 0;
    int rows_taken = 0;
    for (size_t i=1; i<P; i++){
        rows_taken += block_size[i-1];
        block_size[i] = int(ceil(double(m - rows_taken )/(P-i)));
        block_size_sum[i] = rows_taken;
    }

    // Calculate counts and displacements
    displs[0] = 0;
    counts[0] = block_size[rank]*block_size[0];
    int sum_counts = 0;

    for (size_t i=1; i<P; i++){
        sum_counts += counts[i-1];
        counts[i] = block_size[rank]*block_size[i]; 
        displs[i] = sum_counts;
    }

    if (rank == 0){        
        valarray<int> my_block_sizes (block_size, P);
        int rows_dist = my_block_sizes.sum();
        // cout<< "Check that all m rows are distributed:"<<endl;

        if (rows_dist != m){
            cout<<"rows distributed = "<<rows_dist<<" != m:"<<m<<endl;
        }
        else{
           // cout<<"OK"<<endl; Removed when running on Idun
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
    double *z[t]; // create a z vector for each thread in this process
    double u_max = 0.0; // for holding the max value of the solution u
    double e_max = 0.0; // verification test

    /*
     * Initialize the right hand side data for a given rhs function.
     * 
     */

    // Collapse the loop into one large iteration space     
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (size_t i = 0; i < block_size[rank]; i++) {
            for (size_t j = 0; j < m; j++) {
                b[i][j] = h * h * rhs_alternative(grid[block_size_sum[rank]+i+1], grid[j+1]);
            }
        }
    }
/*
    for( int i =0; i<block_size[rank]; i++){
        for (int j = 0; j<m; j++){
            cout<<b[i][j]<<" ";
        }
        cout<<endl;
    }
*/
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


   
/*
    // Initialize b for debugging purposes.
    int counter = rank*100;

    for (int i =0; i<block_size[rank];i++){
        for (int j=0; j<m; j++){
            counter += 1;
            b[i][j] = counter;
        }
    }
    // Print b
    for( int i =0; i<block_size[rank]; i++){
        for (int j = 0; j<m; j++){
            cout<<b[i][j]<<" ";
        }
        cout<<endl;
    }
*/

    double *block_vec_b = mk_1D_array(block_size[rank]*m, false);
    double *block_vec_b_pre_transp = mk_1D_array(block_size[rank]*m, false);
    double *block_vec_bt= mk_1D_array(block_size[rank]*m, false);

    #pragma omp parallel for
    for (int i = 0; i<t;i++){
        z[i] = mk_1D_array(nn,false);
    }

    // Create a parallel section that lasts until the end of main
    #pragma omp parallel
    {
        //t = omp_get_num_threads();
        //t = 3;
        //omp_set_num_threads(t);
        //cout<<"We are using t="<<t<<" threads."<<endl;
        // TODO: Burde være organisert annerledes?
        //PRØVER
        //double *z = mk_1D_array(nn, false);

/*
        for (int i = 0; i<t; i++){
            cout<<&z[i]<<" "<<endl;
        }
*/

        // Find fst of b
        #pragma omp parallel for num_threads(t)
        for (size_t i = 0; i < block_size[rank]; i++) {
            fst_(b[i], &n, z[omp_get_thread_num()], &nn);
        }
        
        int ind_k;
        // Reorder b from row-by-row to subrow-by-subrow
        for (size_t i = 0; i < block_size[rank]; i++){
            for (size_t j = 0; j<P; j++){
                int ind_k = 0;
                for (size_t k = block_size_sum[j]; k<block_size_sum[j] + block_size[j]; k++){
                    //cout<<"pos:"<<displs[j] + i*block_size[j] + ind_k<<" b[i][k]"<<b[i][k]<<endl;
                    block_vec_b[ displs[j] + i*block_size[j] + ind_k] = b[i][k];
                    ind_k++;
                }
            }
        }

        /*
        cout<<"Print blockvec b"<<endl;
        for(int i=0; i<block_size[rank]*m; i++){
            cout<<block_vec_b[i]<<" ";
        }
        cout<<endl;
        */


        // Send rows to all processes
        #pragma omp master 
        {
            MPI_Alltoallv(block_vec_b,counts,displs,MPI_DOUBLE, block_vec_b_pre_transp, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        }
        /*
        cout<<"b received"<<endl;
        for( int i =0; i<block_size[rank]*m; i++){
            cout<<block_vec_b_pre_transp[i]<<" ";

        }
        cout<<endl;

        */


        
        // Transpose block wise
        for (size_t i = 0; i < P; i++){
            int M = counts[i]/block_size[i]; // Rows of orig block
            #pragma omp for collapse(2) 
            for (size_t j = 0; j<block_size[i]; j++){
                for (size_t k =0; k<M; k++){
                    block_vec_bt[ displs[i] + k*block_size[i] + j] = block_vec_b_pre_transp[displs[i] + M*j + k];
                    // block_vec_bt[ displs[i] + k*block_size[i] + j] = block_vec_b_pre_transp[displs[i] + M*j + k];
                    //cout<<"pos:"<<displs[i] + k*block_size[i] + j<<" displs[i] + M*j + k]"<<displs[i] + M*j + k<<endl;
                }
            }
        }
        /*
        cout<<"Print blockvec b_t"<<endl;
        for(int i=0; i<block_size[rank]*m; i++){
            cout<<block_vec_bt[i]<<" ";
        }
        cout<<endl;
        */
        

        // Reorder back from vector to matrix bt
        for (size_t i = 0; i < block_size[rank]; i++){
            for (size_t j = 0; j<P; j++){
                int ind_k = 0;
                for (size_t k = block_size_sum[j]; k<block_size_sum[j] + block_size[j]; k++){
                    bt[i][k] = block_vec_bt[ displs[j] + i*block_size[j] + ind_k];
                    ind_k++;
                }
            }
        }    


        // Print bt
        /*cout<<"b transposed rank"<<rank<<endl;
        for( int i =0; i<block_size[rank]; i++){
            for (int j = 0; j<m; j++){
                cout<<bt[i][j]<<" ";
            }
            cout<<endl;
        }*/
        

        // Apply fstinv on bt
        #pragma omp parallel for num_threads(t)
        for (size_t i = 0; i < block_size[rank]; i++) {
            fstinv_(bt[i], &n, z[omp_get_thread_num()], &nn);
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
        #pragma omp parallel for num_threads(t)
        for (size_t i = 0; i < block_size[rank]; i++) {
            fst_(bt[i], &n, z[omp_get_thread_num()], &nn);
        }
        //Do the transpose procedure again.

        // Reorder bt from row-by-row to subrow-by-subrow
        for (size_t i = 0; i < block_size[rank]; i++){
            for (size_t j = 0; j<P; j++){
                int ind_k = 0;
                for (size_t k = block_size_sum[j]; k<block_size_sum[j] + block_size[j]; k++){
                    block_vec_bt[ displs[j] + i*block_size[j] + ind_k] = bt[i][k];
                }
            }
        }

        
        // Send rows to all processes
        #pragma omp master 
        {
            MPI_Alltoallv(block_vec_bt, counts, displs, MPI_DOUBLE, block_vec_b_pre_transp, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
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
        for (size_t i = 0; i < block_size[rank]; i++){
            for (size_t j = 0; j<P; j++){
                int ind_k = 0;
                for (size_t k = block_size_sum[j]; k<block_size_sum[j] + block_size[j]; k++){
                    b[i][k] = block_vec_b[ displs[j] + i*block_size[j] + ind_k];
                    ind_k++;
                }
            }
        }

        // Transpose finished
        

        // Apply fstinv on b
        #pragma omp parallel for num_threads(t)
        for (size_t i = 0; i < block_size[rank]; i++) {
            fstinv_(b[i], &n, z[omp_get_thread_num()], &nn);
            cout <<"i"<<i<<" omp_get_thread_num"<<omp_get_thread_num()<<endl;

        }


        double u_max_all;
        double e_max_all;

        // cout<<"rank:"<<rank<<" block_size_sum[rank]"<< block_size_sum[rank]<<" "<<grid[block_size_sum[rank]]<<endl;
        #pragma omp for reduction(max:u_max) //collapse(2)
        for (size_t i = 0; i < block_size[rank]; i++) {
            for (size_t j = 0; j < m; j++) {
                //cout<<"grid"<<grid[i+1]<<grid[j+1]<<" ";
                if (rank == 0){
                    cout<<" b:"<<b[i][j]<<" u:"<<u_analytical(grid[block_size_sum[rank] + i + 1], grid[j + 1]);
                }
                // Stability test in infinity norm
                if (u_max <= fabs(b[i][j])){
                    u_max = fabs(b[i][j]);
                }
                // Verification test
                if (e_max <= fabs(b[i][j] - u_analytical(grid[block_size_sum[rank] + i + 1], grid[j + 1]))){
                    e_max = fabs(b[i][j] - u_analytical(grid[block_size_sum[rank] + i + 1], grid[j + 1]));
                }

            }
            cout<<endl;
        }
        #pragma omp master
        {
            MPI_Reduce(&u_max, &u_max_all, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&e_max, &e_max_all, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        }
        if (rank == 0){
            #pragma omp master
            {
                double time_used = MPI_Wtime() - time_start;
                printf("for n = %i, P=%i and t = %i we get: \n time: %e\n u_max: %e\n e_max: %e\n", n, P, t, time_used, u_max_all,e_max_all);
            }
        }
        
    } // end pragma

    // Release memory
    // Why is it not necessary to release the rest?
    //
    /*
    for (size_t i = 0; i<block_size[rank]; i++){
        free(bt[i]);
        free(b[i]);
    }
    free(block_vec_b);
    free(block_vec_bt);
    free(block_vec_b_pre_transp);
    */
    free(displs);
    free(counts);
    free(block_size);
    free(block_size_sum);
    
    

    MPI_Finalize();

    return 0;
}

/*
 * This function is used for initializing the right-hand side of the equation.
 * Other functions can be defined to swtich between problem definitions.
 */
double rhs(double x, double y) {
    return 1.0;
}

double rhs_alternative(double x, double y) {
    return 5.0*PI*PI*sin(PI*x)*sin(2.0*PI*y);
}

/*
 * This function is calculating the corresponding analytical u
 */

double u_analytical(double x, double y){
    return sin(PI*x)*sin(2.0*PI*y);
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
