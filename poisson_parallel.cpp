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
    if ((n & (n-1)) != 0 && n!=5) {
      printf("n must be a power-of-two\n");
      return 2;
    }



    int m = n - 1;
    double h = 1.0 / n;

    // The MPI section starts here 
    int rank, P, t; // myrank, number of mpi mprocesses and number of threads
    double time_start;
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&P);

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

    int *block_size = (int*) calloc(P,sizeof(int)); // block sizes, i.e. num rows
    int *block_size_sum = (int*) calloc(P, sizeof(int)); // sum of all block sizes i = 0, ..., i=i-1 control mechanism
    int *counts = (int*) calloc(P,sizeof(int)); // n elements in block ?
    int *displs = (int*) calloc(P,sizeof(int)); // global position


    //cout<<"TEST:"<<endl;
    //cout<<int(ceil(double(2)/1))<<endl;
    //cout<<"TEST over"<<endl;
    

    // Let's let the first one be rounded down
    block_size[0] = m/P;//int(ceil(double(m)/P));
    //cout << "m,P, ceil(m/P)"<< m << P << block_size[0]<<endl;
    block_size_sum[0] = 0;
    int rows_taken = 0;

    // Distribute using openmp
    // SOMETHING ODD IS HAPPENING HERE
    // That is because the loops are NOT independent!
    // #pragma omp parallel for
    for (size_t i=1; i<P; i++){
        rows_taken += block_size[i-1];
        block_size[i] = int(ceil(double(m - rows_taken )/(P-i)));
        block_size_sum[i] = rows_taken;
    }



    displs[0] = 0;
    counts[0] = block_size[rank]*block_size[0];
    int sum_counts = 0;
    
    cout<<"rank="<<rank<<endl;
    cout<<"blocksize="<<block_size[rank]<<endl;
    cout<<"counts og displs:\n";
    cout<<counts[0]<<" "<<displs[0]<<endl;
    

    //#pragma omp parallel for
    // remove parallel for as sum_counts is not independent
    for (size_t i=1; i<P; i++){
        sum_counts += counts[i-1];
        counts[i] = block_size[rank]*block_size[i]; // Num elements in block i = my number of rows*n_cols ????
        displs[i] = sum_counts; // Global position?
        cout<<counts[i]<<" "<<displs[i]<<endl;
    }

    // Check that all m rows are distributed:
    cout<< "Check that all m rows are distributed:"<<endl;
    cout<<"Show blocksizes  and sums"<<endl;
    for (int i = 0; i<P;i++){
        cout<<"i="<<i<<" block_size[i]="<<block_size[i]<<" block_size_sum[i]"<<block_size_sum[i]<<endl;
    }
    if (rank == 0){        
        valarray<int> my_block_sizes (block_size, P);
        int rows_dist = my_block_sizes.sum();
        if (rows_dist != m){
            cout<<"rows distributed = "<<rows_dist<<" != m:"<<m<<endl;
        }
        else{
            cout<<"OK"<<endl;
        }
    }



    /*
     * Grid points are generated with constant mesh size on both x- and y-axis.
     * TODO: IS THIS CORRECT?
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
     * CORRECT?????
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
    double u_max = 0.0;



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
                b[i][j] = h * h * rhs(grid[block_size_sum[rank]+i], grid[j]);
            }
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


    //////////////////////////////////////////////////////////////////////


    /* 
    if (n==5){
        double ** my_test_matrix = mk_2D_array(4, 4, false);
        double ** my_test_matrix_transpose = mk_2D_array(4, 4, false);

        my_test_matrix[0][0] = 1; 
        my_test_matrix[0][1] = 2;
        my_test_matrix[0][2] = 3;
        my_test_matrix[0][3] = 4;
        my_test_matrix[1][0] = 5; 
        my_test_matrix[1][1] = 6;
        my_test_matrix[1][2] = 7;
        my_test_matrix[1][3] = 8; 
        my_test_matrix[2][0] = 1;
        my_test_matrix[2][1] = 2;
        my_test_matrix[2][2] = 3;
        my_test_matrix[2][3] = 4;
        my_test_matrix[3][0] = 5;
        my_test_matrix[3][1] = 6;
        my_test_matrix[3][2] = 7;
        my_test_matrix[3][3] = 8;
        cout<<"Print test matrix:\n";
        for (int i=0; i<4;i++){
            for (int j=0; j<4;j++){
                cout<<my_test_matrix[i][j]<<" ";
            }
            cout<<endl;
        }

        double *block_vec_b = mk_1D_array(block_size[rank]*m, false);
        double *block_vec_b_pre_transp = mk_1D_array(block_size[rank]*m, false);
        double *block_vec_bt= mk_1D_array(block_size[rank]*m, false);
    
    
    
    
        // Reorder b from row-by-row to subrow-by-subrow
        #pragma omp for collapse(2)
        for (size_t i = 0; i < block_size[rank]; i++){
            for (size_t j = 0; j<P; j++){
                for (size_t k = block_size_sum[j]; k<block_size_sum[j] + block_size[j]; k++){
                    block_vec_b[ displs[j] + i*block_size[j] + k] = my_test_matrix[i][k];
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
                    my_test_matrix_transpose[i][k] = block_vec_bt[ displs[j] + i*block_size[j] + k];
                }
            }
        }    

        cout<<"Print test matrix's transpose:\n";
        for (int i=0; i<4;i++){
            for (int j=0; j<4;j++){
                cout<<my_test_matrix_transpose[i][j]<<" ";
            }
            cout<<endl;
        }

        free(block_vec_b);
        free(block_vec_bt);
        free(block_vec_b_pre_transp);

    }
    */
/////////////////////////////////////////////////////////////////////


    // Initialize b for debugging purposes.
    int counter = 0;
    if (rank == 1){
        counter = 100;
    }
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


    double *block_vec_b = mk_1D_array(block_size[rank]*m, false);
    double *block_vec_b_pre_transp = mk_1D_array(block_size[rank]*m, false);
    double *block_vec_bt= mk_1D_array(block_size[rank]*m, false);


    // Create a parallel section that lasts until the end of main
    #pragma omp parallel
    {
        t = omp_get_num_threads();
        double *z = mk_1D_array(nn, false);

/*
        // Find fst of b
        cout<<"Before fst"<<endl;
        for (size_t i = 0; i < block_size[rank]; i++) {
            fst_(b[i], &n, z, &nn);
        }
        cout<<"After fst"<<endl;
  */      
    int ind_k;
        // Reorder b from row-by-row to subrow-by-subrow
        #pragma omp for collapse(2)
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
        cout<<"Print blockvec b"<<endl;
        for(int i=0; i<block_size[rank]*m; i++){
            cout<<block_vec_b[i]<<" ";
        }
        cout<<endl;


        cout<<"First alltoall\n";
        // Broadcast data
        #pragma omp master 
        {
            MPI_Alltoallv(block_vec_b,counts,displs,MPI_DOUBLE, block_vec_b_pre_transp, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        }
        cout<<"After first alltoall\n";
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
            //#pragma omp for collapse(2)
            for (size_t j = 0; j<block_size[i]; j++){
                for (size_t k =0; k<M; k++){
                    block_vec_bt[ displs[i] + k*block_size[i] + j] = block_vec_b_pre_transp[displs[i] + M*j + k];
                    // block_vec_bt[ displs[i] + k*block_size[i] + j] = block_vec_b_pre_transp[displs[i] + M*j + k];
                    cout<<"pos:"<<displs[i] + k*block_size[i] + j<<" displs[i] + M*j + k]"<<displs[i] + M*j + k<<endl;
                }
            }
        }

        cout<<"Print blockvec b_t"<<endl;
        for(int i=0; i<block_size[rank]*m; i++){
            cout<<block_vec_bt[i]<<" ";
        }
        cout<<endl;

        

        // Reorder back from vector to matrix bt
        //#pragma omp for collapse(2)
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
        
        cout<<"b transposed"<<endl;
        for( int i =0; i<block_size[rank]; i++){
            for (int j = 0; j<m; j++){
                cout<<bt[i][j]<<" ";
            }
            cout<<endl;
        }
        
        /*
        //transpose(bt, b, m);
        // Apply fstinv on bt
        for (size_t i = 0; i < block_size[rank]; i++) {
            fstinv_(bt[i], &n, z, &nn);
        }

        /*
        * Solve Lambda * \tilde U = \tilde G (Chapter 9. page 101 step 2)
        */

    /*
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
        /*
        for (size_t i = 0; i < block_size[rank]; i++) {
            fst_(bt[i], &n, z, &nn);
        }

        //Do the transpose procedure again.
        //transpose(b, bt, m);

    /*
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

    /*
        double u_max_all;

        #pragma omp for reduction(max:u_max) collapse(2)
        for (size_t i = 0; i < block_size[rank]; i++) {
            for (size_t j = 0; j < m; j++) {
                // Write the if sentence more readable
                if (u_max <= fabs(b[i][j])){
                    u_max = fabs(b[i][j]);
                }
                //u_max = u_max > fabs(b[i][j]) ? u_max : fabs(b[i][j]);
            }
        }

        cout<<"Hey there00\n";

        MPI_Reduce(&u_max, &u_max_all, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        cout<<"Hey there0\n";

        if (rank == 0){
            #pragma omp master
            {
                cout<<"Hey there\n";
                double time_used = MPI_Wtime() - time_start;
                printf("for n = %i, P=%i and t = %i we get: \n time: %e\n u_max: %e\n", n, P, t, time_used, u_max_all);
            }
        }
        */
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
    
    free(displs);
    free(counts);
    free(block_size);
    free(block_size_sum);
    */
    
    
    

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
