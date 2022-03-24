//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/wathen_demo.c: test wathen
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Construct a matrix using the Demo/Source/wathen.c method.
//
//  wathen_demo nx ny method nthreads

// macro used by OK(...) to free workspace if an error occurs
#define FREE_ALL            \
    GrB_Matrix_free (&A) ;  \

#include "graphblas_demos.h"
#ifdef _OPENMP
#include "omp.h"
#endif

int main (int argc, char **argv)
{
    GrB_Matrix A = NULL ;
    GrB_Info info ;
    OK (GrB_init (GrB_NONBLOCKING)) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    int64_t nx = 10, ny = 10 ;
    int method = 0 ;
    int nthreads ;
    if (argc > 1) nx = strtol (argv [1], NULL, 0) ;
    if (argc > 2) ny = strtol (argv [2], NULL, 0) ;
    if (argc > 3) method = strtol (argv [3], NULL, 0) ;
    if (argc > 4)
    {
        nthreads = strtol (argv [4], NULL, 0) ;
        OK (GxB_Global_Option_set (GxB_GLOBAL_NTHREADS, nthreads)) ;
    }
    OK (GxB_Global_Option_get (GxB_GLOBAL_NTHREADS, &nthreads)) ;

    fprintf (stderr, "Wathen: nx %ld ny %ld method: %d nthreads: %d ",
        nx, ny, method, nthreads) ;

    //--------------------------------------------------------------------------
    // create a Wathen matrix
    //--------------------------------------------------------------------------

    #ifdef _OPENMP
    double t = omp_get_wtime ( ) ;
    #endif
    OK (wathen (&A, nx, ny, false, method, NULL)) ;
    #ifdef _OPENMP
    t = omp_get_wtime ( ) - t ;
    fprintf (stderr, "time: %g", t) ;
    #endif
    fprintf (stderr, "\n") ;

    OK (GxB_print (A, GxB_SUMMARY)) ;

    FREE_ALL ;

    //--------------------------------------------------------------------------
    // benchmark wathen matrices
    //--------------------------------------------------------------------------

    GrB_Descriptor Desc_Gustavson = NULL, Desc_Hash = NULL ;
    GrB_Descriptor_new (&Desc_Gustavson) ;
    GrB_Descriptor_new (&Desc_Hash) ;
    GxB_Desc_set (Desc_Gustavson, GxB_AxB_METHOD, GxB_AxB_GUSTAVSON) ;
    GxB_Desc_set (Desc_Hash     , GxB_AxB_METHOD, GxB_AxB_HASH     ) ;

    for (nx = 100 ; nx <= 4000 ; nx += 100)
    {
        printf ("\n ------------------- nx %ld\n", nx) ;
        GxB_set (GxB_NTHREADS, 40) ;

        // create the wathen matrix
        t = omp_get_wtime ( ) ;
        OK (wathen (&A, nx, nx, false, 0, NULL)) ;
        t = omp_get_wtime ( ) - t ;
        GrB_Index nvals1, nvals2, n ;
        OK (GrB_Matrix_nrows (&n, A)) ;
        OK (GrB_Matrix_nvals (&nvals1, A)) ;
        double t2_sequential, t4_sequential, t8_sequential ;

        printf ("\n=================================\n"
            "nx %5ld n %10.3fM nvals %10.3fM create: %10.3f sec\n",
            nx, (double) n / 1e6, (double) nvals1 / 1e6, t) ;

        int Nthreads [7] = {1, 2, 4, 8, 16, 20, 40} ;
        for (int k = 0 ; k < 7 ; k++)
        {
            // set the # of threads to use
            int nth = Nthreads [k] ;
            GxB_set (GxB_NTHREADS, nth) ;

            // for (int algo = 0 ; algo <= 2 ; algo++)
            for (int algo = 0 ; algo <= 1 ; algo++)
            {
                GrB_Descriptor desc = NULL ;
                if (algo == 1) desc = Desc_Gustavson ;
                if (algo == 2) desc = Desc_Hash ;

                GrB_Matrix C = NULL ;
                OK (GrB_Matrix_new (&C, GrB_FP64, n, n)) ;
                char *algo_name = (algo == 0) ? "Auto" :
                    ((algo == 1) ? "Gustavson" : "Hash") ;
                printf ("\nalgo: %s nthreads: %d\n", algo_name, nth) ;
                // if (nth == 2 || nth == 40)
                GxB_set (GxB_BURBLE, true) ;

                // square it: C = A*A
                double t2 = omp_get_wtime ( ) ;
                OK (GrB_mxm (C, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64,
                    A, A, desc)) ;
                t2 = omp_get_wtime ( ) - t2 ;
                GxB_set (GxB_BURBLE, false) ;
                OK (GrB_Matrix_nvals (&nvals2, C)) ;
                GxB_set (GxB_BURBLE, true) ;
                printf ("C=A^2 threads: %2d mxm: %10.3f nvals %10.3fM ",
                    nth, t2, ((double) nvals2) / 1e6) ;
                if (nth == 1) t2_sequential = t2 ;
                printf ("speedup: %g\n", t2_sequential/t2) ;

                // square it again: C = C*C to get A^4
                // if (nx <= 4000)
                {
                    double t4 = omp_get_wtime ( ) ;
                    OK (GrB_mxm (C, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64,
                        C, C, desc)) ;
                    t4 = omp_get_wtime ( ) - t4 ;
                    GxB_set (GxB_BURBLE, false) ;
                    OK (GrB_Matrix_nvals (&nvals2, C)) ;
                    GxB_set (GxB_BURBLE, true) ;
                    printf ("C=A^4 threads: %2d mxm: %10.3f nvals %10.3fM ",
                        nth, t4, ((double) nvals2) / 1e6) ;
                    if (nth == 1) t4_sequential = t4 ;
                    printf ("speedup: %g\n", t4_sequential/t4) ;
                }

                // square it again: C = C*C to get A^8
                // if (nx <= 1000)
                {
                    double t8 = omp_get_wtime ( ) ;
                    OK (GrB_mxm (C, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64,
                        C, C, desc)) ;
                    t8 = omp_get_wtime ( ) - t8 ;
                    GxB_set (GxB_BURBLE, false) ;
                    OK (GrB_Matrix_nvals (&nvals2, C)) ;
                    GxB_set (GxB_BURBLE, true) ;
                    printf ("C=A^8 threads: %2d mxm: %10.3f nvals %10.3fM ",
                        nth, t8, ((double) nvals2) / 1e6) ;
                    if (nth == 1) t8_sequential = t8 ;
                    printf ("speedup: %g\n", t8_sequential/t8) ;
                }

                GxB_set (GxB_BURBLE, false) ;
                GrB_Matrix_free (&C) ;
            }
        }
        GrB_Matrix_free (&A) ;
    }

    GrB_free (&Desc_Gustavson) ;
    GrB_free (&Desc_Hash) ;

    OK (GrB_finalize ( )) ;
}

