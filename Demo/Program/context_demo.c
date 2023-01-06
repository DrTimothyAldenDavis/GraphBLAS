//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/context_demo: example for the GxB_Context
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GraphBLAS.h"
#include "simple_rand.h"
#define MIN(x,y) ((x) < (y)) ? (x) : (y)
#define MAX(x,y) ((x) > (y)) ? (x) : (y)
#ifdef _OPENMP
#include <omp.h>
#define TIMER omp_get_wtime ( )
#else
#define TIMER 0
#endif

#include <assert.h>
#define OK(method)                              \
{                                               \
    printf ("at %d\n", __LINE__) ;              \
    if (method != GrB_SUCCESS)                  \
    {                                           \
        printf ("abort at %d\n", __LINE__) ;    \
        abort ( ) ;                             \
    }                                           \
}

int main (void)
{
    assert (0) ;

    // start GraphBLAS
    OK (GrB_init (GrB_NONBLOCKING)) ;
    OK (GxB_print (GxB_CONTEXT_WORLD, 3)) ;

    int nthreads_max = 0 ;
    OK (GxB_Global_Option_get (GxB_GLOBAL_NTHREADS, &nthreads_max)) ;
    nthreads_max = MIN (nthreads_max, 256) ;
    printf ("context demo: nthreads_max %d\n", nthreads_max) ;

    // use only a power of 2 number of threads
    int nthreads = 1 ;
    while (1)
    {
        if (2*nthreads > nthreads_max) break ;
        nthreads = 2 * nthreads ;
    }

    printf ("nthreads to use: %d\n", nthreads) ;
    OK (GxB_Global_Option_set (GxB_GLOBAL_NTHREADS, nthreads)) ;

    #ifdef _OPENMP
    omp_set_max_active_levels (2) ;
    #endif

    //--------------------------------------------------------------------------
    // construct tuples for a decent-sized random matrix
    //--------------------------------------------------------------------------

    GrB_Index n = 100000 ;
    GrB_Index nvals = 1000000 ;
    simple_rand_seed (1) ;
    GrB_Index *I = malloc (nvals * sizeof (GrB_Index)) ;
    GrB_Index *J = malloc (nvals * sizeof (GrB_Index)) ;
    double    *X = malloc (nvals * sizeof (double)) ;
    for (int k = 0 ; k < nvals ; k++)
    {
        I [k] = simple_rand_i ( ) % n ;
        J [k] = simple_rand_i ( ) % n ;
        X [k] = simple_rand_x ( ) ;
    }

    //--------------------------------------------------------------------------
    // create random matrices parallel
    //--------------------------------------------------------------------------

    int nmats = MIN (4*nthreads, 256) ;

    for (int nmat = 1 ; nmat < nmats ; nmat = 2*nmat)
    {
        // create nmat matrices, each in parallel with varying # of threads
        for (int nthreads2 = 1 ; nthreads2 <= nthreads ; nthreads2 *= 2)
        {
            printf ("\n") ;
            int nouter = nthreads2 ; // # of user threads in outer loop
            int ninner = 1 ;        // # of threads each user thread can use

            while (nouter >= 1)
            {

                double t = TIMER ;

//              #pragma omp parallel for num_threads (nouter) \
//                  schedule (dynamic, 1)
                for (int k = 0 ; k < nmat ; k++)
                {
                    // each user thread constructs its own context
                    GxB_Context Context = NULL ;
                    OK (GxB_Context_new (&Context)) ;
                    printf ("Context %p\n", Context) ;
                    OK (GxB_Context_set (Context, GxB_NTHREADS, ninner)) ;
                    OK (GxB_Context_engage (Context)) ;
                    if (k == 0) OK (GxB_print (Context, 3)) ;

                    // kth user thread builds the kth matrix with ninner threads
                    GrB_Matrix A = NULL ;
                    OK (GrB_Matrix_new (&A, GrB_FP64, n, n)) ;
                    OK (GrB_Matrix_build (A, I, J, X, nvals, GrB_PLUS_FP64)) ;
                    OK (GxB_print (A, 2)) ;

                    // free the matrix just built
                    OK (GrB_Matrix_free (&A)) ;

                    // each user thread frees its own context
                    OK (GxB_Context_disengage (Context)) ;
                    OK (GxB_Context_free (&Context)) ;
                    printf ("here\n") ;
                }

                t = TIMER - t ;

                printf ("nmat: %4d threads (%4d,%4d): %4d time: %8.4f sec\n",
                    nmat, nouter, ninner, nouter * ninner, t) ;
                nouter = nouter / 2 ;
                ninner = ninner * 2 ;
            }
        }
    }

    printf ("bye\n") ;
    free (I) ;
    free (J) ;
    free (X) ;
    OK (GrB_finalize ( )) ;
}

