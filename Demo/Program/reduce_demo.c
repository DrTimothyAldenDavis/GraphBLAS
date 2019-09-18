//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/reduce_demo: reduce a matrix to a scalar
//------------------------------------------------------------------------------

#include "GraphBLAS.h"

#define N 65536

int main (void)
{
    // start GraphBLAS
    GrB_init (GrB_NONBLOCKING) ;
    printf ("demo: reduce a matrix to a scalar\n") ;

    GrB_Index nrows = N ;
    GrB_Index ncols = N ;
    GrB_Matrix A ;
    GrB_Matrix_new (&A, GrB_INT32, nrows, ncols) ;

    int64_t k = 0 ;
    for (int i = 0 ; i < nrows ; i++)
    {
        for (int j = 0 ; j < ncols ; j++)
        {
            // int x = (int) (rand ( ) & 0xFF) ;
            int x = (int) (k & 0xFF) ;
            k++ ;
            GrB_Matrix_setElement (A, x, i, j) ;
        }
    }

    GrB_Index anz ;
    GrB_Matrix_nvals (&anz, A) ;
    GxB_print (A, 2) ;

    int nthreads_max ;
    GxB_get (GxB_NTHREADS, &nthreads_max) ;
    fprintf (stderr, "# of threads: %d\n", nthreads_max) ;

    GrB_Index result ;

    #if defined ( _OPENMP )
    double t = omp_get_wtime ( ) ;
    #endif
    GrB_reduce (&result, NULL, GxB_PLUS_INT32_MONOID, A, NULL) ;
    #if defined ( _OPENMP )
    t = omp_get_wtime ( ) - t ;
    printf ("time: %g (default # of threads)\n", t) ;
    #endif

    double t1 ;

    for (int nthreads = 1 ; nthreads <= nthreads_max ; nthreads++)
    {
        GxB_set (GxB_NTHREADS, nthreads) ;
        #if defined ( _OPENMP )
        double t = omp_get_wtime ( ) ;
        #endif
        GrB_reduce (&result, NULL, GxB_PLUS_INT32_MONOID, A, NULL) ;
        #if defined ( _OPENMP )
        t = omp_get_wtime ( ) - t ;
        if (nthreads == 1) t1 = t ;
        printf ("nthreads %3d time: %12.6fg speedup %8.2f\n", 
            nthreads, t, t1/t) ;
        #endif
    }

    printf ("result %"PRId64"\n", result) ;

    // free everyting
    GrB_free (&A) ;
    GrB_finalize ( ) ;
}

