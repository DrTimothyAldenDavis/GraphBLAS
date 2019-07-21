//
// basic parallel merge sort.  No attempts have been made to optmize this code.
//
#include "sort.h"
#include "omp.h"

#ifndef BASECASE
#define BASECASE (32 * 1024)
#endif

/* -------------------------------------------------------------------------- */
/* merge Left [0..nleft-1] and Right [0..nright-1] into S [0..nleft+nright-1] */
/* -------------------------------------------------------------------------- */

static void merge
(
    Int S [ ],              /* output of length nleft + nright */
    const Int Left [ ],     /* left input of length nleft */
    const long nleft,
    const Int Right [ ],    /* right input of length nright */
    const long nright
)
{
    long p, pleft, pright ;

    /* merge the two inputs, Left and Right, while both inputs exist */
    for (p = 0, pleft = 0, pright = 0 ; pleft < nleft && pright < nright ; p++)
    {
        if (Left [pleft] < Right [pright])
        {
            S [p] = Left [pleft++] ;
        }
        else
        {
            S [p] = Right [pright++] ;
        }
    }

    /* either input is exhausted; copy the remaining list into S */
    for ( ; pleft < nleft ; p++)
    {
        S [p] = Left [pleft++] ;
    }
    for ( ; pright < nright ; p++)
    {
        S [p] = Right [pright++] ;
    }
}


/* -------------------------------------------------------------------------- */
/* pmergesort */
/* -------------------------------------------------------------------------- */

/* pmergesort (A,W,n) sorts an Int array A of length n in ascending order. W is
 * a workspace array of size n.  function is used for sorting the row indices
 * in each column of C.  Small lists (of length SMALL or less) are sorted with
 * a bubble sort.  A value of 10 for SMALL works well on an Intel Core Duo, an
 * Intel Pentium 4M, and a 64-bit AMD Opteron.  SMALL must be in the range 4 to
 * 10. */

void pmergesort
(
    Int A [ ],      /* array to sort, of size n */
    Int W [ ],      /* workspace of size n */
    long n
)
{
    if (n <= BASECASE)
    {
       ssmergesort(A, W, n);
    }
    else
    {

        /* ------------------------------------------------------------------ */
        /* recursive ssmergesort if A has length greater than BASECASE        */
        /* ------------------------------------------------------------------ */

        long n1,  n2,  n3,  n4; /* length of each of the four sections of A */
        long n12, n123, n34;    

        n12 = n / 2 ;           /* split n into n12 and n34 */
        n34 = n - n12 ;

        n1 = n12 / 2 ;          /* split n12 into n1 and n2 */
        n2 = n12 - n1 ;

        n3 = n34 / 2 ;          /* split n34 into n3 and n4 */
        n4 = n34 - n3 ;

        n123 = n12 + n3 ;       /* start of 4th subset = n1 + n2 + n3 */

        Int *A0 = A ;
        Int *A1 = A + n1 ;
        Int *A2 = A + n12 ;
        Int *A3 = A + n123 ;

        Int *W0 = W ;
        Int *W1 = W + n1 ;
        Int *W2 = W + n12 ;
        Int *W3 = W + n123 ;

        #pragma omp task firstprivate(A0,W0,n1)
        pmergesort (A0, W0, n1) ;       /* sort A [0  ... n1-1] */

        #pragma omp task firstprivate(A1,W1,n2)
        pmergesort (A1, W1, n2) ;       /* sort A [n1 ... n12-1] */

        #pragma omp task firstprivate(A2,W2,n3)
        pmergesort (A2, W2, n3) ;       /* sort A [n12 ... n123-1] */

        #pragma omp task firstprivate(A3,W3,n4)
        pmergesort (A3, W3, n4) ;       /* sort A [n123 ... n-1]  */

        #pragma omp taskwait
        
        /* merge A [0 ... n1-1] and A [n1 ... n12-1] into W [0 ... n12-1] */
        #pragma omp task firstprivate(W0,A0,A1,n1,n2)
        merge (W0, A0, n1, A1, n2) ;

        /* merge A [n12 ... n123-1] and A [n123 ... n-1] into W [n12 ... n-1] */
        #pragma omp task firstprivate(W2,A2,A3,n3,n4)
        merge (W2, A2, n3, A3, n4) ;

        #pragma omp taskwait

        /* merge W [0 ... n12-1] and W [n12 ... n-1] into A [0 ... n-1] */
        merge (A, W0, n12, W2, n34) ;

    }
}

void parsort1(Int *A, Int *W, long n)
{
  if(omp_get_num_threads() >1)
    pmergesort(A, W, n);
  else
  {
    #pragma omp parallel
    #pragma omp master
      pmergesort(A, W, n);
  }
}
