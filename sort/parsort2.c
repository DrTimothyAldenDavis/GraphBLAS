//
// basic parallel merge sort.  No attempts have been made to optmize this code.
//
// this version is simpler ... we split by two at each level.  
//
#include "sort.h"
#include "omp.h"

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
/* pmergesortbin */
/* -------------------------------------------------------------------------- */

/* pmergesortbin (A,W,n) sorts an Int array A of length n in ascending order. W is
 * a workspace array of size n.  function is used for sorting the row indices
 * in each column of C.  Small lists (of length SMALL or less) are sorted with
 * a bubble sort.  A value of 10 for SMALL works well on an Intel Core Duo, an
 * Intel Pentium 4M, and a 64-bit AMD Opteron.  SMALL must be in the range 4 to
 * 10. */

#ifndef BASECASE
#define BASECASE 64
#endif

void pmergesortbin
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
        /* recursive ssmergesort if A has length SMALL or more */
        /* ------------------------------------------------------------------ */

          long n1, n2; 
   
          n1 = n/2;
          n2 = n-n1;

//      #pragma omp task firstprivate(n1)
        pmergesortbin (A,        W, n1) ;       /* sort A [0  ... n1-1] */

//      #pragma omp task firstprivate(n1, n)
        pmergesortbin (A + n1,   W, n2) ;       /* sort A [n1 ... n-1] */

//      #pragma omp taskwait
        
        /* merge A [0 ... n1-1] and A [n1 ... n-1] into W [0 ... n-1] */
        merge (W, A, n1, A + n1, n2 ) ;

        /* copy W into A ... a hack to match expected output of sorted array */
        for(int i=0; i< n; i++) A[i] = W[i];

    }
}

void parsort2(Int *A, Int *W, long n)
{
  if(omp_get_num_threads() >1)
    pmergesortbin(A, W, n);
  else
  {
//  #pragma omp parallel
//  #pragma omp master
      pmergesortbin(A, W, n);
  }
}
