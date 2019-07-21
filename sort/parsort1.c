//
// basic parallel merge sort.  No attempts have been made to optmize this code.
//
#include "sort.h"
#include "omp.h"
#include <stdbool.h>
#include <stdio.h>
#include "GB_qsort.h"

// to dump output:
// #define DUMP 1
#define DUMP 0

// to debug
// #define DEBUG 1
#define DEBUG 0

#define ASSERT(x)                                                   \
{                                                                   \
    if (!(x))                                                       \
    {                                                               \
        printf ("\nhey! %s %d\n", __FILE__, __LINE__) ;             \
        fflush (stdout) ;                                           \
        abort ( ) ;                                                 \
    }                                                               \
}

#ifndef BASECASE
// #define BASECASE (32)
#define BASECASE (1024 * 1024)
#endif

// prototypes:
void pmerge                 // sequential or parallel merge
(
    Int S [ ],              // S [0..nleft+nright-1]: output
    const Int Left [ ],     // Left [0..nleft-1]: input on the left
    const long nleft,
    const Int Right [ ],    // Right [0..nright-1]: input on the righ
    const long nright
) ;

void pmerge2                // parallel merge, where nbigger >= nsmaller
(
    Int S [ ],              // S [0..nbigger+nsmaller-1]: output
    const Int Bigger [ ],   // Bigger [0..nbigger-1]: larger input
    const long nbigger,
    const Int Smaller [ ],  // Smaller [0..nsmaller-1]: smaller input
    const long nsmaller
) ;

void check_sort
(
    char *name,
    const Int S [ ],
    const long n
) ;

void dump_list
(
    char *name,
    const Int S [ ],
    const long n
)
{
    if (DUMP) printf ("dump list %s [%g]:", name, (double) n) ;
    if (DUMP) if (n == 0) { printf ("\n") ; return ; }
    if (DUMP) printf (" %g ", (double) S [0]) ;
    for (int64_t k = 1 ; k < n ; k++)
    {
        if (DUMP) printf ("%g ", (double) S [k]) ;
    }
    if (DUMP) printf ("\n") ;
}

//------------------------------------------------------------------------------
// check_sorted
//------------------------------------------------------------------------------

void check_sort
(
    char *name,
    const Int S [ ],
    const long n
)
{
    if (DUMP) printf ("check sort %s [%g]:", name, (double) n) ;
    if (DUMP) if (n == 0) { printf ("\n") ; return ; }
    if (DUMP) printf (" %g ", (double) S [0]) ;
    for (int64_t k = 1 ; k < n ; k++)
    {
        if (DUMP) printf ("%g ", (double) S [k]) ;
        ASSERT (S [k-1] <= S [k]) ;
    }
    if (DUMP) printf ("\n") ;
}

/* -------------------------------------------------------------------------- */
/* merge Left [0..nleft-1] and Right [0..nright-1] into S [0..nleft+nright-1] */
/* -------------------------------------------------------------------------- */

void check_merge
(
    const Int S [ ],        /* output of length nleft + nright */
    const Int Left [ ],     /* left input of length nleft */
    const long nleft,
    const Int Right [ ],    /* right input of length nright */
    const long nright
)
{
    long p, pleft, pright ;

    if (DUMP) printf ("check merge %g %g\n", (double) nleft, (double) nright) ;
    dump_list ("Left ", Left, nleft) ;
    dump_list ("Right", Right, nright) ;
    dump_list ("S    ", S, nleft+nright) ;

    /* merge the two inputs, Left and Right, while both inputs exist */
    for (p = 0, pleft = 0, pright = 0 ; pleft < nleft && pright < nright ; p++)
    {
        if (Left [pleft] < Right [pright])
        {
            ASSERT (S [p] == Left [pleft]) ;
            pleft++ ;
        }
        else
        {
            ASSERT (S [p] == Right [pright]) ;
            pright++ ;
        }
    }

    /* either input is exhausted; copy the remaining list into S */
    for ( ; pleft < nleft ; p++)
    {
        ASSERT (S [p] == Left [pleft]) ;
        pleft++ ;
    }
    for ( ; pright < nright ; p++)
    {
        ASSERT (S [p] == Right [pright]) ;
        pright++ ;
    }

    check_sort ("S", S, nleft+nright) ;
}

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

    #if DEBUG
    if (DUMP) printf ("\nmerge %g %g\n", (double) nleft, (double) nright) ;
    check_sort ("Left ", Left, nleft) ;
    check_sort ("Right", Right, nright) ;
    #endif

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

    #if DEBUG
    check_merge (S, Left, nleft, Right, nright) ;
    #endif
}

//------------------------------------------------------------------------------
// pmerge2: parallel merge
//------------------------------------------------------------------------------

// The two input arrays, Bigger [0..nbigger-1] and Smaller [0..nsmaller-1], are
// sorted.  They are merged into the output array S [0..nleft+nright-1], using
// a parallel merge.  nbigger >= nsmaller always holds.

void pmerge2
(
    Int S [ ],              // S [0..nbigger+nsmaller-1]: output
    const Int Bigger [ ],   // Bigger [0..nbigger-1]: larger input
    const long nbigger,
    const Int Smaller [ ],  // Smaller [0..nsmaller-1]: smaller input
    const long nsmaller
)
{

    #if DEBUG
    if (DUMP) printf ("\npmerge2 %g %g\n", (double) nbigger, (double) nsmaller);
    ASSERT (nbigger >= nsmaller) ;
    check_sort ("Bigger ", Bigger, nbigger) ;
    check_sort ("Smaller", Smaller, nsmaller) ;
    #endif

    //--------------------------------------------------------------------------
    // split the bigger input in half
    //--------------------------------------------------------------------------

    // The first task will handle Bigger [0..nhalf-1], and the second task
    // will handle Bigger [nhalf..n-1].

    // long n = nbigger + nsmaller ;
    long nhalf = nbigger/2 ;
    Int i = Bigger [nhalf] ;

    #if DEBUG
    for (int64_t k = 0     ; k < nhalf   ; k++) ASSERT (Bigger [k] <= i) ;
    for (int64_t k = nhalf ; k < nbigger ; k++) ASSERT (Bigger [k] >= i) ;
    #endif

    //--------------------------------------------------------------------------
    // find where entry i appears in the smaller list
    //--------------------------------------------------------------------------

    // This is done in the GB_BINARY_TRIM_SEARCH macro in GraphBLAS:

    // binary search of Smaller [0..nsmaller-1] for the integer i

    long pleft = 0, pright = nsmaller-1 ;
    while (pleft < pright)
    {
        long pmiddle = (pleft + pright) / 2 ;
        if (Smaller [pmiddle] < i)
        {
            // if in the list, i appears in [pmiddle+1..pright]
            pleft = pmiddle + 1 ;
        }
        else
        {
            // if in the list, i appears in [pleft..pmiddle]
            pright = pmiddle ;
        }
    }

    // binary search is narrowed down to a single item
    // or it has found the list is empty:
    ASSERT (pleft == pright || pleft == pright + 1) ;

    // If found is true then Smaller [pleft == pright] == i.  If duplicates
    // appear then Smaller [pleft] is any one of the entries with value i in
    // the list.  If found is false then
    //    Smaller [original_pleft ... pleft-1] < i and
    //    Smaller [pleft+1 ... original_pright] > i holds.
    //    The value Smaller [pleft] may be either < or > i.
    bool found = (pleft == pright && Smaller [pleft] == i) ;

    // Modify pleft and pright:
    if (!found && (pleft == pright))
    {
        if (i > Smaller [pleft])
        {
            pleft++ ;
        }
        else
        {
            pright++ ;
        }
    }

    // Now the following conditions hold:

    // If found is false then
    //    Smaller [original_pleft ... pleft-1] < i and
    //    Smaller [pleft ... original_pright] > i holds, and pleft-1 == pright

    // If Smaller has no duplicates, then whether or not i is found,
    //    Smaller [original_pleft ... pleft-1] < i and
    //    Smaller [pleft ... original_pright] >= i holds.

    #if DEBUG
    for (int64_t k = 0     ; k < pleft    ; k++) ASSERT (Smaller [k] <= i) ;
    for (int64_t k = pleft ; k < nsmaller ; k++) ASSERT (Smaller [k] >= i) ;
    #endif

    //--------------------------------------------------------------------------
    // merge each part in parallel
    //--------------------------------------------------------------------------

    // The first task merges Bigger [0..nhalf-1] and Smaller [0..pleft-1] into
    // the output S [0..nhalf+pleft-1].  The entries in Bigger [0..nhalf-1] are
    // all < i (if no duplicates appear in Bigger) or <= i otherwise.

    Int *S0 = S ;
    const Int *Left0 = Bigger ;
    long nleft0 = nhalf ;
    const Int *Right0 = Smaller ;
    long nright0 = pleft ;

    // The second task merges Bigger [nhalf..nbigger-1] and
    // Smaller [pleft..nsmaller-1] into the output S [nhalf+pleft..n-1].
    // The entries in Bigger [nhalf..nbigger-1] and Smaller [pleft..nsmaller-1]
    // are all >= i.

    Int *S1 = S + nhalf + pleft ;
    const Int *Left1 = Bigger + nhalf ;
    long nleft1 = (nbigger - nhalf) ;
    const Int *Right1 = Smaller + pleft ;
    long nright1 = (nsmaller - pleft) ;

    #pragma omp task firstprivate(S0, Left0, nleft0, Right0, nright0)
    pmerge (S0, Left0, nleft0, Right0, nright0) ;

    #pragma omp task firstprivate(S1, Left1, nleft1, Right1, nright1)
    pmerge (S1, Left1, nleft1, Right1, nright1) ;

    #pragma omp taskwait

    #if DEBUG
    check_merge (S, Bigger, nbigger, Smaller, nsmaller) ;
    #endif
}

//------------------------------------------------------------------------------
// pmerge: parallel or sequential merge
//------------------------------------------------------------------------------

// The two input arrays, Left [0..nleft-1] and Right [0..nright-1], are sorted.
// They are merged into the output array S [0..nleft+nright-1], using either
// the sequential merge (for small lists) or the parallel merge (for big
// lists).

void pmerge
(
    Int S [ ],              // S [0..nleft+nright-1]: output
    const Int Left [ ],     // Left [0..nleft-1]: input on the left
    const long nleft,
    const Int Right [ ],    // Right [0..nright-1]: input on the righ
    const long nright
)
{
    if (nleft + nright < BASECASE || nleft == 0 || nright == 0)
    {
        // sequential merge
        merge (S, Left, nleft, Right, nright) ;
    }
    else if (nleft >= nright)
    {
        // parallel merge, where Left [0..nleft-1] is the bigger of the two.
        pmerge2 (S, Left, nleft, Right, nright) ;
    }
    else
    {
        // parallel merge, where Right [0..nright-1] is the bigger of the two.
        pmerge2 (S, Right, nright, Left, nleft) ;
    }

    #if DEBUG
    check_merge (S, Left, nleft, Right, nright) ;
    #endif
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
        #if DEBUG
        if (DUMP) printf ("sequential sort %g\n", (double) n) ;
        #endif
        // sequential quicksort; no workspace needed
        GB_qsort_1a ((int64_t *) A, n) ;
        // sequential mergesort: using workspace W
        // ssmergesort(A, W, n);
    }
    else
    {

        /* ------------------------------------------------------------------ */
        /* recursive ssmergesort if A has length greater than BASECASE        */
        /* ------------------------------------------------------------------ */

        #if DEBUG
        if (DUMP) printf ("\nparallel mergsort %g\n", (double) n) ;
        #endif

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
        pmerge (W0, A0, n1, A1, n2) ;

        /* merge A [n12 ... n123-1] and A [n123 ... n-1] into W [n12 ... n-1] */
        #pragma omp task firstprivate(W2,A2,A3,n3,n4)
        pmerge (W2, A2, n3, A3, n4) ;

        #pragma omp taskwait

        /* merge W [0 ... n12-1] and W [n12 ... n-1] into A [0 ... n-1] */
        pmerge (A, W0, n12, W2, n34) ;
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
