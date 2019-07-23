/* ========================================================================== */
/* === ssmult_saxpy ========================================================= */
/* ========================================================================== */

/* C = ssmult_saxpy (A,B) multiplies two sparse matrices A and B, in MATLAB.
 * Either A or B, or both, can be complex.  C is returned as a proper MATLAB
 * sparse matrix, with sorted row indices, or with unsorted row indices.  No
 * explicit zero entries appear in C.  If A or B are complex, but the imaginary
 * part of C is computed to be zero, then C is returned as a real sparse matrix
 * (as in MATLAB).
 *
 * Copyright 2007-2009, Timothy A. Davis, http://www.suitesparse.com
 */

// #include "ssmult.h"
#include "sort.h"

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
/* SORT(a,b) sorts [a b] in ascending order, so that a < b holds on output */
/* -------------------------------------------------------------------------- */

#define SORT(a,b) { if (a > b) { t = a ; a = b ; b = t ; } }


/* -------------------------------------------------------------------------- */
/* BUBBLE(a,b) sorts [a b] in ascending order, and sets done to 0 if it swaps */
/* -------------------------------------------------------------------------- */

#define BUBBLE(a,b) { if (a > b) { t = a ; a = b ; b = t ; done = 0 ; } }


/* -------------------------------------------------------------------------- */
/* ssmergesort */
/* -------------------------------------------------------------------------- */

/* ssmergesort (A,W,n) sorts an Int array A of length n in ascending order. W is
 * a workspace array of size n.  function is used for sorting the row indices
 * in each column of C.  Small lists (of length SMALL or less) are sorted with
 * a bubble sort.  A value of 10 for SMALL works well on an Intel Core Duo, an
 * Intel Pentium 4M, and a 64-bit AMD Opteron.  SMALL must be in the range 4 to
 * 10. */

#ifndef SMALL
#define SMALL 10
#endif

void ssmergesort
(
    Int A [ ],      /* array to sort, of size n */
    Int W [ ],      /* workspace of size n */
    long n
)
{
    if (n <= SMALL)
    {

        /* ------------------------------------------------------------------ */
        /* bubble sort for small lists of length SMALL or less */
        /* ------------------------------------------------------------------ */

        Int t, done ;
        switch (n)
        {

#if SMALL >= 10
            case 10:
                /* 10-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                BUBBLE (A [2], A [3]) ;
                BUBBLE (A [3], A [4]) ;
                BUBBLE (A [4], A [5]) ;
                BUBBLE (A [5], A [6]) ;
                BUBBLE (A [6], A [7]) ;
                BUBBLE (A [7], A [8]) ;
                BUBBLE (A [8], A [9]) ;
                if (done) return ;
#endif

#if SMALL >= 9
            case 9:
                /* 9-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                BUBBLE (A [2], A [3]) ;
                BUBBLE (A [3], A [4]) ;
                BUBBLE (A [4], A [5]) ;
                BUBBLE (A [5], A [6]) ;
                BUBBLE (A [6], A [7]) ;
                BUBBLE (A [7], A [8]) ;
                if (done) return ;
#endif

#if SMALL >= 8
            case 8:
                /* 7-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                BUBBLE (A [2], A [3]) ;
                BUBBLE (A [3], A [4]) ;
                BUBBLE (A [4], A [5]) ;
                BUBBLE (A [5], A [6]) ;
                BUBBLE (A [6], A [7]) ;
                if (done) return ;
#endif

#if SMALL >= 7
            case 7:
                /* 7-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                BUBBLE (A [2], A [3]) ;
                BUBBLE (A [3], A [4]) ;
                BUBBLE (A [4], A [5]) ;
                BUBBLE (A [5], A [6]) ;
                if (done) return ;
#endif

#if SMALL >= 6
            case 6:
                /* 6-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                BUBBLE (A [2], A [3]) ;
                BUBBLE (A [3], A [4]) ;
                BUBBLE (A [4], A [5]) ;
                if (done) return ;
#endif

#if SMALL >= 5
            case 5:
                /* 5-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                BUBBLE (A [2], A [3]) ;
                BUBBLE (A [3], A [4]) ;
                if (done) return ;
#endif

            case 4:
                /* 4-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                BUBBLE (A [2], A [3]) ;
                if (done) return ;

            case 3:
                /* 3-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                if (done) return ;

            case 2:
                /* 2-element bubble sort */
                SORT (A [0], A [1]) ; 

            case 1:
            case 0:
                /* nothing to do */
                ;
        }

    }
    else
    {

        /* ------------------------------------------------------------------ */
        /* recursive ssmergesort if A has length 5 or more */
        /* ------------------------------------------------------------------ */

        long n1, n2, n3, n4, n12, n34, n123 ;

        n12 = n / 2 ;           /* split n into n12 and n34 */
        n34 = n - n12 ;

        n1 = n12 / 2 ;          /* split n12 into n1 and n2 */
        n2 = n12 - n1 ;

        n3 = n34 / 2 ;          /* split n34 into n3 and n4 */
        n4 = n34 - n3 ;

        n123 = n12 + n3 ;       /* start of 4th subset = n1 + n2 + n3 */

        ssmergesort (A,        W, n1) ;       /* sort A [0  ... n1-1] */
        ssmergesort (A + n1,   W, n2) ;       /* sort A [n1 ... n12-1] */
        ssmergesort (A + n12,  W, n3) ;       /* sort A [n12 ... n123-1] */
        ssmergesort (A + n123, W, n4) ;       /* sort A [n123 ... n-1]  */

        /* merge A [0 ... n1-1] and A [n1 ... n12-1] into W [0 ... n12-1] */
        merge (W, A, n1, A + n1, n2) ;

        /* merge A [n12 ... n123-1] and A [n123 ... n-1] into W [n12 ... n-1] */
        merge (W + n12, A + n12, n3, A + n123, n4) ;

        /* merge W [0 ... n12-1] and W [n12 ... n-1] into A [0 ... n-1] */
        merge (A, W, n12, W + n12, n34) ;
    }
}

