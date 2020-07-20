//------------------------------------------------------------------------------
// GB_matvec_check: print a GraphBLAS matrix and check if it is valid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// for additional diagnostics, use:
// #define GB_DEVELOPER 1

#include "GB_Pending.h"
#include "GB.h"

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_matvec_check    // check a GraphBLAS matrix or vector
(
    const GrB_Matrix A,     // GraphBLAS matrix to print and check
    const char *name,       // name of the matrix, optional
    int pr,                 // print level; if negative, ignore nzombie
                            // conditions and use GB_FLIP(pr) for diagnostics
    FILE *f,                // file for output
    const char *kind        // "matrix" or "vector"
)
{

    //--------------------------------------------------------------------------
    // decide what to print
    //--------------------------------------------------------------------------

    bool ignore_zombies = false ;
    if (pr < 0)
    {   GB_cov[2740]++ ;
// covered (2740): 24
        pr = GB_FLIP (pr) ;
        ignore_zombies = true ;
    }
    pr = GB_IMIN (pr, GxB_COMPLETE_VERBOSE) ;
    bool pr_silent   = (pr == GxB_SILENT) ;
    bool pr_complete = (pr == GxB_COMPLETE || pr == GxB_COMPLETE_VERBOSE) ;
    bool pr_short    = (pr == GxB_SHORT    || pr == GxB_SHORT_VERBOSE   ) ;
    bool one_based = GB_Global_print_one_based_get ( ) ;
    int64_t offset = (one_based) ? 1 : 0 ;
    #if GB_DEVELOPER
    int pr_type = pr ;
    #else
    int pr_type = 0 ;
    #endif

    GBPR0 ("\n  " GBd "x" GBd " GraphBLAS %s %s",
        (A != NULL) ? GB_NROWS (A) : 0,
        (A != NULL) ? GB_NCOLS (A) : 0,
        (A != NULL && A->type != NULL && A->type->name != NULL) ?
         A->type->name : "", kind) ;

    //--------------------------------------------------------------------------
    // check if null, freed, or uninitialized
    //--------------------------------------------------------------------------

    if (A == NULL)
    {   GB_cov[2741]++ ;
// covered (2741): 4
        GBPR0 (" NULL\n") ;
        return (GrB_NULL_POINTER) ;
    }

    GB_CHECK_MAGIC (A, kind) ;

    //--------------------------------------------------------------------------
    // print the header
    //--------------------------------------------------------------------------

    bool is_hyper = (A->h != NULL) ;
    bool is_full = GB_IS_FULL (A) ;
    bool is_sparse = !is_full ;

    GBPR0 (", %s", is_hyper ? "hypersparse" : (is_sparse ? "sparse" : "full")) ;
    GBPR0 (" %s:\n", A->is_csc ? "by col" : "by row") ;

    #if GB_DEVELOPER
    GBPR0 ("  max # entries: " GBd "\n", A->nzmax) ;
    GBPR0 ("  vlen: " GBd , A->vlen) ;
    if (A->nvec_nonempty != -1)
    {
        GBPR0 (" nvec_nonempty: " GBd , A->nvec_nonempty) ;
    }
    GBPR0 (" nvec: " GBd " plen: " GBd  " vdim: " GBd " hyper_ratio %g\n",
        A->nvec, A->plen, A->vdim, A->hyper_ratio) ;
    #endif

    //--------------------------------------------------------------------------
    // check the dimensions
    //--------------------------------------------------------------------------

    if (A->vlen < 0 || A->vlen > GxB_INDEX_MAX ||
        A->vdim < 0 || A->vdim > GxB_INDEX_MAX ||
        A->nzmax < 0 || A->nzmax > GxB_INDEX_MAX)
    {   GB_cov[2742]++ ;
// covered (2742): 2
        GBPR0 ("  invalid %s dimensions\n", kind) ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // check vector structure
    //--------------------------------------------------------------------------

    if (is_hyper)
    {
        // A is hypersparse
        if (! (A->nvec >= 0 && A->nvec <= A->plen && A->plen <= A->vdim))
        {   GB_cov[2743]++ ;
// covered (2743): 2
            GBPR0 ("  invalid hypersparse %s structure\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }
    else if (is_sparse)
    {
        // A is sparse
        if (! (A->nvec == A->plen && A->plen == A->vdim))
        {   GB_cov[2744]++ ;
// covered (2744): 4
            GBPR0 ("  invalid sparse %s structure\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }
    else
    {
        // A is full
        if (! (A->nvec == A->vdim && A->plen == -1))
        {   GB_cov[2745]++ ;
// NOT COVERED (2745):
            GBPR0 ("  invalid full %s structure\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    //--------------------------------------------------------------------------
    // count the allocated blocks
    //--------------------------------------------------------------------------

    GB_Pending Pending = A->Pending ;

    #if GB_DEVELOPER
    // a matrix contains 1 to 9 different allocated blocks
    int64_t nallocs = 1 +                       // header
        (A->h != NULL && !A->h_shallow) +       // A->h, if not shallow
        (A->p != NULL && !A->p_shallow) +       // A->p, if not shallow
        (A->i != NULL && !A->i_shallow) +       // A->i, if not shallow
        (A->x != NULL && !A->x_shallow) +       // A->x, if not shallow
        (Pending != NULL) +
        (Pending != NULL && Pending->i != NULL) +
        (Pending != NULL && Pending->j != NULL) +
        (Pending != NULL && Pending->x != NULL) ;
    if (pr_short || pr_complete)
    {
        GBPR ("  A %p number of memory blocks: " GBd "\n", A, nallocs) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // check the type
    //--------------------------------------------------------------------------

    GrB_Info info = GB_Type_check (A->type, "", pr_type, f) ;
    if (info != GrB_SUCCESS)
    {   GB_cov[2746]++ ;
// covered (2746): 2
        GBPR0 ("  %s has an invalid type\n", kind) ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // report shallow structure
    //--------------------------------------------------------------------------

    #if GB_DEVELOPER
    if (pr_short || pr_complete)
    {
        GBPR ("  ->h: %p shallow: %d\n", A->h, A->h_shallow) ;
        GBPR ("  ->p: %p shallow: %d\n", A->p, A->p_shallow) ;
        GBPR ("  ->i: %p shallow: %d\n", A->i, A->i_shallow) ;
        GBPR ("  ->x: %p shallow: %d\n", A->x, A->x_shallow) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // check p
    //--------------------------------------------------------------------------

    if (is_hyper || is_sparse)
    {
        if (A->p == NULL)
        {   GB_cov[2747]++ ;
// covered (2747): 2
            GBPR0 ("  ->p is NULL, invalid %s\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    //--------------------------------------------------------------------------
    // check an empty matrix
    //--------------------------------------------------------------------------

    bool A_empty = (A->nzmax == 0) ;

    if (A_empty)
    {
        // A->x and A->i pointers must be NULL and shallow must be false
        if (A->i != NULL || A->i_shallow || A->x_shallow)
        {   GB_cov[2748]++ ;
// covered (2748): 2
            GBPR0 ("  invalid empty %s\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }

        // check the vector pointers
        if (is_hyper || is_sparse)
        {
            for (int64_t j = 0 ; j <= A->nvec ; j++)
            {
                if (A->p [j] != 0)
                {   GB_cov[2749]++ ;
// covered (2749): 2
                    GBPR0 ("  ->p [" GBd "] = " GBd " invalid\n", j, A->p [j]) ;
                    return (GrB_INVALID_OBJECT) ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // check a non-empty matrix
    //--------------------------------------------------------------------------

    if (is_hyper || is_sparse)
    {
        if (!A_empty && A->i == NULL)
        {   GB_cov[2750]++ ;
// covered (2750): 2
            GBPR0 ("  ->i is NULL, invalid %s\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    //--------------------------------------------------------------------------
    // check the content of p
    //--------------------------------------------------------------------------

    if (is_hyper || is_sparse)
    {
        if (A->p [0] != 0)
        {   GB_cov[2751]++ ;
// covered (2751): 4
            GBPR0 ("  ->p [0] = " GBd " invalid\n", A->p [0]) ;
            return (GrB_INVALID_OBJECT) ;
        }

        for (int64_t j = 0 ; j < A->nvec ; j++)
        {
            if (A->p [j+1] < A->p [j] || A->p [j+1] > A->nzmax)
            {   GB_cov[2752]++ ;
// covered (2752): 4
                GBPR0 ("  ->p [" GBd "] = " GBd " invalid\n", j+1, A->p [j+1]) ;
                return (GrB_INVALID_OBJECT) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // check the content of h
    //--------------------------------------------------------------------------

    if (is_hyper)
    {
        int64_t jlast = -1 ;
        for (int64_t k = 0 ; k < A->nvec ; k++)
        {
            int64_t j = A->h [k] ;
            if (jlast >= j || j < 0 || j >= A->vdim)
            {   GB_cov[2753]++ ;
// covered (2753): 8
                GBPR0 ("  ->h [" GBd "] = " GBd " invalid\n", k, j) ;
                return (GrB_INVALID_OBJECT) ;
            }
            jlast = j ;
        }
    }

    //--------------------------------------------------------------------------
    // report name and number of entries
    //--------------------------------------------------------------------------

    GBPR0 ("  ") ;
    if (name != NULL && strlen (name) > 0)
    {   GB_cov[2754]++ ;
// covered (2754): 92155
        GBPR0 ("%s, ", GB_NAME) ;
    }

    // # of entries cannot be computed until all the tests above are OK
    int64_t anz = GB_NNZ (A) ;
    if (anz == 0)
    {   GB_cov[2755]++ ;
// covered (2755): 3924
        GBPR0 ("no entries\n") ;
    }
    else if (anz == 1)
    {   GB_cov[2756]++ ;
// covered (2756): 84716
        GBPR0 ("1 entry\n") ;
    }
    else
    {   GB_cov[2757]++ ;
// covered (2757): 3519
        GBPR0 ( GBd " entries\n", anz) ;
    }

    //--------------------------------------------------------------------------
    // report the number of pending tuples and zombies
    //--------------------------------------------------------------------------

    if (Pending != NULL || A->nzombies != 0)
    {   GB_cov[2758]++ ;
// covered (2758): 60
        GBPR0 ("  pending tuples: " GBd " max pending: " GBd 
            " zombies: " GBd "\n", GB_Pending_n (A),
            (Pending == NULL) ? 0 : (Pending->nmax),
            A->nzombies) ;
    }

    if (!ignore_zombies && (A->nzombies < 0 || A->nzombies > anz))
    {   GB_cov[2759]++ ;
// covered (2759): 4
        GBPR0 ("  invalid number of zombies: " GBd " "
            "must be >= 0 and <= # entries (" GBd ")\n", A->nzombies, anz) ;
        return (GrB_INVALID_OBJECT) ;
    }

    if (is_full)
    {
        if (A->nzombies != 0 || Pending != NULL)
        {   GB_cov[2760]++ ;
// NOT COVERED (2760):
            GBPR0 ("  full matrix cannot have zombies or pending tuples\n") ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    //--------------------------------------------------------------------------
    // check and print the row indices and numerical values
    //--------------------------------------------------------------------------

    if (anz > 0) GBPR0 ("\n") ;

    #define GB_NBRIEF 10
    #define GB_NZBRIEF 30

    int64_t nzombies = 0 ;
    int64_t jcount = 0 ;
    bool truncated = false ;

    // for each vector of A
    for (int64_t k = 0 ; k < A->nvec ; k++)
    {
        int64_t ilast = -1 ;
        int64_t j = GBH (A->h, k) ;
        int64_t p = GBP (A->p, k, A->vlen) ;
        int64_t pend = GBP (A->p, k+1, A->vlen) ;

        // for each entry in A(:,j), the kth vector of A
        for ( ; p < pend ; p++)
        {
            bool prcol = ((pr_short && jcount < GB_NBRIEF) || pr_complete) ;
            if (ilast == -1)
            {
                // print the header for vector j
                if (prcol)
                {   GB_cov[2761]++ ;
// covered (2761): 23105
                    #if GB_DEVELOPER
                    GBPR ("  %s: " GBd " : " GBd " entries [" GBd ":" GBd "]\n",
                        A->is_csc ? "column" : "row", j, pend - p, p, pend-1) ;
                    #endif
                }
                else if (pr_short && jcount == GB_NBRIEF)
                {   GB_cov[2762]++ ;
// covered (2762): 2
                    truncated = true ;
                    #if GB_DEVELOPER
                    GBPR ("    ...\n") ;
                    #endif
                }
                jcount++ ;      // count # of vectors printed so far
            }
            int64_t i = GBI (A->i, p, A->vlen) ;
            bool is_zombie = GB_IS_ZOMBIE (i) ;
            i = GB_UNFLIP (i) ;
            if (is_zombie) nzombies++ ;
            if (prcol)
            {   GB_cov[2763]++ ;
// covered (2763): 460209
                if ((pr_short && p < GB_NZBRIEF) || pr_complete)
                {   GB_cov[2764]++ ;
// covered (2764): 460069
                    #if GB_DEVELOPER
                    GBPR ("    %s " GBd ": ", A->is_csc ? "row":"column", i) ;
                    #else
                    if (A->is_csc)
                    {
                        GBPR ("    (" GBd "," GBd ") ", i + offset, j + offset);
                    }
                    else
                    {
                        GBPR ("    (" GBd "," GBd ") ", j + offset, i + offset);
                    }
                    #endif
                }
                else if (pr_short && (ilast == -1 || p == GB_NZBRIEF))
                {   GB_cov[2765]++ ;
// covered (2765): 16
                    truncated = true ;
                    #if GB_DEVELOPER
                    GBPR ("        ...\n") ;
                    #endif
                }
            }
            int64_t row = A->is_csc ? i : j ;
            int64_t col = A->is_csc ? j : i ;
            if (i < 0 || i >= A->vlen)
            {   GB_cov[2766]++ ;
// covered (2766): 2
                GBPR0 ("  index (" GBd "," GBd ") out of range\n", row, col) ;
                return (GrB_INVALID_OBJECT) ;
            }

            // print the value
            bool print_value = prcol &&
                ((pr_short && p < GB_NZBRIEF) || pr_complete) ;
            if (print_value)
            {   GB_cov[2767]++ ;
// covered (2767): 460067
                if (is_zombie)
                {   GB_cov[2768]++ ;
// covered (2768): 4
                    GBPR ("zombie") ;
                }
                else if (A->x != NULL)
                {   GB_cov[2769]++ ;
// covered (2769): 460043
                    GB_void *Ax = (GB_void *) A->x ;
                    info = GB_entry_check (A->type, Ax +(p * (A->type->size)),
                        pr, f) ;
                    if (info != GrB_SUCCESS) return (info) ;
                }
            }

            if (i <= ilast)
            {   GB_cov[2770]++ ;
// covered (2770): 10
                // indices unsorted, or duplicates present
                GBPR0 (" index (" GBd "," GBd ") jumbled\n", row, col) ;
                return (GrB_INDEX_OUT_OF_BOUNDS) ;
                // print_value = (!pr_silent) ;
            }

            if (print_value)
            {   GB_cov[2771]++ ;
// covered (2771): 460061
                GBPR ("\n") ;
            }
            ilast = i ;
        }
    }

    #if GB_DEVELOPER
    // ... already printed
    #else
    if (pr_short && truncated) GBPR ("    ...\n") ;
    #endif

    //--------------------------------------------------------------------------
    // check the zombie count
    //--------------------------------------------------------------------------

    if (!ignore_zombies && nzombies != A->nzombies)
    {   GB_cov[2772]++ ;
// covered (2772): 2
        GBPR0 ("  invalid zombie count: " GBd " exist but"
            " A->nzombies = " GBd "\n", nzombies, A->nzombies) ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // check and print the pending tuples
    //--------------------------------------------------------------------------

    #if GB_DEVELOPER
    if (pr_short || pr_complete)
    {
        GBPR ("  Pending %p\n", Pending) ;
    }
    #endif

    if (Pending != NULL)
    {

        //---------------------------------------------------------------------
        // A has pending tuples
        //---------------------------------------------------------------------

        #if GB_DEVELOPER
        if (pr_short || pr_complete)
        {
            GBPR ("  Pending->i %p\n", Pending->i) ;
            GBPR ("  Pending->j %p\n", Pending->j) ;
            GBPR ("  Pending->x %p\n", Pending->x) ;
        }
        #endif

        if (Pending->n < 0 || Pending->n > Pending->nmax ||
            Pending->nmax < 0)
        {   GB_cov[2773]++ ;
// covered (2773): 2
            GBPR0 ("  invalid pending count\n") ;
            return (GrB_INVALID_OBJECT) ;
        }

        // matrix has tuples, arrays and type must not be NULL
        if (Pending->i == NULL || Pending->x == NULL ||
            (A->vdim > 1 && Pending->j == NULL))
        {   GB_cov[2774]++ ;
// covered (2774): 2
            GBPR0 ("  invalid pending tuples\n") ;
            return (GrB_INVALID_OBJECT) ;
        }

        GBPR0 ("  pending tuples:\n") ;

        info = GB_Type_check (Pending->type, "", pr, f) ;
        if (info != GrB_SUCCESS || (Pending->type->size != Pending->size))
        {   GB_cov[2775]++ ;
// covered (2775): 4
            GBPR0 ("  %s has an invalid Pending->type\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }

        int64_t ilast = -1 ;
        int64_t jlast = -1 ;
        bool sorted = true ;

        for (int64_t k = 0 ; k < Pending->n ; k++)
        {
            int64_t i = Pending->i [k] ;
            int64_t j = (A->vdim <= 1) ? 0 : (Pending->j [k]) ;
            int64_t row = A->is_csc ? i : j ;
            int64_t col = A->is_csc ? j : i ;

            // print the tuple
            if ((pr_short && k < GB_NZBRIEF) || pr_complete)
            {   GB_cov[2776]++ ;
// covered (2776): 102
                GBPR ("    row: " GBd " col: " GBd " ", row, col) ;
                info = GB_entry_check (Pending->type,
                    Pending->x +(k * Pending->type->size), pr, f) ;
                if (info != GrB_SUCCESS) return (info) ;
                GBPR ("\n") ;
            }

            if (i < 0 || i >= A->vlen || j < 0 || j >= A->vdim)
            {   GB_cov[2777]++ ;
// covered (2777): 2
                GBPR0 ("    tuple (" GBd "," GBd ") out of range\n", row, col) ;
                return (GrB_INVALID_OBJECT) ;
            }

            sorted = sorted && ((jlast < j) || (jlast == j && ilast <= i)) ;
            ilast = i ;
            jlast = j ;
        }

        if (sorted != Pending->sorted)
        {   GB_cov[2778]++ ;
// covered (2778): 4
            GBPR0 ("  invalid pending tuples: invalid sort\n") ;
            return (GrB_INVALID_OBJECT) ;
        }

        if (Pending->op == NULL)
        {   GB_cov[2779]++ ;
// covered (2779): 30
            GBPR0 ("  pending operator: implicit 2nd\n") ;
        }
        else
        {
            info = GB_BinaryOp_check (Pending->op, "pending operator:", pr, f) ;
            if (info != GrB_SUCCESS)
            {   GB_cov[2780]++ ;
// covered (2780): 2
                GBPR0 ("  invalid pending operator\n") ;
                return (GrB_INVALID_OBJECT) ;
            }
        }
    }

    if (pr_complete)
    {   GB_cov[2781]++ ;
// covered (2781): 1713
        GBPR ("\n") ;
    }

    //--------------------------------------------------------------------------
    // check nvec_nonempty
    //--------------------------------------------------------------------------

    // A->nvec_nonempty == -1 denotes that the value has not been computed.
    // This is valid, and can occur for matrices imported with
    // GxB_Matrix_import*, and in other cases when its computation is postponed
    // or not needed.  If not -1, however, the value must be correct.

    int64_t actual_nvec_nonempty = GB_nvec_nonempty (A, NULL) ;

    if (! ((A->nvec_nonempty == actual_nvec_nonempty) ||
           (A->nvec_nonempty == -1)))
    {   GB_cov[2782]++ ;
// NOT COVERED (2782):
        GBPR0 ("  invalid count of non-empty vectors\n"
            "A->nvec_nonempty = " GBd " actual " GBd "\n",
            A->nvec_nonempty, actual_nvec_nonempty) ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

