//------------------------------------------------------------------------------
// GB_matvec_check: print a GraphBLAS matrix and check if it is valid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// OK: BITMAP

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
    { 
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
    { 
        GBPR0 (" NULL\n") ;
        return (GrB_NULL_POINTER) ;
    }

    GB_CHECK_MAGIC (A, kind) ;

    //--------------------------------------------------------------------------
    // print the header
    //--------------------------------------------------------------------------

    bool is_hyper = (A->h != NULL) ;
    bool is_full = GB_IS_FULL (A) ;
    bool is_bitmap = GB_IS_BITMAP (A) ;
    bool is_sparse = !(is_full || is_bitmap || is_hyper) ;

    if (is_full)
    { 
        GBPR0 (", full") ;
    }
    else if (is_bitmap)
    { 
        GBPR0 (", bitmap") ;
    }
    else if (is_sparse)
    { 
        GBPR0 (", sparse") ;
    }
    else if (is_hyper)
    { 
        GBPR0 (", hypersparse") ;
    }
    else
    {
        GBPR0 (" invalid structure\n") ;
        return (GrB_INVALID_OBJECT) ;
    }
    if (A->jumbled)
    { 
        GBPR0 (" (jumbled)") ;
    }
    GBPR0 (" %s:\n", A->is_csc ? "by col" : "by row") ;

    #if GB_DEVELOPER
    GBPR0 ("  max # entries: " GBd "\n", A->nzmax) ;
    GBPR0 ("  vlen: " GBd , A->vlen) ;
    if (A->nvec_nonempty != -1)
    {
        GBPR0 (" nvec_nonempty: " GBd , A->nvec_nonempty) ;
    }
    GBPR0 (" nvec: " GBd " plen: " GBd  " vdim: " GBd " hyper_switch %g\n",
        A->nvec, A->plen, A->vdim, A->hyper_switch) ;
    GBPR0 ("  sparsity control: ") ;
    switch (A->sparsity)
    {

        // 1
        case GxB_HYPERSPARSE :
            GBPR0 (" hypersparse only\n") ;
            break ;

        // 2
        case GxB_SPARSE :
            GBPR0 (" sparse only\n") ;
            break ;

        // 3
        case GxB_HYPERSPARSE + GxB_SPARSE :
            GBPR0 (" sparse or hypersparse\n") ;
            break ;

        // 4
        case GxB_BITMAP :
            GBPR0 (" bitmap only\n") ;
            break ;

        // 5
        case GxB_HYPERSPARSE + GxB_BITMAP :
            GBPR0 (" hypersparse or bitmap\n") ;
            break ;

        // 6
        case GxB_SPARSE + GxB_BITMAP :
            GBPR0 (" sparse or bitmap\n") ;
            break ;

        // 7
        case GxB_HYPERSPARSE + GxB_SPARSE + GxB_BITMAP :
            GBPR0 (" hypersparse, sparse, or bitmap\n") ;
            break ;

        // 8 and 12: these options are treated the same
        case GxB_FULL :
        case GxB_FULL + GxB_BITMAP :
            GBPR0 (" bitmap or full\n") ;
            break ;

        // 9
        case GxB_HYPERSPARSE + GxB_FULL :
            GBPR0 (" hypersparse or full\n") ;
            break ;

        // 10
        case GxB_SPARSE + GxB_FULL :
            GBPR0 (" sparse or full\n") ;
            break ;

        // 11
        case GxB_SPARSE + GxB_FULL :
            GBPR0 (" sparse or full\n") ;
            break ;

        // 13
        case GxB_HYPERSPARSE + GxB_BITMAP + GxB_FULL ;
            GBPR0 (" hypersparse, bitmap, or full\n") ;
            break ;

        // 14
        case GxB_SPARSE + GxB_BITMAP + GxB_FULL ;
            GBPR0 (" sparse, bitmap, or full\n") ;
            break ;

        // 15
        default :
        case GxB_AUTO_SPARSITY :
            GBPR0 (" auto\n") ;
            break ;
    }
    #endif

    //--------------------------------------------------------------------------
    // check the dimensions
    //--------------------------------------------------------------------------

    if (A->vlen < 0 || A->vlen > GxB_INDEX_MAX ||
        A->vdim < 0 || A->vdim > GxB_INDEX_MAX ||
        A->nzmax < 0 || A->nzmax > GxB_INDEX_MAX)
    { 
        GBPR0 ("  invalid %s dimensions\n", kind) ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // check vector structure
    //--------------------------------------------------------------------------

    if (is_full) 
    {
        // A is full
        if (! (A->nvec == A->vdim && A->plen == -1))
        { 
            GBPR0 ("  invalid full %s structure\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }
    else if (is_bitmap) 
    {
        // A is bitmap
        if (! (A->nvec == A->vdim && A->plen == -1 &&
               A->h == NULL && A->p == NULL && A->i == NULL))
        { 
            GBPR0 ("  invalid bitmap %s structure\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }
    else if (is_sparse)
    {
        // A is sparse
        if (! (A->nvec == A->plen && A->plen == A->vdim))
        { 
            GBPR0 ("  invalid sparse %s structure\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }
    else // if (is_hyper)
    {
        // A is hypersparse
        if (! (A->nvec >= 0 && A->nvec <= A->plen && A->plen <= A->vdim))
        { 
            GBPR0 ("  invalid hypersparse %s structure\n", kind) ;
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
        (A->b != NULL && !A->b_shallow) +       // A->b, if not shallow
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
    { 
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
        GBPR ("  ->b: %p shallow: %d\n", A->b, A->b_shallow) ;
        GBPR ("  ->x: %p shallow: %d\n", A->x, A->x_shallow) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // check p
    //--------------------------------------------------------------------------

    if (is_hyper || is_sparse)
    {
        if (A->p == NULL)
        { 
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
        { 
            GBPR0 ("  invalid empty %s\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }

        // check the vector pointers
        if (is_hyper || is_sparse)
        {
            for (int64_t j = 0 ; j <= A->nvec ; j++)
            {
                if (A->p [j] != 0)
                { 
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
        { 
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
        { 
            GBPR0 ("  ->p [0] = " GBd " invalid\n", A->p [0]) ;
            return (GrB_INVALID_OBJECT) ;
        }

        for (int64_t j = 0 ; j < A->nvec ; j++)
        {
            if (A->p [j+1] < A->p [j] || A->p [j+1] > A->nzmax)
            { 
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
            { 
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
    { 
        GBPR0 ("%s, ", GB_NAME) ;
    }

    // # of entries cannot be computed until all the tests above are OK
    int64_t anz = GB_NNZ (A) ;
    if (anz == 0)
    { 
        GBPR0 ("no entries\n") ;
    }
    else if (anz == 1)
    { 
        GBPR0 ("1 entry\n") ;
    }
    else
    { 
        GBPR0 ( GBd " entries\n", anz) ;
    }

    //--------------------------------------------------------------------------
    // report the number of pending tuples and zombies
    //--------------------------------------------------------------------------

    if (Pending != NULL || A->nzombies != 0)
    { 
        GBPR0 ("  pending tuples: " GBd " max pending: " GBd 
            " zombies: " GBd "\n", GB_Pending_n (A),
            (Pending == NULL) ? 0 : (Pending->nmax),
            A->nzombies) ;
    }

    if (!ignore_zombies && (A->nzombies < 0 || A->nzombies > anz))
    { 
        GBPR0 ("  invalid number of zombies: " GBd " "
            "must be >= 0 and <= # entries (" GBd ")\n", A->nzombies, anz) ;
        return (GrB_INVALID_OBJECT) ;
    }

    if (is_full || is_bitmap)
    {
        if (A->nzombies != 0)
        { 
            GBPR0 ("  %s %s cannot have zombies\n",
                is_full ? "full" : "bitmap", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
        if (Pending != NULL)
        { 
            GBPR0 ("  %s %s cannot have pending tuples\n",
                is_full ? "full" : "bitmap", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
        if (A->jumbled)
        { 
            GBPR0 ("  %s %s cannot be jumbled\n",
                is_full ? "full" : "bitmap", kind) ;
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
    int64_t icount = 0 ;
    bool truncated = false ;
    int64_t anz_actual = 0 ;

    // for each vector of A
    for (int64_t k = 0 ; k < A->nvec ; k++)
    {
        int64_t ilast = -1 ;
        int64_t j = GBH (A->h, k) ;
        int64_t p = GBP (A->p, k, A->vlen) ;
        int64_t pend = GBP (A->p, k+1, A->vlen) ;

        // count the entries in A(:,j)
        int64_t ajnz = pend - p ;
        if (is_bitmap)
        {
            ajnz = 0 ;
            for (int64_t p2 = p ; p2 < pend ; p2++)
            {
                ajnz += (A->b [p2] != 0)  ;
            }
        }

        bool prcol = ((pr_short && jcount < GB_NBRIEF) || pr_complete) ;
        // print the header for vector j
        if (prcol)
        { 
            #if GB_DEVELOPER
            GBPR ("  %s: " GBd " : " GBd " entries [" GBd ":" GBd "]\n",
                A->is_csc ? "column" : "row", j, ajnz, p, pend-1) ;
            #endif
        }
        else if (pr_short && jcount == GB_NBRIEF)
        { 
            truncated = true ;
            #if GB_DEVELOPER
            GBPR ("    ...\n") ;
            #endif
        }
        jcount++ ;      // count # of vectors printed so far

        // for each entry in A(:,j), the kth vector of A
        for ( ; p < pend ; p++)
        {
            if (!GBB (A->b, p)) continue ;
            anz_actual++ ;
            icount++ ;

            int64_t i = GBI (A->i, p, A->vlen) ;
            bool is_zombie = GB_IS_ZOMBIE (i) ;
            i = GB_UNFLIP (i) ;
            if (is_zombie) nzombies++ ;
            if (prcol)
            { 
                if ((pr_short && p < GB_NZBRIEF) || pr_complete)
                { 
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
                else if (pr_short && (ilast == -1 || icount == GB_NZBRIEF))
                { 
                    truncated = true ;
                    #if GB_DEVELOPER
                    GBPR ("        ...\n") ;
                    #endif
                }
            }
            int64_t row = A->is_csc ? i : j ;
            int64_t col = A->is_csc ? j : i ;
            if (i < 0 || i >= A->vlen)
            { 
                GBPR0 ("  index (" GBd "," GBd ") out of range\n", row, col) ;
                return (GrB_INVALID_OBJECT) ;
            }

            // print the value
            bool print_value = prcol &&
                ((pr_short && icount < GB_NZBRIEF) || pr_complete) ;
            if (print_value)
            { 
                if (is_zombie)
                { 
                    GBPR ("zombie") ;
                }
                else if (A->x != NULL)
                { 
                    GB_void *Ax = (GB_void *) A->x ;
                    info = GB_entry_check (A->type, Ax +(p * (A->type->size)),
                        pr, f) ;
                    if (info != GrB_SUCCESS) return (info) ;
                }
            }

            // If the matrix is known to be jumbled, then out-of-order
            // indices are OK (but duplicates are not OK).  If the matrix is
            // unjumbled, then all indices must appear in ascending order.
            if (A->jumbled ? (i == ilast) : (i <= ilast))
            { 
                // indices unsorted, or duplicates present
                GBPR0 (" index (" GBd "," GBd ") invalid\n", row, col) ;
                return (GrB_INDEX_OUT_OF_BOUNDS) ;
            }

            if (print_value)
            { 
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
    // check the entry count in the bitmap
    //--------------------------------------------------------------------------

    if (is_bitmap && anz != anz_actual)
    { 
        // this case can only occur for bitmapped matrices
        GBPR0 ("  invalid bitmap count: " GBd " exist but"
            " A->nvals = " GBd "\n", anz_actual, anz) ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // check the zombie count
    //--------------------------------------------------------------------------

    if (!ignore_zombies && nzombies != A->nzombies)
    { 
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
        { 
            GBPR0 ("  invalid pending count\n") ;
            return (GrB_INVALID_OBJECT) ;
        }

        // matrix has tuples, arrays and type must not be NULL
        if (Pending->i == NULL || Pending->x == NULL ||
            (A->vdim > 1 && Pending->j == NULL))
        { 
            GBPR0 ("  invalid pending tuples\n") ;
            return (GrB_INVALID_OBJECT) ;
        }

        GBPR0 ("  pending tuples:\n") ;

        info = GB_Type_check (Pending->type, "", pr, f) ;
        if (info != GrB_SUCCESS || (Pending->type->size != Pending->size))
        { 
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
            { 
                GBPR ("    row: " GBd " col: " GBd " ", row, col) ;
                info = GB_entry_check (Pending->type,
                    Pending->x +(k * Pending->type->size), pr, f) ;
                if (info != GrB_SUCCESS) return (info) ;
                GBPR ("\n") ;
            }

            if (i < 0 || i >= A->vlen || j < 0 || j >= A->vdim)
            { 
                GBPR0 ("    tuple (" GBd "," GBd ") out of range\n", row, col) ;
                return (GrB_INVALID_OBJECT) ;
            }

            sorted = sorted && ((jlast < j) || (jlast == j && ilast <= i)) ;
            ilast = i ;
            jlast = j ;
        }

        if (sorted != Pending->sorted)
        { 
            GBPR0 ("  invalid pending tuples: invalid sort\n") ;
            return (GrB_INVALID_OBJECT) ;
        }

        if (Pending->op == NULL)
        { 
            GBPR0 ("  pending operator: implicit 2nd\n") ;
        }
        else
        {
            info = GB_BinaryOp_check (Pending->op, "pending operator:", pr, f) ;
            if (info != GrB_SUCCESS)
            { 
                GBPR0 ("  invalid pending operator\n") ;
                return (GrB_INVALID_OBJECT) ;
            }
        }
    }

    if (pr_complete)
    { 
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
    { 
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

