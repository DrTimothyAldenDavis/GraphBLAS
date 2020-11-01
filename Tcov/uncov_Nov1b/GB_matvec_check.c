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

    bool is_hyper = GB_IS_HYPERSPARSE (A) ;
    bool is_full = GB_IS_FULL (A) ;
    bool is_bitmap = GB_IS_BITMAP (A) ;
    bool is_sparse = GB_IS_SPARSE (A) ;

    bool ignore_zombies = false ;
    if (pr < 0)
    {   GB_cov[3493]++ ;
// covered (3493): 24
        pr = GB_FLIP (pr) ;
        ignore_zombies = true ;
    }
    pr = GB_IMIN (pr, GxB_COMPLETE_VERBOSE) ;
    bool phantom = (is_full && A->x == NULL) ;
    if (phantom)
    {   GB_cov[3494]++ ;
// covered (3494): 4
        // convert GxB_COMPLETE* to GxB_SHORT*
        if (pr == GxB_COMPLETE_VERBOSE) pr = GxB_SHORT_VERBOSE ;
        if (pr == GxB_COMPLETE        ) pr = GxB_SHORT ;
    }
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
    {   GB_cov[3495]++ ;
// covered (3495): 4
        GBPR0 (" NULL\n") ;
        return (GrB_NULL_POINTER) ;
    }

    GB_CHECK_MAGIC (A, kind) ;

    //--------------------------------------------------------------------------
    // print the header
    //--------------------------------------------------------------------------

    if (is_full)
    {   GB_cov[3496]++ ;
// covered (3496): 163
        GBPR0 (", full") ;
    }
    else if (is_bitmap)
    {   GB_cov[3497]++ ;
// covered (3497): 1360
        GBPR0 (", bitmap") ;
    }
    else if (is_sparse)
    {   GB_cov[3498]++ ;
// covered (3498): 88358
        GBPR0 (", sparse") ;
    }
    else if (is_hyper)
    {   GB_cov[3499]++ ;
// covered (3499): 6522
        GBPR0 (", hypersparse") ;
    }
    else
    {   GB_cov[3500]++ ;
// NOT COVERED (3500):
GB_GOTCHA ;
        GBPR0 (" invalid structure\n") ;
        return (GrB_INVALID_OBJECT) ;
    }
    if (A->jumbled)
    {   GB_cov[3501]++ ;
// covered (3501): 2
        GBPR0 (" (jumbled)") ;
    }
    GBPR0 (" %s\n", A->is_csc ? "by col" : "by row") ;

    #if GB_DEVELOPER
    GBPR0 ("  max # entries: " GBd "\n", A->nzmax) ;
    GBPR0 ("  vlen: " GBd , A->vlen) ;
    if (A->nvec_nonempty != -1)                                 // TODO::OK
    {
        GBPR0 (" nvec_nonempty: " GBd , A->nvec_nonempty) ;     // TODO::OK
    }
    GBPR0 (" nvec: " GBd " plen: " GBd  " vdim: " GBd " hyper_switch %g\n",
        A->nvec, A->plen, A->vdim, A->hyper_switch) ;
    #endif

    switch (A->sparsity)
    {

        // 1
        case GxB_HYPERSPARSE  : GB_cov[3502]++ ;  
// covered (3502): 108
            GBPR0 ("  sparsity control: hypersparse only\n") ;
            break ;

        // 2
        case GxB_SPARSE  : GB_cov[3503]++ ;  
// covered (3503): 64
            GBPR0 ("  sparsity control: sparse only\n") ;
            break ;

        // 3
        case GxB_HYPERSPARSE + GxB_SPARSE  : GB_cov[3504]++ ;  
// covered (3504): 52
            GBPR0 ("  sparsity control: sparse/hypersparse\n") ;
            break ;

        // 4
        case GxB_BITMAP  : GB_cov[3505]++ ;  
// covered (3505): 52
            GBPR0 ("  sparsity control: bitmap only\n") ;
            break ;

        // 5
        case GxB_HYPERSPARSE + GxB_BITMAP  : GB_cov[3506]++ ;  
// covered (3506): 52
            GBPR0 ("  sparsity control: hypersparse/bitmap\n") ;
            break ;

        // 6
        case GxB_SPARSE + GxB_BITMAP  : GB_cov[3507]++ ;  
// covered (3507): 52
            GBPR0 ("  sparsity control: sparse/bitmap\n") ;
            break ;

        // 7
        case GxB_HYPERSPARSE + GxB_SPARSE + GxB_BITMAP  : GB_cov[3508]++ ;  
// covered (3508): 52
            GBPR0 ("  sparsity control: hypersparse/sparse/bitmap\n") ;
            break ;

        // 8 and 12: these options are treated the same
        case GxB_FULL  : GB_cov[3509]++ ;  
// covered (3509): 52
        case GxB_FULL + GxB_BITMAP  : GB_cov[3510]++ ;  
// covered (3510): 104
            GBPR0 ("  sparsity control: bitmap/full\n") ;
            break ;

        // 9
        case GxB_HYPERSPARSE + GxB_FULL  : GB_cov[3511]++ ;  
// covered (3511): 52
            GBPR0 ("  sparsity control: hypersparse/full\n") ;
            break ;

        // 10
GB_GOTCHA ;
        case GxB_SPARSE + GxB_FULL  : GB_cov[3512]++ ;  
// covered (3512): 52
            GBPR0 ("  sparsity control: sparse/full\n") ;
            break ;

        // 11
GB_GOTCHA ;
        case GxB_HYPERSPARSE + GxB_SPARSE + GxB_FULL  : GB_cov[3513]++ ;  
// covered (3513): 52
            GBPR0 ("  sparsity control: hypersparse/sparse/full\n") ;
            break ;

        // 13
GB_GOTCHA ;
        case GxB_HYPERSPARSE + GxB_BITMAP + GxB_FULL  : GB_cov[3514]++ ;  
// covered (3514): 52
            GBPR0 ("  sparsity control: hypersparse/bitmap/full\n") ;
            break ;

        // 14
GB_GOTCHA ;
        case GxB_SPARSE + GxB_BITMAP + GxB_FULL  : GB_cov[3515]++ ;  
// covered (3515): 52
            GBPR0 ("  sparsity control: sparse/bitmap/full\n") ;
            break ;

        // 15
        case GxB_AUTO_SPARSITY  : GB_cov[3516]++ ;  
// covered (3516): 95607
            #if GB_DEVELOPER
            GBPR0 ("  sparsity control: auto\n") ;
            #endif
            break ;

        default  : GB_cov[3517]++ ;  
// NOT COVERED (3517):
GB_GOTCHA ;
            GBPR0 ("  sparsity control: invalid\n") ;
            return (GrB_INVALID_OBJECT) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // check the dimensions
    //--------------------------------------------------------------------------

    if (A->vlen < 0 || A->vlen > GxB_INDEX_MAX ||
        A->vdim < 0 || A->vdim > GxB_INDEX_MAX ||
        A->nzmax < 0 || A->nzmax > GxB_INDEX_MAX)
    {   GB_cov[3518]++ ;
// covered (3518): 2
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
        {   GB_cov[3519]++ ;
// NOT COVERED (3519):
GB_GOTCHA ;
            GBPR0 ("  invalid full %s structure\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }
    else if (is_bitmap) 
    {
        // A is bitmap
        if (! (A->nvec == A->vdim && A->plen == -1 &&
               A->h == NULL && A->p == NULL && A->i == NULL))
        {   GB_cov[3520]++ ;
// NOT COVERED (3520):
GB_GOTCHA ;
            GBPR0 ("  invalid bitmap %s structure\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }
    else if (is_sparse)
    {
        // A is sparse
        if (! (A->nvec == A->plen && A->plen == A->vdim))
        {   GB_cov[3521]++ ;
// covered (3521): 4
            GBPR0 ("  invalid sparse %s structure\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }
    else // if (is_hyper)
    {
        // A is hypersparse
        if (! (A->nvec >= 0 && A->nvec <= A->plen && A->plen <= A->vdim))
        {   GB_cov[3522]++ ;
// covered (3522): 2
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
    {   GB_cov[3523]++ ;
// covered (3523): 2
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
        {   GB_cov[3524]++ ;
// covered (3524): 2
            GBPR0 ("  ->p is NULL, invalid %s\n", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    //--------------------------------------------------------------------------
    // check a non-empty matrix
    //--------------------------------------------------------------------------

    bool A_empty = (A->nzmax == 0) ;
    if (is_hyper || is_sparse)
    {
        if (!A_empty && A->i == NULL)
        {   GB_cov[3525]++ ;
// covered (3525): 2
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
        {   GB_cov[3526]++ ;
// covered (3526): 6
            GBPR0 ("  ->p [0] = " GBd " invalid\n", A->p [0]) ;
            return (GrB_INVALID_OBJECT) ;
        }

        for (int64_t j = 0 ; j < A->nvec ; j++)
        {
            if (A->p [j+1] < A->p [j] || A->p [j+1] > A->nzmax)
            {   GB_cov[3527]++ ;
// covered (3527): 4
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
            {   GB_cov[3528]++ ;
// covered (3528): 8
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
    {   GB_cov[3529]++ ;
// covered (3529): 95241
        GBPR0 ("%s, ", GB_NAME) ;
    }

    // # of entries cannot be computed until all the tests above are OK
    int64_t anz = is_full ? GB_NNZ_FULL (A) : GB_NNZ (A) ;   // TODO
    if (anz == 0)
    {   GB_cov[3530]++ ;
// covered (3530): 3116
        GBPR0 ("no entries\n") ;
    }
    else if (anz == 1)
    {   GB_cov[3531]++ ;
// covered (3531): 88572
        GBPR0 ("1 entry\n") ;
    }
    else
    {   GB_cov[3532]++ ;
// covered (3532): 4683
        GBPR0 ( GBd " entries\n", anz) ;
    }

    //--------------------------------------------------------------------------
    // report the number of pending tuples and zombies
    //--------------------------------------------------------------------------

    if (Pending != NULL || A->nzombies != 0)
    {   GB_cov[3533]++ ;
// covered (3533): 50
        GBPR0 ("  pending tuples: " GBd " max pending: " GBd 
            " zombies: " GBd "\n", GB_Pending_n (A),
            (Pending == NULL) ? 0 : (Pending->nmax),
            A->nzombies) ;
    }

    if (!ignore_zombies && (A->nzombies < 0 || A->nzombies > anz))
    {   GB_cov[3534]++ ;
// covered (3534): 4
        GBPR0 ("  invalid number of zombies: " GBd " "
            "must be >= 0 and <= # entries (" GBd ")\n", A->nzombies, anz) ;
        return (GrB_INVALID_OBJECT) ;
    }

    if (is_full || is_bitmap)
    {
        if (A->nzombies != 0)
        {   GB_cov[3535]++ ;
// NOT COVERED (3535):
GB_GOTCHA ;
            GBPR0 ("  %s %s cannot have zombies\n",
                is_full ? "full" : "bitmap", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
        if (Pending != NULL)
        {   GB_cov[3536]++ ;
// NOT COVERED (3536):
GB_GOTCHA ;
            GBPR0 ("  %s %s cannot have pending tuples\n",
                is_full ? "full" : "bitmap", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
        if (A->jumbled)
        {   GB_cov[3537]++ ;
// NOT COVERED (3537):
GB_GOTCHA ;
            GBPR0 ("  %s %s cannot be jumbled\n",
                is_full ? "full" : "bitmap", kind) ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    //--------------------------------------------------------------------------
    // check and print the row indices and numerical values
    //--------------------------------------------------------------------------

    if (anz > 0 && !phantom) GBPR0 ("\n") ;

    #define GB_NBRIEF 10
    #define GB_NZBRIEF 30

    int64_t nzombies = 0 ;
    int64_t icount = 0 ;
    bool truncated = false ;
    int64_t anz_actual = 0 ;

    // for each vector of A
    for (int64_t k = 0 ; k < A->nvec ; k++)
    {
        if (phantom) break ;
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
                int8_t ab = A->b [p2] ;
                if (ab < 0 || ab > 1)
                {   GB_cov[3538]++ ;
// NOT COVERED (3538):
GB_GOTCHA ;
                    GBPR0 ("invalid bitmap\n") ;
                    return (GrB_INVALID_OBJECT) ;
                }
                ajnz += (ab != 0)  ;
            }
        }

        bool prcol = ((pr_short && !truncated) || pr_complete) ;
        // print the header for vector j
        if (prcol)
        {   GB_cov[3539]++ ;
// covered (3539): 21319
            #if GB_DEVELOPER
            GBPR ("  %s: " GBd " : " GBd " entries [" GBd ":" GBd "]\n",
                A->is_csc ? "column" : "row", j, ajnz, p, pend-1) ;
            if (pr_short && k == GB_NBRIEF) truncated = true ;
            #endif
        }

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
            bool print_value = false ;
            if (prcol)
            {   GB_cov[3540]++ ;
// covered (3540): 18721
                if ((pr_short && icount < GB_NZBRIEF) || pr_complete)
                {   GB_cov[3541]++ ;
// covered (3541): 18325
                    print_value = true ;
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
                {   GB_cov[3542]++ ;
// covered (3542): 134
                    truncated = true ;
                }
            }
            int64_t row = A->is_csc ? i : j ;
            int64_t col = A->is_csc ? j : i ;
            if (i < 0 || i >= A->vlen)
            {   GB_cov[3543]++ ;
// covered (3543): 2
                GBPR0 ("  index (" GBd "," GBd ") out of range\n", row, col) ;
                return (GrB_INVALID_OBJECT) ;
            }

            // print the value
            if (print_value)
            {   GB_cov[3544]++ ;
// covered (3544): 18323
                if (is_zombie)
                {   GB_cov[3545]++ ;
// covered (3545): 4
                    GBPR ("zombie") ;
                }
                else if (A->x != NULL)
                {   GB_cov[3546]++ ;
// covered (3546): 18299
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
            {   GB_cov[3547]++ ;
// covered (3547): 10
                // indices unsorted, or duplicates present
                GBPR0 (" index (" GBd "," GBd ") invalid\n", row, col) ;
                return (GrB_INDEX_OUT_OF_BOUNDS) ;
            }

            if (print_value)
            {   GB_cov[3548]++ ;
// covered (3548): 18317
                GBPR ("\n") ;
            }
            ilast = i ;
        }
    }

    if (pr_short && truncated) GBPR ("    ...\n") ;

    //--------------------------------------------------------------------------
    // check the entry count in the bitmap
    //--------------------------------------------------------------------------

    if (is_bitmap && anz != anz_actual)
    {   GB_cov[3549]++ ;
// NOT COVERED (3549):
GB_GOTCHA ;
        // this case can only occur for bitmapped matrices
        GBPR0 ("  invalid bitmap count: " GBd " exist but"
            " A->nvals = " GBd "\n", anz_actual, anz) ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // check the zombie count
    //--------------------------------------------------------------------------

    if (!ignore_zombies && nzombies != A->nzombies)
    {   GB_cov[3550]++ ;
// covered (3550): 2
        GBPR0 ("  invalid zombie count: " GBd " exist but"
            " A->nzombies = " GBd "\n", nzombies, A->nzombies) ;
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // check and print the pending tuples
    //--------------------------------------------------------------------------

    #if GB_DEVELOPER
    if ((pr_short || pr_complete) && (is_sparse || is_hyper))
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
        {   GB_cov[3551]++ ;
// covered (3551): 2
            GBPR0 ("  invalid pending count\n") ;
            return (GrB_INVALID_OBJECT) ;
        }

        // matrix has tuples, arrays and type must not be NULL
        if (Pending->i == NULL || Pending->x == NULL ||
            (A->vdim > 1 && Pending->j == NULL))
        {   GB_cov[3552]++ ;
// covered (3552): 2
            GBPR0 ("  invalid pending tuples\n") ;
            return (GrB_INVALID_OBJECT) ;
        }

        GBPR0 ("  pending tuples:\n") ;

        info = GB_Type_check (Pending->type, "", pr, f) ;
        if (info != GrB_SUCCESS || (Pending->type->size != Pending->size))
        {   GB_cov[3553]++ ;
// NOT COVERED (3553):
GB_GOTCHA ;
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
            {   GB_cov[3554]++ ;
// covered (3554): 76
                GBPR ("    row: " GBd " col: " GBd " ", row, col) ;
                info = GB_entry_check (Pending->type,
                    Pending->x +(k * Pending->type->size), pr, f) ;
                if (info != GrB_SUCCESS) return (info) ;
                GBPR ("\n") ;
            }

            if (i < 0 || i >= A->vlen || j < 0 || j >= A->vdim)
            {   GB_cov[3555]++ ;
// covered (3555): 2
                GBPR0 ("    tuple (" GBd "," GBd ") out of range\n", row, col) ;
                return (GrB_INVALID_OBJECT) ;
            }

            sorted = sorted && ((jlast < j) || (jlast == j && ilast <= i)) ;
            ilast = i ;
            jlast = j ;
        }

        if (sorted != Pending->sorted)
        {   GB_cov[3556]++ ;
// covered (3556): 4
            GBPR0 ("  invalid pending tuples: invalid sort\n") ;
            return (GrB_INVALID_OBJECT) ;
        }

        if (Pending->op == NULL)
        {   GB_cov[3557]++ ;
// covered (3557): 24
            GBPR0 ("  pending operator: implicit 2nd\n") ;
        }
        else
        {
            info = GB_BinaryOp_check (Pending->op, "pending operator:", pr, f) ;
            if (info != GrB_SUCCESS)
            {   GB_cov[3558]++ ;
// covered (3558): 2
                GBPR0 ("  invalid pending operator\n") ;
                return (GrB_INVALID_OBJECT) ;
            }
        }
    }

    if (pr_complete)
    {   GB_cov[3559]++ ;
// covered (3559): 257
        GBPR ("\n") ;
    }

    //--------------------------------------------------------------------------
    // check nvec_nonempty
    //--------------------------------------------------------------------------

    // A->nvec_nonempty == -1 denotes that the value has not been computed.
    // This is valid, and can occur for matrices imported with
    // GxB_Matrix_import*, and in other cases when its computation is postponed
    // or not needed.  If not -1, however, the value must be correct.

    int64_t actual_nvec_nonempty = GB_nvec_nonempty (A, NULL) ; // OK:non-atomic

    if (! ((A->nvec_nonempty == actual_nvec_nonempty) ||    // TODO::OK
           (A->nvec_nonempty == -1)))                       // TODO::OK
    {   GB_cov[3560]++ ;
// NOT COVERED (3560):
GB_GOTCHA ;
        GBPR0 ("  invalid count of non-empty vectors\n"
            "A->nvec_nonempty = " GBd " actual " GBd "\n",
            A->nvec_nonempty, actual_nvec_nonempty) ;       // TODO::OK
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

