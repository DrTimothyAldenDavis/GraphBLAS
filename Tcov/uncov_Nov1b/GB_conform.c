//------------------------------------------------------------------------------
// GB_conform: conform any matrix to its desired sparsity structure
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// On input, the matrix has any one of four sparsity structures: hypersparse,
// sparse, bitmap, or full.  A bitmap or full matrix never has pending work.  A
// sparse or hypersparse matrix may have pending work (zombies, jumbled, and/or
// pending tuples).  The pending work is not finished unless the matrix is
// converted to bitmap or full.  If this method fails, the matrix is cleared
// of all entries.

#include "GB.h"

#define GB_FREE_ALL GB_phbix_free (A) ;

//------------------------------------------------------------------------------
// GB_hyper_or_bitmap: ensure a matrix is either hypersparse or bitmap
//------------------------------------------------------------------------------

static inline GrB_Info GB_hyper_or_bitmap
(
    bool is_hyper, bool is_sparse, bool is_bitmap, bool is_full,
    GrB_Matrix A, GB_Context Context
)
{
    GrB_Info info ;
    if (is_full || ((is_hyper || is_sparse) &&
        GB_convert_sparse_to_bitmap_test (A->bitmap_switch,
            GB_NNZ (A), A->vlen, A->vdim)))
    {   GB_cov[2979]++ ;
// covered (2979): 39
        // if full or sparse/hypersparse with many entries: to bitmap
        GB_OK (GB_convert_any_to_bitmap (A, Context)) ;
    }
    else if (is_sparse || (is_bitmap &&
        GB_convert_bitmap_to_sparse_test (A->bitmap_switch,
            GB_NNZ (A), A->vlen, A->vdim)))
    {   GB_cov[2980]++ ;
// covered (2980): 26
        // if sparse or bitmap with few entries: to hypersparse
        GB_OK (GB_convert_any_to_hyper (A, Context)) ;
    }
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_sparse_or_bitmap: ensure a matrix is either sparse or bitmap
//------------------------------------------------------------------------------

static inline GrB_Info GB_sparse_or_bitmap
(
    bool is_hyper, bool is_sparse, bool is_bitmap, bool is_full,
    GrB_Matrix A, GB_Context Context
)
{
    GrB_Info info ;
    if (is_full || ((is_hyper || is_sparse) &&
        GB_convert_sparse_to_bitmap_test (A->bitmap_switch,
            GB_NNZ (A), A->vlen, A->vdim)))
    {   GB_cov[2981]++ ;
// covered (2981): 495682
        // if full or sparse/hypersparse with many entries: to bitmap
        GB_OK (GB_convert_any_to_bitmap (A, Context)) ;
    }
    else if (is_hyper || (is_bitmap &&
        GB_convert_bitmap_to_sparse_test (A->bitmap_switch,
            GB_NNZ (A), A->vlen, A->vdim)))
    {   GB_cov[2982]++ ;
// covered (2982): 103091
        // if hypersparse or bitmap with few entries: to sparse
        GB_OK (GB_convert_any_to_sparse (A, Context)) ;
    }
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_hyper_sparse_or_bitmap: ensure a matrix is hypersparse, sparse, or bitmap
//------------------------------------------------------------------------------

static inline GrB_Info GB_hyper_sparse_or_bitmap
(
    bool is_hyper, bool is_sparse, bool is_bitmap, bool is_full,
    GrB_Matrix A, GB_Context Context
)
{
    GrB_Info info ;
    if (is_full || ((is_hyper || is_sparse) &&
        GB_convert_sparse_to_bitmap_test (A->bitmap_switch,
            GB_NNZ (A), A->vlen, A->vdim)))
    {   GB_cov[2983]++ ;
// covered (2983): 11913883
        // if full or sparse/hypersparse with many entries: to bitmap
        GB_OK (GB_convert_any_to_bitmap (A, Context)) ;
    }
    else if (is_bitmap)
    {
        if (GB_convert_bitmap_to_sparse_test (A->bitmap_switch,
            GB_NNZ (A), A->vlen, A->vdim))
        {   GB_cov[2984]++ ;
// covered (2984): 3139
            // if bitmap with few entries: to sparse
            GB_OK (GB_convert_bitmap_to_sparse (A, Context)) ;
            // conform between sparse and hypersparse
            GB_OK (GB_conform_hyper (A, Context)) ;
        }
    }
    else // is_hyper || is_sparse
    {   GB_cov[2985]++ ;
// covered (2985): 1223870
        // conform between sparse and hypersparse
        GB_OK (GB_conform_hyper (A, Context)) ;
    }
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_conform
//------------------------------------------------------------------------------

GrB_Info GB_conform     // conform a matrix to its desired sparsity structure
(
    GrB_Matrix A,       // matrix to conform
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A to conform", GB0) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;
    bool is_hyper = GB_IS_HYPERSPARSE (A) ;
    bool is_sparse = GB_IS_SPARSE (A) ;
    bool is_full = GB_IS_FULL (A) ;
    bool is_bitmap = GB_IS_BITMAP (A) ;
    bool is_full_or_dense_with_no_pending_work = is_full || (GB_is_dense (A)
        && !GB_ZOMBIES (A) && !GB_JUMBLED (A) && !GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // select the sparsity structure
    //--------------------------------------------------------------------------

    int sparsity = A->sparsity ;
    if (A->vdim <= 1)
    {
        if (sparsity & GxB_HYPERSPARSE)
        {   GB_cov[2986]++ ;
// covered (2986): 2102567
            // a GxB_Scalar, GrB_Vector, or a GrB_Matrix with a single vector,
            // cannot be converted to hypersparse.  If the sparsity control
            // allows for the hypersparse case, disable it and enable the
            // sparse case instead.
            sparsity = sparsity & (GxB_FULL + GxB_BITMAP + GxB_SPARSE) ;
            sparsity = sparsity | GxB_SPARSE ;
        }
    }

    switch (sparsity)
    {

        //----------------------------------------------------------------------
        // (1) always hypersparse
        //----------------------------------------------------------------------

        case GxB_HYPERSPARSE  : GB_cov[2987]++ ;  
// covered (2987): 7725260

            GB_OK (GB_convert_any_to_hyper (A, Context)) ;
            break ;

        //----------------------------------------------------------------------
        // (2) always sparse
        //----------------------------------------------------------------------

        case GxB_SPARSE  : GB_cov[2988]++ ;  
// covered (2988): 440683

            GB_OK (GB_convert_any_to_sparse (A, Context)) ;
            break ;

        //----------------------------------------------------------------------
        // (3) sparse or hypersparse
        //----------------------------------------------------------------------

        case GxB_HYPERSPARSE + GxB_SPARSE  : GB_cov[2989]++ ;  
// covered (2989): 104

            if (is_full || is_bitmap)
            {   GB_cov[2990]++ ;
// covered (2990): 13
                // if full or bitmap: to sparse
                GB_OK (GB_convert_any_to_sparse (A, Context)) ;
            }
            // conform between sparse and hypersparse
            GB_OK (GB_conform_hyper (A, Context)) ;
            break ;

        //----------------------------------------------------------------------
        // (4) always bitmap
        //----------------------------------------------------------------------

        case GxB_BITMAP  : GB_cov[2991]++ ;  
// covered (2991): 288668

            GB_OK (GB_convert_any_to_bitmap (A, Context)) ;
            break ;

        //----------------------------------------------------------------------
        // (5) hypersparse or bitmap
        //----------------------------------------------------------------------

        case GxB_HYPERSPARSE + GxB_BITMAP  : GB_cov[2992]++ ;  
// covered (2992): 78

            // ensure the matrix is hypersparse or bitmap
            GB_OK (GB_hyper_or_bitmap (is_hyper, is_sparse, is_bitmap,
                is_full, A, Context)) ;
            break ;

        //----------------------------------------------------------------------
        // (6) sparse or bitmap
        //----------------------------------------------------------------------

        case GxB_SPARSE + GxB_BITMAP  : GB_cov[2993]++ ;  
// covered (2993): 78

            // ensure the matrix is sparse or bitmap
            GB_OK (GB_sparse_or_bitmap (is_hyper, is_sparse, is_bitmap,
                is_full, A, Context)) ;
            break ;

        //----------------------------------------------------------------------
        // (7) hypersparse, sparse, or bitmap
        //----------------------------------------------------------------------

        case GxB_HYPERSPARSE + GxB_SPARSE + GxB_BITMAP  : GB_cov[2994]++ ;  
// covered (2994): 78

            // ensure the matrix is hypersparse, sparse, or bitmap
            GB_OK (GB_hyper_sparse_or_bitmap (is_hyper, is_sparse,
                is_bitmap, is_full, A, Context)) ;
            break ;

        //----------------------------------------------------------------------
        // (8), (12): bitmap or full
        //----------------------------------------------------------------------

        case GxB_FULL  : GB_cov[2995]++ ;  
// covered (2995): 52
        case GxB_FULL + GxB_BITMAP  : GB_cov[2996]++ ;  
// covered (2996): 104

            if (is_full_or_dense_with_no_pending_work)
            {   GB_cov[2997]++ ;
// covered (2997): 26
                // if full or all entries present: to full
                GB_convert_any_to_full (A) ;
            }
            else
            {   GB_cov[2998]++ ;
// covered (2998): 78
                // otherwise: to bitmap
                GB_OK (GB_convert_any_to_bitmap (A, Context)) ;
            }
            break ;

        //----------------------------------------------------------------------
        // (9) hypersparse or full
        //----------------------------------------------------------------------

        case GxB_HYPERSPARSE + GxB_FULL  : GB_cov[2999]++ ;  
// covered (2999): 91

            if (is_full_or_dense_with_no_pending_work)
            {   GB_cov[3000]++ ;
// covered (3000): 13
                // if all entries present: to full
                GB_convert_any_to_full (A) ;
            }
            else
            {   GB_cov[3001]++ ;
// covered (3001): 78
                // otherwise: to hypersparse
                GB_OK (GB_convert_any_to_hyper (A, Context)) ;
            }
            break ;

        //----------------------------------------------------------------------
        // (10) sparse or full
        //----------------------------------------------------------------------

        case GxB_SPARSE + GxB_FULL  : GB_cov[3002]++ ;   
// covered (3002): 3561657

            if (is_full_or_dense_with_no_pending_work)
            {   GB_cov[3003]++ ;
// covered (3003): 755595
                // if full or all entries present: to full
                GB_convert_any_to_full (A) ;
            }
            else
            {   GB_cov[3004]++ ;
// covered (3004): 2806062
                // otherwise: to sparse
                GB_OK (GB_convert_any_to_sparse (A, Context)) ;
            }
            break ;

        //----------------------------------------------------------------------
        // (11) hypersparse, sparse, or full
        //----------------------------------------------------------------------

        case GxB_HYPERSPARSE + GxB_SPARSE + GxB_FULL  : GB_cov[3005]++ ;  
// covered (3005): 91

            if (is_full_or_dense_with_no_pending_work)
            {   GB_cov[3006]++ ;
// covered (3006): 13
                // if full or all entries present: to full
                GB_convert_any_to_full (A) ;
            }
            else if (is_bitmap)
            {   GB_cov[3007]++ ;
// NOT COVERED (3007):
GB_GOTCHA ;
                // if bitmap: to sparse
                GB_OK (GB_convert_bitmap_to_sparse (A, Context)) ;
                // conform between sparse and hypersparse
                GB_OK (GB_conform_hyper (A, Context)) ;
            }
            else
            {   GB_cov[3008]++ ;
// covered (3008): 78
                // conform between sparse and hypersparse
                GB_OK (GB_conform_hyper (A, Context)) ;
            }
            break ;

        //----------------------------------------------------------------------
        // (13) hypersparse, bitmap, or full
        //----------------------------------------------------------------------

        case GxB_HYPERSPARSE + GxB_BITMAP + GxB_FULL  : GB_cov[3009]++ ;  
// covered (3009): 78

            if (is_full_or_dense_with_no_pending_work)
            {   GB_cov[3010]++ ;
// covered (3010): 13
                // if full or all entries present: to full
                GB_convert_any_to_full (A) ;
            }
            else
            {   GB_cov[3011]++ ;
// covered (3011): 65
                // ensure the matrix is hypersparse or bitmap
                GB_OK (GB_hyper_or_bitmap (is_hyper, is_sparse, is_bitmap,
                    is_full, A, Context)) ;
            }
            break ;

        //----------------------------------------------------------------------
        // (14) sparse, bitmap, or full
        //----------------------------------------------------------------------

        // This is the default case for a GrB_Matrix with one vector,
        // a GrB_Vector, or GxB_scalar.

        case GxB_SPARSE + GxB_BITMAP + GxB_FULL  : GB_cov[3012]++ ;  
// covered (3012): 1947824

            if (is_full_or_dense_with_no_pending_work)
            {   GB_cov[3013]++ ;
// covered (3013): 925736
                // if full or all entries present: to full
                GB_convert_any_to_full (A) ;
            }
            else
            {   GB_cov[3014]++ ;
// covered (3014): 1022088
                // ensure the matrix is sparse or bitmap
                GB_OK (GB_sparse_or_bitmap (is_hyper, is_sparse, is_bitmap,
                    is_full, A, Context)) ;
            }
            break ;

        //----------------------------------------------------------------------
        // (15) default: hypersparse, sparse, bitmap, or full
        //----------------------------------------------------------------------

        // This is the default case for a GrB_Matrix with more than one vector.

        case GxB_AUTO_SPARSITY  : GB_cov[3015]++ ;  
// covered (3015): 19296473
        default:

            if (is_full_or_dense_with_no_pending_work)
            {   GB_cov[3016]++ ;
// covered (3016): 263301
                // if full or all entries present: to full
                GB_convert_any_to_full (A) ;
            }
            else
            {   GB_cov[3017]++ ;
// covered (3017): 19033172
                // ensure the matrix is hypersparse, sparse, or bitmap
                GB_OK (GB_hyper_sparse_or_bitmap (is_hyper, is_sparse,
                    is_bitmap, is_full, A, Context)) ;
            }
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A conformed", GB0) ;
    return (GrB_SUCCESS) ;
}

