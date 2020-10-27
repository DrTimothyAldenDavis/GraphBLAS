//------------------------------------------------------------------------------
// GB_bitmap_assign: assign to C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Implements GrB_Row_assign, GrB_Col_assign, GrB_assign, GxB_subassign when C
// is in bitmap form, or when C is converted into bitmap form.

// TODO NOW: add special cases when Ikind and Jkind are both GB_ALL, for assign
// and subassign (not row/col assign).  The assign_kind can be ignored in this
// case.  Handle by matrix and scalar assignment.  This may impact performance
// for the GAP benchmarks.

#include "GB_bitmap_assign_methods.h"
#include "GB_dense.h"

#define GB_FREE_ALL GB_phbix_free (C) ;

GrB_Info GB_bitmap_assign
(
    // input/output:
    GrB_Matrix C,               // input/output matrix
    // inputs:
    const bool C_replace,       // descriptor for C
    const GrB_Index *I,         // I index list
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,         // J index list
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,         // mask matrix, NULL if not present
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
    const GrB_BinaryOp accum,   // present here
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    const int assign_kind,      // row assign, col assign, assign, or subassign
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C for bitmap assign", GB0) ;

    //--------------------------------------------------------------------------
    // ensure C is in bitmap form
    //--------------------------------------------------------------------------

    GB_OK (GB_convert_any_to_bitmap (C, Context)) ;
    ASSERT (GB_IS_BITMAP (C)) ;

    //--------------------------------------------------------------------------
    // do the assignment
    //--------------------------------------------------------------------------

    if (M == NULL)
    {
        if (accum == NULL)
        { 
            // C(I,J) = A or scalar, no mask
            GB_OK (GB_bitmap_assign_noM_noaccum (C, C_replace,
                I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                /* no M, */ Mask_comp, Mask_struct, /* no accum, */
                A, scalar, scalar_type, assign_kind, Context)) ;
        }
        else
        { 
            // C(I,J) += A or scalar, no mask
            GB_OK (GB_bitmap_assign_noM_accum (C, C_replace,
                I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                /* no M, */ Mask_comp, Mask_struct, accum,
                A, scalar, scalar_type, assign_kind, Context)) ;
        }
    }
    else if (GB_IS_BITMAP (M) || GB_IS_FULL (M))
    {
        if (accum == NULL)
        { 
            // C<M or !M, where M is full>(I,J) = A or scalar
            GB_OK (GB_bitmap_assign_fullM_noaccum (C, C_replace,
                I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                M, Mask_comp, Mask_struct, /* no accum, */
                A, scalar, scalar_type, assign_kind, Context)) ;
        }
        else
        { 
            // C<M or !M, where M is full>(I,J) = A or scalar
            GB_OK (GB_bitmap_assign_fullM_accum (C, C_replace,
                I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                M, Mask_comp, Mask_struct, accum,
                A, scalar, scalar_type, assign_kind, Context)) ;
        }
    }
    else if (!Mask_comp)
    {
        if (accum == NULL)
        { 
            // C<M>(I,J) = A or scalar, M is sparse or hypersparse
            GB_OK (GB_bitmap_assign_M_noaccum (C, C_replace,
                I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                M, /* Mask_comp false, */ Mask_struct, /* no accum, */
                A, scalar, scalar_type, assign_kind, Context)) ;
        }
        else
        { 
            // C<M>(I,J) += A or scalar, M is sparse or hypersparse
            GB_OK (GB_bitmap_assign_M_accum (C, C_replace,
                I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                M, /* Mask_comp false, */ Mask_struct, accum,
                A, scalar, scalar_type, assign_kind, Context)) ;
        }
    }
    else // Mask_comp is true
    {
        if (accum == NULL)
        { 
            // C<!M>(I,J) = A or scalar, M is sparse or hypersparse
            GB_OK (GB_bitmap_assign_notM_noaccum (C, C_replace,
                I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                M, /* Mask_comp true, */ Mask_struct, /* no accum, */
                A, scalar, scalar_type, assign_kind, Context)) ;
        }
        else
        { 
            // C<!M>(I,J) += A or scalar, M is sparse or hypersparse
            GB_OK (GB_bitmap_assign_notM_accum (C, C_replace,
                I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon,
                M, /* Mask_comp true, */ Mask_struct, accum,
                A, scalar, scalar_type, assign_kind, Context)) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "final C for bitmap assign", GB0) ;
    ASSERT (GB_IS_BITMAP (C)) ;
    return (GrB_SUCCESS) ;
}

