//------------------------------------------------------------------------------
// GB_bitmap_M_scatter_whole: scatter M into/from the C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_bitmap_assign_methods.h"

void GB_bitmap_M_scatter_whole  // scatter M into the C bitmap
(
    // input/output:
    GrB_Matrix C,
    // int64_t *cnvals_handle,     // cnvals = C->nvals
    // inputs:
    const GrB_Matrix M,         // mask to scatter into the C bitmap
    const bool Mask_struct,     // true if M is structural, false if valued
    const int operation,        // +=2, -=2, or %=2
    const int64_t *GB_RESTRICT pstart_Mslice, // size ntasks+1
    const int64_t *GB_RESTRICT kfirst_Mslice, // size ntasks
    const int64_t *GB_RESTRICT klast_Mslice,  // size ntasks
    const int mthreads,
    const int mtasks,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (M, "M for bitmap scatter, whoe", GB0) ;
    ASSERT (GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M)) ;

    //--------------------------------------------------------------------------
    // get C and M
    //--------------------------------------------------------------------------

    GB_GET_M ;
    int8_t *Cb = C->b ;
    const int64_t cvlen = C->vlen ;
    int64_t cnvals = 0 ; // (*cnvals_handle) ;

    //--------------------------------------------------------------------------
    // scatter M into the C bitmap
    //--------------------------------------------------------------------------

    switch (operation)
    {

        case GB_BITMAP_M_SCATTER_PLUS_2 :       // Cb (i,j) += 2

            #undef  GB_MASK_WORK
            #define GB_MASK_WORK(pC) Cb [pC] += 2
            #include "GB_bitmap_assign_M_all_template.c"
            break ;

        case GB_BITMAP_M_SCATTER_MINUS_2 :      // Cb (i,j) -= 2

            #undef  GB_MASK_WORK
            #define GB_MASK_WORK(pC) Cb [pC] -= 2
            #include "GB_bitmap_assign_M_all_template.c"
            break ;

        case GB_BITMAP_M_SCATTER_MOD_2 :        // Cb (i,j) %= 2

            // TODO unused so far
            #undef  GB_MASK_WORK
            #define GB_MASK_WORK(pC) Cb [pC] %= 2
            #include "GB_bitmap_assign_M_all_template.c"
            break ;

        default: ;
    }

    // (*cnvals_handle) = cnvals ;
}

