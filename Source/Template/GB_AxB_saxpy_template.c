//------------------------------------------------------------------------------
// GB_AxB_saxpy_template: C=A*B, C<M>=A*B, or C<!M>=A*B via saxpy method
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// All 4 matrices have any format: hypersparse, sparse, bitmap, or full.

{
    switch (saxpy_method)
    {

        case GB_SAXPY_METHOD_3 :
        { 
            // C is sparse or hypersparse, using minimal workspace.
            ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;
            #include "GB_AxB_saxpy3_template.c"
        }
        break ;

        case GB_SAXPY_METHOD_4 :
        { 
            // C is sparse or hypersparse, using large workspace
            ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;
            ASSERT (GB_IS_SPARSE (A)) ;
            ASSERT (GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B)) ;
            #include "GB_AxB_saxpy4_template.c"
        }
        break ;

        case GB_SAXPY_METHOD_BITMAP :
        { 
            // C is bitmap or full
            #include "GB_bitmap_AxB_saxpy_template.c"
        }

        default:;
    }
}

