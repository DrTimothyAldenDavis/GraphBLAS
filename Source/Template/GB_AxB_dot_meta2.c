//------------------------------------------------------------------------------
// GB_AxB_dot_meta2:  C=A'B, C<!M>=A'*B, or C<M>=A'*B via dot products
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// TODO: rename GB_AxB_dot_meta9.c for the 9 cases

// TODO: use 16 cases, not 9, but disable the hypersparse cases
// for dot2

// TODO: add dot3_phase1_template

{

    if (A_is_sparse_or_hyper)
    {
        if (B_is_sparse_or_hyper)
        { 
            // both A and B are sparse/hyper
            #define GB_A_IS_SPARSE_OR_HYPER 1
            #define GB_A_IS_BITMAP          0
            #define GB_A_IS_FULL            0
            #define GB_B_IS_SPARSE_OR_HYPER 1
            #define GB_B_IS_BITMAP          0
            #define GB_B_IS_FULL            0
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
        else if (B_is_bitmap)
        { 
            // A is sparse/hyper, B is bitmap
            #define GB_A_IS_SPARSE_OR_HYPER 1
            #define GB_A_IS_BITMAP          0
            #define GB_A_IS_FULL            0
            #define GB_B_IS_SPARSE_OR_HYPER 0
            #define GB_B_IS_BITMAP          1
            #define GB_B_IS_FULL            0
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
        else
        { 
            // A is sparse/hyper, B is full
            #define GB_A_IS_SPARSE_OR_HYPER 1
            #define GB_A_IS_BITMAP          0
            #define GB_A_IS_FULL            0
            #define GB_B_IS_SPARSE_OR_HYPER 0
            #define GB_B_IS_BITMAP          0
            #define GB_B_IS_FULL            1
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
    }
    else if (A_is_bitmap)
    {
        if (B_is_sparse_or_hyper)
        { 
            // A is bitmap, B is sparse/hyper
            #define GB_A_IS_SPARSE_OR_HYPER 0
            #define GB_A_IS_BITMAP          1
            #define GB_A_IS_FULL            0
            #define GB_B_IS_SPARSE_OR_HYPER 1
            #define GB_B_IS_BITMAP          0
            #define GB_B_IS_FULL            0
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
        else if (B_is_bitmap)
        { 
            // both A and B are bitmap
            #define GB_A_IS_SPARSE_OR_HYPER 0
            #define GB_A_IS_BITMAP          1
            #define GB_A_IS_FULL            0
            #define GB_B_IS_SPARSE_OR_HYPER 0
            #define GB_B_IS_BITMAP          1
            #define GB_B_IS_FULL            0
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
        else
        { 
            // A is bitmap, B is full
            #define GB_A_IS_SPARSE_OR_HYPER 0
            #define GB_A_IS_BITMAP          1
            #define GB_A_IS_FULL            0
            #define GB_B_IS_SPARSE_OR_HYPER 0
            #define GB_B_IS_BITMAP          0
            #define GB_B_IS_FULL            1
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
    }
    else
    {
        if (B_is_sparse_or_hyper)
        { 
            // A is full, B is sparse/hyper
            #define GB_A_IS_SPARSE_OR_HYPER 0
            #define GB_A_IS_BITMAP          0
            #define GB_A_IS_FULL            1
            #define GB_B_IS_SPARSE_OR_HYPER 1
            #define GB_B_IS_BITMAP          0
            #define GB_B_IS_FULL            0
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
        else if (B_is_bitmap)
        { 
            // A is full, B is bitmap
            #define GB_A_IS_SPARSE_OR_HYPER 0
            #define GB_A_IS_BITMAP          0
            #define GB_A_IS_FULL            1
            #define GB_B_IS_SPARSE_OR_HYPER 0
            #define GB_B_IS_BITMAP          1
            #define GB_B_IS_FULL            0
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
        else
        { 
            // both A and B are full
            #define GB_A_IS_SPARSE_OR_HYPER 0
            #define GB_A_IS_BITMAP          0
            #define GB_A_IS_FULL            1
            #define GB_B_IS_SPARSE_OR_HYPER 0
            #define GB_B_IS_BITMAP          0
            #define GB_B_IS_FULL            1
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
    }
}

