//------------------------------------------------------------------------------
// GB_AxB_dot_meta2:  C=A'B, C<!M>=A'*B, or C<M>=A'*B via dot products
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// TODO: rename GB_AxB_dot_meta16.c for the 16 cases

// TODO: add dot3_phase1_template

{
    if (A_is_sparse)
    {
        if (B_is_sparse)
        { 
            // both A and B are sparse
            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
        #ifdef GB_DOT3
        else if (B_is_hyper)
        { 
            // A is sparse and B is hyper (dot3 only)
            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_AxB_dot3_template.c"
        }
        #endif
        else if (B_is_bitmap)
        { 
            // A is sparse and B is bitmap
            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
        else
        { 
            // A is sparse and B is full
            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
    }
    #ifdef GB_DOT3
    else if (A_is_hyper)
    {
        if (B_is_sparse)
        { 
            // A is hyper and B is sparse (dot3 only)
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_AxB_dot3_template.c"
        }
        else if (B_is_hyper)
        { 
            // both A and B are hyper (dot3 only)
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_AxB_dot3_template.c"
        }
        else if (B_is_bitmap)
        { 
            // A is hyper and B is bitmap (dot3 only)
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0
            #include "GB_AxB_dot3_template.c"
        }
        else
        { 
            // A is hyper and B is full (dot3 only)
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1
            #include "GB_AxB_dot3_template.c"
        }
    }
    #endif
    else if (A_is_bitmap)
    {
        if (B_is_sparse)
        { 
            // A is bitmap and B is sparse
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
        #ifdef GB_DOT3
        else if (B_is_hyper)
        { 
            // A is bitmap and B is hyper (dot3 only)
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_AxB_dot3_template.c"
        }
        #endif
        else if (B_is_bitmap)
        { 
            // both A and B are bitmap
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
        else
        { 
            // A is bitmap and B is full
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
    }
    else
    {
        if (B_is_sparse)
        { 
            // A is full and B is sparse
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
        #ifdef GB_DOT3
        else if (B_is_hyper)
        { 
            // A is full and B is hyper (dot3 only)
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0
            #include "GB_AxB_dot3_template.c"
        }
        #endif
        else if (B_is_bitmap)
        { 
            // A is full and B is bitmap
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
        else
        { 
            // both A and B are full
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1
            #ifdef GB_DOT3
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
    }
}

