//------------------------------------------------------------------------------
// GB_AxB_dot_meta16:  C=A'B, C<!M>=A'*B, or C<M>=A'*B via dot products
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// All 16 cases are handled: A and B are sparse, hyper, bitmap, or full.
// For dot2, A and B are never hypersparse.

{
    if (A_is_sparse)
    {

        if (B_is_sparse)
        { 

            //------------------------------------------------------------------
            // both A and B are sparse
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else if (B_is_hyper)
        { 

            //------------------------------------------------------------------
            // A is sparse and B is hyper
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else if (B_is_bitmap)
        { 

            //------------------------------------------------------------------
            // A is sparse and B is bitmap
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else
        { 

            //------------------------------------------------------------------
            // A is sparse and B is full
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
    }
    else if (A_is_hyper)
    {
        if (B_is_sparse)
        { 

            //------------------------------------------------------------------
            // A is hyper and B is sparse
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else if (B_is_hyper)
        { 

            //------------------------------------------------------------------
            // both A and B are hyper
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else if (B_is_bitmap)
        { 

            //------------------------------------------------------------------
            // A is hyper and B is bitmap
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else
        { 

            //------------------------------------------------------------------
            // A is hyper and B is full
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
    }
    else if (A_is_bitmap)
    {
        if (B_is_sparse)
        { 

            //------------------------------------------------------------------
            // A is bitmap and B is sparse
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else if (B_is_hyper)
        { 

            //------------------------------------------------------------------
            // A is bitmap and B is hyper
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else if (B_is_bitmap)
        { 

            //------------------------------------------------------------------
            // both A and B are bitmap
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else
        { 

            //------------------------------------------------------------------
            // A is bitmap and B is full
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
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

            //------------------------------------------------------------------
            // A is full and B is sparse
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else if (B_is_hyper)
        { 

            //------------------------------------------------------------------
            // A is full and B is hyper
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else if (B_is_bitmap)
        { 

            //------------------------------------------------------------------
            // A is full and B is bitmap
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else
        { 

            //------------------------------------------------------------------
            // both A and B are full
            //------------------------------------------------------------------

            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1

            #if GB_DOT4
            #include "GB_AxB_dot4_template.c"
            #elif GB_DOT3_PHASE1
            #include "GB_AxB_dot3_phase1_template.c"
            #elif GB_DOT3_PHASE2
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif
        }
    }
}

