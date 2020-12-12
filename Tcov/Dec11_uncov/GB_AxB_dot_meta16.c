//------------------------------------------------------------------------------
// GB_AxB_dot_meta16:  C=A'B, C<!M>=A'*B, or C<M>=A'*B via dot products
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// All 16 cases: A and B are sparse, hyper, bitmap, or full

{
    if (A_is_sparse)
    {
        if (B_is_sparse)
        {   GB_cov[388]++ ;
// covered (388): 367759
            // both A and B are sparse
            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #elif defined ( GB_DOT3_PHASE2 )
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        #ifdef GB_DOT3
        else if (B_is_hyper)
        {   GB_cov[389]++ ;
// covered (389): 246
            // A is sparse and B is hyper (dot3 only)
            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #else
            #include "GB_AxB_dot3_template.c"
            #endif

        }
        #endif
        else if (B_is_bitmap)
        {   GB_cov[390]++ ;
// covered (390): 15542
            // A is sparse and B is bitmap
            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #elif defined ( GB_DOT3_PHASE2 )
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else
        {   GB_cov[391]++ ;
// covered (391): 27544
            // A is sparse and B is full
            #define GB_A_IS_SPARSE 1
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #elif defined ( GB_DOT3_PHASE2 )
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
        {   GB_cov[392]++ ;
// NOT COVERED (392):
            // A is hyper and B is sparse (dot3 only)
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #else
            #include "GB_AxB_dot3_template.c"
            #endif

        }
        else if (B_is_hyper)
        {   GB_cov[393]++ ;
// NOT COVERED (393):
            // both A and B are hyper (dot3 only)
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #else
            #include "GB_AxB_dot3_template.c"
            #endif

        }
        else if (B_is_bitmap)
        {   GB_cov[394]++ ;
// NOT COVERED (394):
            // A is hyper and B is bitmap (dot3 only)
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #else
            #include "GB_AxB_dot3_template.c"
            #endif

        }
        else
        {   GB_cov[395]++ ;
// covered (395): 23850
            // A is hyper and B is full (dot3 only)
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  1
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #else
            #include "GB_AxB_dot3_template.c"
            #endif

        }
    }
    #endif
    else if (A_is_bitmap)
    {
        if (B_is_sparse)
        {   GB_cov[396]++ ;
// covered (396): 330
            // A is bitmap and B is sparse
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #elif defined ( GB_DOT3_PHASE2 )
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        #ifdef GB_DOT3
        else if (B_is_hyper)
        {   GB_cov[397]++ ;
// covered (397): 44
            // A is bitmap and B is hyper (dot3 only)
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #else
            #include "GB_AxB_dot3_template.c"
            #endif

        }
        #endif
        else if (B_is_bitmap)
        {   GB_cov[398]++ ;
// covered (398): 76
            // both A and B are bitmap
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #elif defined ( GB_DOT3_PHASE2 )
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else
        {   GB_cov[399]++ ;
// covered (399): 12750
            // A is bitmap and B is full
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 1
            #define GB_A_IS_FULL   0
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #elif defined ( GB_DOT3_PHASE2 )
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
    }
    else
    {
        if (B_is_sparse)
        {   GB_cov[400]++ ;
// covered (400): 1315
            // A is full and B is sparse
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 1
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #elif defined ( GB_DOT3_PHASE2 )
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        #ifdef GB_DOT3
        else if (B_is_hyper)
        {   GB_cov[401]++ ;
// NOT COVERED (401):
            // A is full and B is hyper (dot3 only)
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  1
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   0

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #else
            #include "GB_AxB_dot3_template.c"
            #endif

        }
        #endif
        else if (B_is_bitmap)
        {   GB_cov[402]++ ;
// covered (402): 34
            // A is full and B is bitmap
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 1
            #define GB_B_IS_FULL   0

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #elif defined ( GB_DOT3_PHASE2 )
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
        else
        {   GB_cov[403]++ ;
// covered (403): 6798
            // both A and B are full
            #define GB_A_IS_SPARSE 0
            #define GB_A_IS_HYPER  0
            #define GB_A_IS_BITMAP 0
            #define GB_A_IS_FULL   1
            #define GB_B_IS_SPARSE 0
            #define GB_B_IS_HYPER  0
            #define GB_B_IS_BITMAP 0
            #define GB_B_IS_FULL   1

            #if defined ( GB_DOT3_PHASE1 )
            #include "GB_AxB_dot3_phase1_template.c"
            #elif defined ( GB_DOT3_PHASE2 )
            #include "GB_AxB_dot3_template.c"
            #else
            #include "GB_AxB_dot2_template.c"
            #endif

        }
    }
}

