//------------------------------------------------------------------------------
// GB_transpose_template: C=op(cast(A')), transpose, typecast, and apply op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    // Ax unused for some uses of this template
    #include "GB_unused.h"

    //--------------------------------------------------------------------------
    // get A and C
    //--------------------------------------------------------------------------

    #undef GBH_S
    #undef GB_S_TYPE

    #ifdef GB_BIND_1ST

        // A is the name of the matrix passed in to this kernel, but it takes
        // the place of the B matrix for C=op(x,A').  As a result, the B macros
        // must be used to access its contents.
        #define GBH_S(Ah,k) GBH_B(Ah,k)
        #define GB_S_TYPE GB_B_TYPE

        #ifdef GB_JIT_KERNEL
        #define GB_S_IS_SPARSE GB_B_IS_SPARSE
        #define GB_S_IS_HYPER  GB_B_IS_HYPER
        #define GB_S_IS_BITMAP GB_B_IS_BITMAP
        #define GB_S_IS_FULL   GB_B_IS_FULL
        #endif

    #else

        // for bind2nd, unary ops, and mere typecasting, use the A macros to
        // access the A matrix.
        #define GBH_S(Ah,k) GBH_A(Ah,k)
        #define GB_S_TYPE GB_A_TYPE

        #ifdef GB_JIT_KERNEL
        #define GB_S_IS_SPARSE GB_A_IS_SPARSE
        #define GB_S_IS_HYPER  GB_A_IS_HYPER
        #define GB_S_IS_BITMAP GB_A_IS_BITMAP
        #define GB_S_IS_FULL   GB_A_IS_FULL
        #endif

    #endif

    #ifndef GB_ISO_TRANSPOSE
    const GB_S_TYPE *restrict Ax = (GB_S_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    //--------------------------------------------------------------------------
    // C = op (cast (A'))
    //--------------------------------------------------------------------------

        if (Workspaces == NULL)
        {
            // A and C are both full or both bitmap
            if (A->b == NULL)
            {
                // A and C are both full
                #include "GB_transpose_full.c"
            }
            else
            {
                // A and C are both bitmap
                #include "GB_transpose_bitmap.c"
            }
        }
        else
        {
            // A is sparse or hypersparse; C is sparse
            #include "GB_transpose_sparse.c"
        }

}

#undef GB_ISO_TRANSPOSE

