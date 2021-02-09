//------------------------------------------------------------------------------
// GB_ewise_generic: generic methods for eWiseMult and eWiseAdd
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_ewise.h"
#include "GB_emult.h"
#include "GB_binop.h"
#include "GB_unused.h"
#include "GB_ek_slice.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL             \
{                               \
    GB_Matrix_free (Chandle) ;  \
}

void GB_ewise_generic       // generic ewise
(
    // input/output:
    GrB_Matrix *Chandle,    // output matrix
    // input:
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    // tasks from phase1a:
    const GB_task_struct *GB_RESTRICT TaskList,  // array of structs
    const int C_ntasks,                         // # of tasks
    const int C_nthreads,                       // # of threads to use
    // analysis from phase0:
    const int64_t *GB_RESTRICT C_to_M,
    const int64_t *GB_RESTRICT C_to_A,
    const int64_t *GB_RESTRICT C_to_B,
    const int C_sparsity,
    // from GB_emult_sparsity:
    const int emult_method,
    // from GB_emult_100:
    const int64_t *GB_RESTRICT Cp_kfirst,
    // to slice M, A, and/or B,
    const int64_t *M_ek_slicing, const int M_ntasks, const int M_nthreads,
    const int64_t *A_ek_slicing, const int A_ntasks, const int A_nthreads,
    const int64_t *B_ek_slicing, const int B_ntasks, const int B_nthreads,
    // original input:
    const GrB_Matrix M,             // optional mask, may be NULL
    const bool Mask_struct,         // if true, use the only structure of M
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Context Context
)
{

    ASSERT_MATRIX_OK_OR_NULL (M, "M for ewise generic", GB0) ;
    ASSERT_MATRIX_OK (A, "A for ewise generic", GB0) ;
    ASSERT_MATRIX_OK (B, "B for ewise generic", GB0) ;
//  printf ("noew here %d %d\n", M_ntasks, M_nthreads) ;

    GrB_Matrix C = (*Chandle) ;
    GrB_Type ctype = C->type ;
    GB_Type_code ccode = ctype->code ;
    GB_Opcode opcode = op->opcode ;
    bool op_is_positional = GB_OPCODE_IS_POSITIONAL (opcode) ;
    bool op_is_first  = (opcode == GB_FIRST_opcode) ;
    bool op_is_second = (opcode == GB_SECOND_opcode) ;
    bool op_is_pair   = (opcode == GB_PAIR_opcode) ;

        GxB_binary_function fmult ;
        size_t csize, asize, bsize, xsize, ysize, zsize ;
        GB_cast_function cast_A_to_X, cast_B_to_Y, cast_Z_to_C ;

        fmult = op->function ;      // NULL if op is positional
        csize = ctype->size ;
        asize = A->type->size ;
        bsize = B->type->size ;

        if (op_is_second || op_is_pair || op_is_positional)
        { 
            // the op does not depend on the value of A(i,j)
//          printf ("\nA is pattern\n") ;
            xsize = 1 ;
            cast_A_to_X = NULL ;
        }
        else
        { 
            xsize = op->xtype->size ;
            cast_A_to_X = GB_cast_factory (op->xtype->code, A->type->code) ;
        }

        if (op_is_first || op_is_pair || op_is_positional)
        { 
            // the op does not depend on the value of B(i,j)
//          printf ("\nB is pattern\n") ;
            ysize = 1 ;
            cast_B_to_Y = NULL ;
        }
        else
        { 
            ysize = op->ytype->size ;
            cast_B_to_Y = GB_cast_factory (op->ytype->code, B->type->code) ;
        }

        zsize = op->ztype->size ;
        cast_Z_to_C = GB_cast_factory (ccode, op->ztype->code) ;
//      printf ("cast_Z_to_C: %p\n", cast_Z_to_C) ;

        // aij = (xtype) A(i,j), located in Ax [pA]
        #define GB_GETA(aij,Ax,pA)                                          \
            GB_void aij [GB_VLA(xsize)] ;                                   \
            if (cast_A_to_X != NULL)                                        \
            {                                                               \
                cast_A_to_X (aij, Ax +((pA)*asize), asize) ;                \
            }

        // bij = (ytype) B(i,j), located in Bx [pB]
        #define GB_GETB(bij,Bx,pB)                                          \
            GB_void bij [GB_VLA(ysize)] ;                                   \
            if (cast_B_to_Y != NULL)                                        \
            {                                                               \
                cast_B_to_Y (bij, Bx +((pB)*bsize), bsize) ;                \
            }

        // address of Cx [p]
        #define GB_CX(p) Cx +((p)*csize)

        #define GB_ATYPE GB_void
        #define GB_BTYPE GB_void
        #define GB_CTYPE GB_void

        #define GB_PHASE_2_OF_2

        // loops cannot be vectorized
        #define GB_PRAGMA_SIMD_VECTORIZE ;

        if (op_is_positional)
        { 

            //------------------------------------------------------------------
            // C(i,j) = positional_op (aij, bij)
            //------------------------------------------------------------------

            const int64_t offset = GB_positional_offset (opcode) ;
            const bool index_is_i = 
                (opcode == GB_FIRSTI_opcode  ) ||
                (opcode == GB_FIRSTI1_opcode ) ||
                (opcode == GB_SECONDI_opcode ) ||
                (opcode == GB_SECONDI1_opcode) ;
            if (op->ztype == GrB_INT64)
            {
                #undef  GB_BINOP
                #define GB_BINOP(cij, aij, bij, i, j)                         \
                    int64_t z = ((index_is_i) ? i : j) + offset ;             \
                    cast_Z_to_C (cij, &z, csize) ;
                if (emult_method == GB_EMULT_METHOD_100)
                {
                    #include "GB_emult_100_template.c"
                }
                else
                {
                    #include "GB_emult_template.c"
                }
            }
            else
            {
                #undef  GB_BINOP
                #define GB_BINOP(cij, aij, bij, i, j)                         \
                    int32_t z = (int32_t) (((index_is_i) ? i : j) + offset) ; \
                    cast_Z_to_C (cij, &z, csize) ;
                if (emult_method == GB_EMULT_METHOD_100)
                {
                    #include "GB_emult_100_template.c"
                }
                else
                {
                    #include "GB_emult_template.c"
                }
            }

        }
        else
        { 

            //------------------------------------------------------------------
            // standard binary operator
            //------------------------------------------------------------------

            // C(i,j) = (ctype) (A(i,j) + B(i,j))
            // not used if op is null
            #undef  GB_BINOP
            #define GB_BINOP(cij, aij, bij, i, j)   \
                ASSERT (op != NULL) ;               \
                GB_void z [GB_VLA(zsize)] ;         \
                fmult (z, aij, bij) ;               \
                cast_Z_to_C (cij, z, csize) ;
            if (emult_method == GB_EMULT_METHOD_100)
            {
//              printf ("\nemult100\n") ;
//  printf ("ok here %d %d\n", M_ntasks, M_nthreads) ;
                #include "GB_emult_100_template.c"
            }
            else
            {
//  printf ("too here %d %d\n", M_ntasks, M_nthreads) ;
                #include "GB_emult_template.c"
            }
        }

    ASSERT_MATRIX_OK (C, "C from ewise generic", GB0) ;
}

