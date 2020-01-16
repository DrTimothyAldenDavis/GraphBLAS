//------------------------------------------------------------------------------
// GB_dense_ewise3_noaccum: C = A+B where all 3 matries are dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_dense.h"
#ifndef GBCOMPACT
#include "GB_binop__include.h"
#endif

GrB_Info GB_dense_ewise3_noaccum    // C = A+B
(
    GrB_Matrix C,                   // input/output matrix
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_BinaryOp op,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C for dense C=A+B", GB0) ;
    ASSERT (!GB_PENDING (C)) ; ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_PENDING (A)) ; ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_PENDING (B)) ; ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (!GB_aliased (C, A)) ;
    ASSERT (!GB_aliased (C, B)) ;
    ASSERT (!GB_aliased (A, B)) ;
    ASSERT (GB_is_dense (C)) ;
    ASSERT (GB_is_dense (A)) ;
    ASSERT (GB_is_dense (B)) ;
    ASSERT_BINARYOP_OK (op, "op for dense C=A+B", GB0) ;
    ASSERT (op->ztype == C->type) ;
    ASSERT (op->xtype == A->type) ;
    ASSERT (op->ytype == B->type) ;
    ASSERT (op->opcode >= GB_MIN_CODE) ;
    ASSERT (op->opcode <= GB_RDIV_CODE) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t cnz = GB_NNZ (C) ;

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (cnz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // define the worker for the switch factory
    //--------------------------------------------------------------------------

    #define GB_Cdense_ewise3_noaccum(op,xyname) \
        GB_Cdense_ewise3_noaccum_ ## op ## xyname

    #define GB_BINOP_WORKER(op,xyname)                                      \
    {                                                                       \
        info = GB_Cdense_ewise3_noaccum(op,xyname) (C, A, B, nthreads) ;    \
    }                                                                       \
    break ;

    //--------------------------------------------------------------------------
    // launch the switch factory
    //--------------------------------------------------------------------------

    // TODO handle more operators?  Mix of 2 operators? Mix of types?

    #ifndef GBCOMPACT

        GB_Opcode opcode ;
        GB_Type_code xycode, zcode ;
        if (GB_binop_builtin (A->type, false, B->type, false, op, false,
            &opcode, &xycode, &zcode))
        { 
            #define GB_BINOP_SUBSET
            #include "GB_binop_factory.c"
        }

    #endif

#if 0

    TODO: extend to handle typecasting and generic operators

    if (!done)
    {
        GB_BURBLE_MATRIX (C, "generic ") ;

        //----------------------------------------------------------------------
        // get operators, functions, workspace, contents of x and C
        //----------------------------------------------------------------------

        GxB_binary_function fadd = op->function ;

        //----------------------------------------------------------------------
        // C += x via function pointers, and typecasting
        //----------------------------------------------------------------------

        // C(i,j) = C(i,j) + scalar
        #define GB_BINOP(cout_ij, cin_aij, ywork) \
            GB_BINARYOP (cout_ij, cin_aij, ywork)

        // binary operator
        #define GB_BINARYOP(z,x,y) fadd (z,x,y)

        // address of Cx [p]
        #define GB_CX(p) Cx +((p)*csize)

        #define GB_CTYPE GB_void

        // no vectorization
        #define GB_PRAGMA_VECTORIZE

        #include "GB_dense_ewise3_noaccum_template.c"
    }
#endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C=A+B output", GB0) ;
    return (GrB_SUCCESS) ;
}

