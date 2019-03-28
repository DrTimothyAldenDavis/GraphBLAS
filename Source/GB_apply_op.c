//------------------------------------------------------------------------------
// GB_apply_op:  apply a unary operator to an array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Cx = op ((xtype) Ax)

// Compare with GB_transpose_op.c

// PARALLEL: do it here, but it is easy.  Might want to split into separate
// files like Generated/GB_AxB*, so worker is not in a macro but in a function.

#include "GB.h"

void GB_apply_op            // apply a unary operator, Cx = op ((xtype) Ax)
(
    GB_void *Cx,            // output array, of type op->ztype
    const GrB_UnaryOp op,   // operator to apply
    const GB_void *Ax,      // input array, of type atype
    const GrB_Type atype,   // type of Ax
    const int64_t anz,      // size of Ax and Cx
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Cx != NULL) ;
    ASSERT (Ax != NULL) ;
    ASSERT (anz >= 0) ;
    ASSERT (atype != NULL) ;
    ASSERT (op != NULL) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;

    //--------------------------------------------------------------------------
    // define the worker for the switch factory
    //--------------------------------------------------------------------------

    // Some unary operators z=f(x) do not use the value x, like z=1.  This is
    // intentional, so the gcc warning is ignored.
    #pragma GCC diagnostic ignored "-Wunused-but-set-variable"

    // For built-in types only, thus xtype == ztype, but atype can differ
    #define GB_WORKER(ztype,atype)                              \
    {                                                           \
        ztype *cx = (ztype *) Cx ;                              \
        atype *ax = (atype *) Ax ;                              \
        for (int64_t p = 0 ; p < anz ; p++)                     \
        {                                                       \
            /* z = (ztype) ax [p], type casting */              \
            ztype z ;                                           \
            GB_CASTING (z, ax [p]) ;                            \
            /* apply the unary operator */                      \
            cx [p] = GB_OP (z) ;                                \
        }                                                       \
        return ;                                                \
    }

    //--------------------------------------------------------------------------
    // launch the switch factory
    //--------------------------------------------------------------------------

    #ifndef GBCOMPACT
    #include "GB_unaryop_factory.c"
    #endif

    // If the switch factory has no worker for the opcode or type, then it
    // falls through to the generic worker below.

    //--------------------------------------------------------------------------
    // generic worker:  apply an operator, with optional typecasting
    //--------------------------------------------------------------------------

    // The generic worker can handle any operator and any type, and it does all
    // required typecasting.  Thus the switch factory can be disabled, and the
    // code will more compact and still work.  It will just be slower.

    int64_t asize = atype->size ;
    int64_t zsize = op->ztype->size ;
    GB_cast_function
        cast_A_to_X = GB_cast_factory (op->xtype->code, atype->code) ;
    GxB_unary_function fop = op->function ;

    // scalar workspace
    char xwork [op->xtype->size] ;

    for (int64_t p = 0 ; p < anz ; p++)
    { 
        // xwork = (xtype) Ax [p]
        cast_A_to_X (xwork, Ax +(p*asize), asize) ;
        // Cx [p] = fop (xwork)
        fop (Cx +(p*zsize), xwork) ;
    }
}

