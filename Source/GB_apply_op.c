//------------------------------------------------------------------------------
// GB_apply_op: typecast and apply a unary operator to an array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Cx = op ((xtype) Ax)

// Compare with GB_transpose_op.c

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_unaryop__include.h"
#endif

void GB_apply_op            // apply a unary operator, Cx = op ((xtype) Ax)
(
    GB_void *Cx,            // output array, of type op->ztype
    const GrB_UnaryOp op,   // operator to apply
    const GB_void *Ax,      // input array, of type Atype
    const GrB_Type Atype,   // type of Ax
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
    ASSERT (Atype != NULL) ;
    ASSERT (op != NULL) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;
    // TODO reduce nthreads for small problem (work: about O(anz))

    //--------------------------------------------------------------------------
    // define the worker for the switch factory
    //--------------------------------------------------------------------------

    // TODO: rename:
    // GrB_AINV_BOOL and GxB_ABS_BOOL to GrB_IDENTITY_BOOL.
    // GrB_MINV_BOOL to GxB_ONE_BOOL.
    // rename GxB_ABS_UINT* to GrB_IDENTITY_UINT*.
    // and do not create these workers

    #define GB_unop(op,zname,aname) GB_unop_ ## op ## zname ## aname

    #define GB_WORKER(op,zname,ztype,aname,atype)                           \
    {                                                                       \
        GB_unop (op,zname,aname) ((ztype *) Cx, (const atype *) Ax,         \
            anz, nthreads) ;                                                \
        return ;                                                            \
    }                                                                       \
    break ;

    //--------------------------------------------------------------------------
    // launch the switch factory
    //--------------------------------------------------------------------------

    #ifndef GBCOMPACT
    #include "GB_unaryop_factory.c"
    #endif

    //--------------------------------------------------------------------------
    // generic worker: typecast and apply an operator
    //--------------------------------------------------------------------------

    size_t asize = Atype->size ;
    size_t zsize = op->ztype->size ;
    size_t xsize = op->xtype->size ;
    GB_cast_function
        cast_A_to_X = GB_cast_factory (op->xtype->code, Atype->code) ;
    GxB_unary_function fop = op->function ;

    // TODO: some user operations are not thread safe.  This fails on
    // Demo/mis code when nthreads > 1.  See Demo/Source/mis_score.c.
    #pragma omp parallel for num_threads(nthreads)
    for (int64_t p = 0 ; p < anz ; p++)
    { 
        // xwork = (xtype) Ax [p]
        GB_void xwork [xsize] ;
        cast_A_to_X (xwork, Ax +(p*asize), asize) ;
        // Cx [p] = fop (xwork)
        fop (Cx +(p*zsize), xwork) ;
    }
}

