//------------------------------------------------------------------------------
// GB_transpose_op: transpose, typecase, and apply an operator to a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// C = op ((xtype) A')

// The values of A are typecasted to op->xtype and then passed to the unary
// operator.  The output is assigned to R, which must be of type op->ztype; no
// output typecasting done with the output of the operator.

// The row pointers of the output matrix have already been computed, in Cp.
// Row i will appear in Ci, in the positions Cp [i] .. Cp [i+1], for the
// version of Cp on *input*.  On output, however, Cp has been shifted down
// by one.  Cp [0:m-1] has been over written with Cp [1:m].  They can be
// shifted back, if needed, but GraphBLAS treats this array Cp, on input
// to this function, as a throw-away copy of Cp.

// Compare with GB_transpose_ix.c and GB_apply_op.c

// Note that the bucket transpose is sequential.

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_unaryop__include.h"
#endif

void GB_transpose_op        // transpose and apply an operator to a matrix
(
    int64_t *Cp,            // size m+1, input: row pointers, shifted on output
    int64_t *Ci,            // size cnz, output column indices
    GB_void *Cx,            // size cnz, output values, type op->ztype
    const GrB_UnaryOp op,   // operator to apply, NULL if no operator
    const GrB_Matrix A,     // input matrix
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A != NULL) ;
    ASSERT (op != NULL) ;
    ASSERT (Cp != NULL && Ci != NULL && Cx != NULL) ;
    ASSERT (op != NULL) ;
    ASSERT (GB_Type_compatible (A->type, op->xtype)) ;
    ASSERT (GB_IMPLIES (op->opcode < GB_USER_C_opcode, op->xtype == op->ztype));
    ASSERT (!GB_ZOMBIES (A)) ;

    //--------------------------------------------------------------------------
    // define the worker for the switch factory
    //--------------------------------------------------------------------------

    GrB_Type Atype = A->type ;

    #define GB_tran(op,zname,aname) GB_tran_ ## op ## zname ## aname

    #define GB_WORKER(op,zname,ztype,aname,atype)                           \
    {                                                                       \
        GB_tran (op,zname,aname) (Cp, Ci, (ztype *) Cx, A) ;                \
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
    // generic worker: transpose, typecast, and apply an operator
    //--------------------------------------------------------------------------

    const int64_t *Ai = A->i ;
    const GB_void *Ax = A->x ;

    size_t asize = Atype->size ;
    size_t zsize = op->ztype->size ;
    size_t xsize = op->xtype->size ;
    GB_cast_function
        cast_A_to_X = GB_cast_factory (op->xtype->code, Atype->code) ;
    GxB_unary_function fop = op->function ;

    GBI_for_each_vector (A)
    {
        GBI_for_each_entry (j, p, pend)
        { 
            int64_t q = Cp [Ai [p]]++ ;
            Ci [q] = j ;
            // xwork = (xtype) Ax [p]
            GB_void xwork [xsize] ;
            cast_A_to_X (xwork, Ax +(p*asize), asize) ;
            // Cx [q] = fop (xwork) ; Cx is of type op->ztype
            fop (Cx +(q*zsize), xwork) ;
        }
    }
}

