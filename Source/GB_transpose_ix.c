//------------------------------------------------------------------------------
// GB_transpose_ix: transpose the values and pattern of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The values of A are typecasted to C_type, the type of the C matrix.
// A can be sparse or hypersparse, but C is not hypersparse.

// The row pointers of the output matrix have already been computed, in Cp.
// Row i will appear in Ci, in the positions Cp [i] .. Cp [i+1], for the
// version of Cp on *input*.  On output, however, Cp has been shifted down
// by one.  Cp [0:m-1] has been over written with Cp [1:m].  They can be
// shifted back, if needed, but GraphBLAS treats this array Cp, on input
// to this function, as a throw-away copy of Cp.

// Compare with GB_transpose_op.c

// The bucket sort is not parallel.

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_unaryop__include.h"
#endif

void GB_transpose_ix        // transpose the pattern and values of a matrix
(
    int64_t *Cp,            // size m+1, input: row pointers, shifted on output
    int64_t *Ci,            // size cnz, output column indices
    GB_void *Cx,            // size cnz, output numerical values, type C_type
    const GrB_Type C_type,  // type of output C (do typecasting into C)
    const GrB_Matrix A,     // input matrix
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A != NULL) ;
    ASSERT (C_type != NULL) ;
    ASSERT (Cp != NULL && Ci != NULL && Cx != NULL) ;
    ASSERT (GB_Type_compatible (A->type, C_type)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    //--------------------------------------------------------------------------
    // define the worker for the switch factory
    //--------------------------------------------------------------------------

    #define GB_tran(zname,aname) GB_tran__identity ## zname ## aname

    #define GB_WORKER(ignore1,zname,ztype,aname,atype)      \
    {                                                       \
        GB_tran (zname,aname) (Cp, Ci, (ztype *) Cx, A) ;   \
        return ;                                            \
    }                                                       \
    break ;

    //--------------------------------------------------------------------------
    // launch the switch factory
    //--------------------------------------------------------------------------

    // switch factory for two types, controlled by code1 and code2
    GB_Type_code code1 = C_type->code ;         // defines ztype
    GB_Type_code code2 = A->type->code ;        // defines atype

    #ifndef GBCOMPACT
    #include "GB_2type_factory.c"
    #endif

    //--------------------------------------------------------------------------
    // generic worker: transpose and typecast
    //--------------------------------------------------------------------------

    const int64_t *Ai = A->i ;
    const GB_void *Ax = A->x ;
    size_t asize = A->type->size ;
    size_t csize = C_type->size ;
    GB_cast_function cast_A_to_X = GB_cast_factory (code1, code2) ;

    GBI_for_each_vector (A)
    {
        GBI_for_each_entry (j, p, pend)
        { 
            int64_t q = Cp [Ai [p]]++ ;
            Ci [q] = j ;
            // Cx [q] = Ax [p]
            cast_A_to_X (Cx +(q*csize), Ax +(p*asize), asize) ;
        }
    }
}

