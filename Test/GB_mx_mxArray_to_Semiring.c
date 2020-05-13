//------------------------------------------------------------------------------
// GB_mx_mxArray_to_Semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Get a semiring struct from MATLAB and convert it into a GraphBLAS semiring.
//
// The semiring MATLAB struct must contain the following strings:
//
//      multiply    a string with the name of the 'multiply' binary operator.
//      add         a string with the name of the 'add' binary operator.
//                  The operator must be commutative.
//      type        the type of x and y for the multiply operator.
//                  ('logical', 'int8', ... 'double complex').  optional.

#include "GB_mex.h"

bool GB_mx_mxArray_to_Semiring         // true if successful
(
    GrB_Semiring *handle,               // the semiring
    const mxArray *semiring_matlab,     // MATLAB version of semiring
    const char *name,                   // name of the argument
    const GrB_Type default_optype       // default operator type
)
{

    GB_WHERE ("GB_mx_mxArray_to_Semiring") ;

    if (default_optype == Complex && Complex != GxB_FC64)
    {
        (*handle) = Complex_plus_times ;
        return (true) ;
    }

    (*handle) = NULL ;
    const mxArray *multiply_mx = NULL, *type_mx = NULL, *add_mx = NULL ;

    if (semiring_matlab == NULL || mxIsEmpty (semiring_matlab))
    {
        // semiring is not present; defaults will be used
        ;
    }
    else if (mxIsStruct (semiring_matlab))
    {
        // look for semiring.multiply
        int fieldnumber = mxGetFieldNumber (semiring_matlab, "multiply") ;
        if (fieldnumber >= 0)
        {
            multiply_mx = mxGetFieldByNumber (semiring_matlab, 0, fieldnumber) ;
        }
        // look for semiring.class
        fieldnumber = mxGetFieldNumber (semiring_matlab, "class") ;
        if (fieldnumber >= 0)
        {
            type_mx = mxGetFieldByNumber (semiring_matlab, 0, fieldnumber) ;
        }
        // look for semiring.add
        fieldnumber = mxGetFieldNumber (semiring_matlab, "add") ;
        if (fieldnumber >= 0)
        {
            add_mx = mxGetFieldByNumber (semiring_matlab, 0, fieldnumber) ;
        }
    }
    else
    {
        mexWarnMsgIdAndTxt ("GB:warn","invalid semiring") ;
        return (false) ;
    }

    // find the corresponding built-in GraphBLAS multiply operator
    GrB_BinaryOp multiply = NULL ;
    if (!GB_mx_string_to_BinaryOp (&multiply, default_optype,
        multiply_mx, type_mx, false) || multiply == NULL)
    {
        mexWarnMsgIdAndTxt ("GB:warn","mult missing") ;
        return (false) ;
    }

    #if 0
    bool zbool ;
    switch (multiply->opcode)
    {
        // z=f(x,y), all x,y,z the same type
        case GB_FIRST_opcode   : zbool = false ; break ;
        case GB_SECOND_opcode  : zbool = false ; break ;
        case GB_PAIR_opcode    : zbool = false ; break ;
        case GB_ANY_opcode     : zbool = false ; break ;
        case GB_MIN_opcode     : zbool = false ; break ;
        case GB_MAX_opcode     : zbool = false ; break ;
        case GB_PLUS_opcode    : zbool = false ; break ;
        case GB_MINUS_opcode   : zbool = false ; break ;
        case GB_RMINUS_opcode  : zbool = false ; break ;
        case GB_TIMES_opcode   : zbool = false ; break ;
        case GB_DIV_opcode     : zbool = false ; break ;
        case GB_RDIV_opcode    : zbool = false ; break ;

        // 6 z=f(x,y), all x,y,z the same type
        case GB_ISEQ_opcode    : zbool = false  ; break ;
        case GB_ISNE_opcode    : zbool = false  ; break ;
        case GB_ISGT_opcode    : zbool = false  ; break ;
        case GB_ISLT_opcode    : zbool = false  ; break ;
        case GB_ISGE_opcode    : zbool = false  ; break ;
        case GB_ISLE_opcode    : zbool = false  ; break ;

        // 6 z=f(x,y), z boolean; x and y are semiring.class
        case GB_EQ_opcode      : zbool = true  ; break ;
        case GB_NE_opcode      : zbool = true  ; break ;
        case GB_GT_opcode      : zbool = true  ; break ;
        case GB_LT_opcode      : zbool = true  ; break ;
        case GB_GE_opcode      : zbool = true  ; break ;
        case GB_LE_opcode      : zbool = true  ; break ;

        // 3 z=f(x,y), all x,y,z the same type
        case GB_LOR_opcode     : zbool = false ; break ;
        case GB_LAND_opcode    : zbool = false ; break ;
        case GB_LXOR_opcode    : zbool = false ; break ;

        // bitwise
        case GB_BOR_opcode     : zbool = false ; break ;
        case GB_BAND_opcode    : zbool = false ; break ;
        case GB_BXOR_opcode    : zbool = false ; break ;
        case GB_BXNOR_opcode   : zbool = false ; break ;

        default :
            mexWarnMsgIdAndTxt ("GB:warn","unsupported multiply operator") ;
            return (false) ;
    }
    #endif

    ASSERT_BINARYOP_OK (multiply, "semiring multiply", GB0) ;

    // find the corresponding built-in GraphBLAS add operator
    GrB_BinaryOp add = NULL ;
    if (!GB_mx_string_to_BinaryOp (&add, multiply->ztype,
        add_mx, NULL, false) || add == NULL)
    {
        mexWarnMsgIdAndTxt ("GB:warn", "add failed") ;
        return (false) ;
    }

    ASSERT_BINARYOP_OK (add, "semiring add", GB0) ;
    ASSERT_BINARYOP_OK (multiply, "semiring multiply", GB0) ;

    // create the monoid with the add operator and its identity value
    GrB_Monoid monoid = GB_mx_builtin_monoid (add) ;
    if (monoid == NULL)
    {
        mexWarnMsgIdAndTxt ("GB:warn", "monoid failed") ;
        return (false) ;
    }

    // create the semiring
    GrB_Semiring semiring = GB_mx_builtin_semiring (monoid, multiply) ;
    if (semiring == NULL)
    {
        mexWarnMsgIdAndTxt ("GB:warn", "semiring failed") ;
        return (false) ;
    }

    ASSERT_SEMIRING_OK (semiring, "semiring", GB0) ;

    (*handle) = semiring ;
    return (true) ;
}

