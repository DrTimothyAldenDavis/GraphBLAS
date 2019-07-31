//------------------------------------------------------------------------------
// gb_string_to_selectop: get a GraphBLAS select operator from a string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gb_matlab.h"

//  select operators, with their equivalent aliases

//  tril
//  triu
//  diag
//  offdiag
//  nonzero     ne0         !=0   ~=0
//  eqzero      eq0         ==0
//  gtzero      gt0         >0
//  gezero      ge0         >=0
//  ltzero      lt0         <0
//  lezero      le0         <=0
//  nethunk     !=thunk     ~=thunk
//  eqthunk     ==thunk
//  gtthunk     >thunk
//  gethunk     >=thunk
//  ltthunk     <thunk
//  lethunk     <=thunk

GrB_BinaryOp gb_string_to_selectop      // return select operator from a string
(
    char *opstring                      // string defining the operator
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (opstring == NULL || opstring [0] == '\0', "invalid selectop") ;

    //--------------------------------------------------------------------------
    // convert the string to a GraphBLAS select operator
    //--------------------------------------------------------------------------

    if (MATCH (opstring, "tril"))
    {
        return (GxB_TRIL) ;
    }
    else if (MATCH (opstring, "triu"))
    {
        return (GxB_TRIU) ;
    }
    else if (MATCH (opstring, "diag"))
    {
        return (GxB_DIAG) ;
    }
    else if (MATCH (opstring, "offdiag"))
    {
        return (GxB_OFFDIAG) ;
    }
    else if (MATCH (opstring, "nonzero") || MATCH (opstring, "ne0") ||
             MATCH (opstring, "!=0") || MATCH (opstring, "~=0"))
    {
        return (GxB_NONZERO) ;
    }
    else if (MATCH (opstring, "eqzero") || MATCH (opstring, "eq0") ||
             MATCH (opstring, "==0"))
    {
        return (GxB_EQ_ZERO) ;
    }
    else if (MATCH (opstring, "gtzero") || MATCH (opstring, "gt0") ||
             MATCH (opstring, ">0"))
    {
        return (GxB_GT_ZERO) ;
    }
    else if (MATCH (opstring, "gezero") || MATCH (opstring, "ge0") ||
             MATCH (opstring, ">=0"))
    {
        return (GxB_GE_ZERO) ;
    }
    else if (MATCH (opstring, "ltzero") || MATCH (opstring, "lt0") ||
             MATCH (opstring, "<0"))
    {
        return (GxB_LT_ZERO) ;
    }
    else if (MATCH (opstring, "lezero") || MATCH (opstring, "le0") ||
             MATCH (opstring, "<=0"))
    {
        return (GxB_LE_ZERO) ;
    }
    else if (MATCH (opstring, "nonzero") || MATCH (opstring, "nethunk") ||
             MATCH (opstring, "!=thunk") || MATCH (opstring, "~=thunk"))
    {
        return (GxB_NE_THUNK) ;
    }
    else if (MATCH (opstring, "eqthunk") || MATCH (opstring, "==thunk"))
    {
        return (GxB_EQ_THUNK) ;
    }
    else if (MATCH (opstring, "gtthunk") || MATCH (opstring, ">thunk"))
    {
        return (GxB_GT_THUNK) ;
    }
    else if (MATCH (opstring, "gethunk") || MATCH (opstring, ">=thunk"))
    {
        return (GxB_GE_THUNK) ;
    }
    else if (MATCH (opstring, "ltthunk") || MATCH (opstring, "<thunk"))
    {
        return (GxB_LT_THUNK) ;
    }
    else if (MATCH (opstring, "lethunk") || MATCH (opstring, "<=thunk"))
    {
        return (GxB_LE_THUNK) ;
    }

    ERROR2 ("selectop unknown: %s\n", opstring) ;
    return (NULL) ;
}

