//------------------------------------------------------------------------------
// GzB_IndexBinaryOp_free: free an index_binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GzB_IndexBinaryOp_free     // free a user-created index binary operator
(
    GzB_IndexBinaryOp *op           // handle of index binary operator to free
)
{ 
// GB_GOTCHA ;
    return (GB_Op_free ((GB_Operator *) op)) ;
}

