//------------------------------------------------------------------------------
// GrB_IndexUnaryOp_free: free an index_unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_IndexUnaryOp_free          // free a user-created index_unary op
(
    GrB_IndexUnaryOp *indexunaryop      // handle of operator to free
)
{

    if (indexunaryop != NULL)
    {
        // only free a dynamically-allocated operator
        GrB_IndexUnaryOp op = *indexunaryop ;
        if (op != NULL)
        {
            size_t header_size = op->header_size ;
            if (header_size > 0)
            { 
                op->magic = GB_FREED ;  // to help detect dangling pointers
                op->header_size = 0 ;
                GB_FREE (indexunaryop, header_size) ;
            }
        }
    }

    return (GrB_SUCCESS) ;
}

