//------------------------------------------------------------------------------
// GxB_Context_new: create a new Context
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Default values are set to the current GxB_CONTEXT_WORLD settings.

#include "GB.h"

GrB_Info GxB_Context_new            // create a new Context
(
    GxB_Context *Context_handle     // handle of Context to create
)
{ 
    int header_arena = GrB_DEFAULT ;
    return (GxB_Context_new_arena (Context_handle, header_arena)) ;
}

