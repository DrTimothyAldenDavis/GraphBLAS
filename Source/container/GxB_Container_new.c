//------------------------------------------------------------------------------
// GxB_Container_new: create a new Container
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_container.h"
#define GB_FREE_ALL ;

//------------------------------------------------------------------------------
// GxB_Container_new
//------------------------------------------------------------------------------

GrB_Info GxB_Container_new
(
    GxB_Container *Container
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (Container) ;

    //--------------------------------------------------------------------------
    // allocate the new Container
    //--------------------------------------------------------------------------

    int header_arena = GrB_DEFAULT ;
    int data_arena = GrB_DEFAULT ;
    return (GxB_Container_new_arena (Container, header_arena, data_arena)) ;
}

