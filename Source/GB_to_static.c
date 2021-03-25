//------------------------------------------------------------------------------
// GB_to_static: ensure a matrix has a static header, not dynamic
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input matrix &A has a dynamic header, and is copied into a static
// header.  The input matrix &A header is freed, after its contents are
// transplanted into the static header.

// GB_transplant can also be used, but that method is more general.  This
// method is simpler for the case where A_static is just an empty header on
// input, and the header of &A can be directly copied into it.

#include "GB_dynamic.h"

void GB_to_static
(
    // output
    GrB_Matrix A_static,        // output matrix with static header
    // input
    GrB_Matrix *Ahandle,        // input matrix &A with dynamic header
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A_static != NULL && A_static->static_header) ;
    ASSERT (Ahandle != NULL && (*Ahandle) != NULL) ;
    ASSERT (!((*Ahandle)->static_header)) ;

    //--------------------------------------------------------------------------
    // copy the dynamic header of &A into the static header, A_static
    //--------------------------------------------------------------------------

    size_t header_size = (*Ahandle)->header_size ;
    memcpy (A_static, *Ahandle, sizeof (struct GB_Matrix_opaque)) ;
    A_static->header_size = 0 ;
    A_static->static_header = true ;
    GB_FREE (Ahandle, header_size) ;
}

