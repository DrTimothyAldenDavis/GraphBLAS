//------------------------------------------------------------------------------
// GxB_Scalar_nvals: number of entries in a sparse GxB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Scalar_nvals   // get the number of entries in a GxB_Scalar
(
    GrB_Index *nvals,       // number of entries (1 or 0)
    const GxB_Scalar s      // GxB_Scalar to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Scalar_nvals (&nvals, s)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (s) ;
    ASSERT (GB_SCALAR_OK (s)) ;

    //--------------------------------------------------------------------------
    // get the number of entries
    //--------------------------------------------------------------------------

    return (GB_nvals (nvals, (GrB_Matrix) s, Context)) ;
}

