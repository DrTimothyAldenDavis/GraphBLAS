//------------------------------------------------------------------------------
// GrB_getVersion: get the C API version this library implements
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// This function appears in the V1.3 draft of the GraphBLAS C API.

#include "GB.h"

GrB_Info GrB_getVersion
(
    unsigned int *version,
    unsigned int *subversion
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CONTEXT ("GrB_getVersion (&version, &subversion") ;
    GB_RETURN_IF_NULL (version) ;
    GB_RETURN_IF_NULL (subversion) ;

    //--------------------------------------------------------------------------
    // return the C API version
    //--------------------------------------------------------------------------

    (*version) = GrB_VERSION ;
    (*subversion) = GrB_SUBVERSION ;
    return (GrB_SUCCESS) ;
}

