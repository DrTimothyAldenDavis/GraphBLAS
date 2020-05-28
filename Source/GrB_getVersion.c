//------------------------------------------------------------------------------
// GrB_getVersion: get the version number of the GraphBLAS C API standard
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// For compile-time access, use GrB_VERSION and GrB_SUBVERSION.

#include "GB.h"

GrB_Info GrB_getVersion         // runtime access to C API version number
(
    unsigned int *version,      // returns GrB_VERSION
    unsigned int *subversion    // returns GrB_SUBVERSION
)
{ 

    //--------------------------------------------------------------------------
    // get the version number
    //--------------------------------------------------------------------------

    if (version    != NULL) (*version   ) = GrB_VERSION ;
    if (subversion != NULL) (*subversion) = GrB_SUBVERSION ;

    return (GrB_SUCCESS) ;
}

