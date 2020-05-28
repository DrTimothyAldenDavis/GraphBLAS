//------------------------------------------------------------------------------
// GxB_getLibraryVersion: get information about the GraphBLAS library
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// These values are available at runtime via GxB_getLibraryVersion, and at
// compile time via GxB_IMPLEMENTATION_* macros.

#include "GB.h"

GB_PUBLIC
GrB_Info GxB_getLibraryVersion
(
    const char **name,              // name of the library
    const char **date,              // date of release
    const char **about,             // information about the library
    const char **license,           // library license
    unsigned int *version,          // version numbers
    unsigned int *minorversion,
    unsigned int *subversion
)
{ 

    //--------------------------------------------------------------------------
    // get the library version information
    //--------------------------------------------------------------------------

    if (name         != NULL) (*name        ) = GxB_IMPLEMENTATION_NAME ;
    if (date         != NULL) (*date        ) = GxB_IMPLEMENTATION_DATE ;
    if (about        != NULL) (*about       ) = GxB_IMPLEMENTATION_ABOUT ;
    if (license      != NULL) (*license     ) = GxB_IMPLEMENTATION_LICENSE ;
    if (version      != NULL) (*version     ) = GxB_IMPLEMENTATION_MAJOR ;
    if (minorversion != NULL) (*minorversion) = GxB_IMPLEMENTATION_MINOR ;
    if (subversion   != NULL) (*subversion  ) = GxB_IMPLEMENTATION_SUB ;

    return (GrB_SUCCESS) ;
}

