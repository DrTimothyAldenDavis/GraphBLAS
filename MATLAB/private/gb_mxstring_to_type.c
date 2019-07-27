//------------------------------------------------------------------------------
// gb_mxstring_to_type: return the GraphBLAS type from a MATLAB string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gbmex.h"

GrB_Type gb_mxstring_to_type    // return the GrB_Type from a MATLAB string
(
    const mxArray *s
)
{

    #define LEN 256
    char classname [LEN+2] ;
    int len = gb_mxstring_to_string (classname, LEN, s) ;

    CHECK_ERROR (len < 0, "unknown type") ;

    if (MATCH (classname, "logical" )) return (GrB_BOOL) ;
    if (MATCH (classname, "int8"    )) return (GrB_INT8) ;
    if (MATCH (classname, "int16"   )) return (GrB_INT16) ;
    if (MATCH (classname, "int32"   )) return (GrB_INT32) ;
    if (MATCH (classname, "int64"   )) return (GrB_INT64) ;
    if (MATCH (classname, "uint8"   )) return (GrB_UINT8) ;
    if (MATCH (classname, "uint16"  )) return (GrB_UINT16) ;
    if (MATCH (classname, "uint32"  )) return (GrB_UINT32) ;
    if (MATCH (classname, "uint64"  )) return (GrB_UINT64) ;
    if (MATCH (classname, "single"  )) return (GrB_FP32) ;
    if (MATCH (classname, "double"  )) return (GrB_FP64) ;
    if (MATCH (classname, "complex" ))
    {
        #ifdef GB_COMPLEX_TYPE
        return (gb_complex_type) ;
        #else
        ERROR ("complex not supported") ;
        #endif
    }

    ERROR ("unknown type") ;
    return (NULL) ;
}

