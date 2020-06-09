//------------------------------------------------------------------------------
// gb_string_to_typecode: return the GraphBLAS type from a string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gb_matlab.h"

int gb_string_to_typecode // return the GB_Type_code from a string
(
    const char *typename
)
{ 

    if (MATCH (typename, "double"  )) return ((int) GB_FP64_code) ;
    if (MATCH (typename, "single"  )) return ((int) GB_FP32_code) ;
    if (MATCH (typename, "logical" )) return ((int) GB_BOOL_code) ;
    if (MATCH (typename, "int8"    )) return ((int) GB_INT8_code) ;
    if (MATCH (typename, "int16"   )) return ((int) GB_INT16_code) ;
    if (MATCH (typename, "int32"   )) return ((int) GB_INT32_code) ;
    if (MATCH (typename, "int64"   )) return ((int) GB_INT64_code) ;
    if (MATCH (typename, "uint8"   )) return ((int) GB_UINT8_code) ;
    if (MATCH (typename, "uint16"  )) return ((int) GB_UINT16_code) ;
    if (MATCH (typename, "uint32"  )) return ((int) GB_UINT32_code) ;
    if (MATCH (typename, "uint64"  )) return ((int) GB_UINT64_code) ;

    if (MATCH (typename, "single complex"))
    { 
        return ((int) GB_FC32_code) ;
    }

    if (MATCH (typename, "double complex") || MATCH (typename, "complex"))
    { 
        return ((int) GB_FC64_code) ;
    }

    // input string is not a valid type name
    return (-1) ;
}

