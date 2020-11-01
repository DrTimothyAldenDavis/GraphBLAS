//------------------------------------------------------------------------------
// GB_Type_compatible: return true if domains are compatible
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Two domains are compatible for typecasting between them if both are built-in
// types (of any kind) or if both are the same user-defined type.

#include "GB.h"
 
GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
bool GB_Type_compatible             // check if two types can be typecast
(
    const GrB_Type atype,
    const GrB_Type btype
)
{

    if (atype == NULL || btype == NULL)
    {   GB_cov[2458]++ ;
// NOT COVERED (2458):
GB_GOTCHA ;
        // built-in positional ops have null op->[xy]type, and are compatible
        // with anything.  TODO: FIXME.
        return (true) ;
    }
    else if (atype->code == GB_UDT_code || btype->code == GB_UDT_code)
    {   GB_cov[2459]++ ;
// covered (2459): 1258030
        // two user types must be identical to be compatible
        return (atype == btype) ;
    }
    else
    {   GB_cov[2460]++ ;
// covered (2460): 61657604
        // any built-in domain is compatible with any other built-in domain
        return (true) ;
    }
}

