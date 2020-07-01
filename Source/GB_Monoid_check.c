//------------------------------------------------------------------------------
// GB_Monoid_check: check and print a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_Monoid_check        // check a GraphBLAS monoid
(
    const GrB_Monoid monoid,    // GraphBLAS monoid to print and check
    const char *name,           // name of the monoid, optional
    int pr,                     // print level
    FILE *f                     // file for output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GBPR0 ("\n    GraphBLAS Monoid: %s ", GB_NAME) ;

    if (monoid == NULL)
    { 
        GBPR0 ("NULL\n") ;
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // check object
    //--------------------------------------------------------------------------

    GB_CHECK_MAGIC (monoid, "Monoid") ;
    GBPR0 (monoid->builtin ? "(built-in)" : "(user-defined)") ;

    GrB_Info info = GB_BinaryOp_check (monoid->op, "monoid->op", pr, f) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("    Monoid contains an invalid operator\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    if (monoid->op->xtype != monoid->op->ztype ||
        monoid->op->ytype != monoid->op->ztype)
    { 
        GBPR0 ("    All domains of operator must be the same\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    // print the identity and terminal values
    if (pr != GxB_SILENT)
    { 
        GBPR ("    identity: [ ") ;
        info = GB_entry_check (monoid->op->ztype, monoid->identity, pr, f) ;
        if (info != GrB_SUCCESS) return (info) ;
        GBPR (" ] ") ;
        // print the terminal value, if present
        if (monoid->terminal != NULL)
        { 
            GBPR ("terminal: [ ") ;
            info = GB_entry_check (monoid->op->ztype, monoid->terminal, pr, f) ;
            if (info != GrB_SUCCESS) return (info) ;
            GBPR (" ]") ;
        }
        GBPR ("\n") ;
    }

    return (GrB_SUCCESS) ;
}

