//------------------------------------------------------------------------------
// GB_UnaryOp_check: check and print a unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_UnaryOp_check   // check a GraphBLAS unary operator
(
    const GrB_UnaryOp op,   // GraphBLAS operator to print and check
    const char *name,       // name of the operator
    int pr,                 // print level
    FILE *f                 // file for output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GBPR0 ("\n    GraphBLAS UnaryOp: %s ", GB_NAME) ;

    if (op == NULL)
    { 
        // GrB_error status not modified since this may be an optional argument
        GBPR0 ("NULL\n") ;
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // check object
    //--------------------------------------------------------------------------

    GB_CHECK_MAGIC (op, "UnaryOp") ;

    if (op->opcode >= GB_USER_opcode)
    { 
        GBPR0 ("(user-defined) ") ;
    }
    else
    { 
        GBPR0 ("(built-in) ") ;
    }

    GBPR0 ("z=%s(x)\n", op->name) ;

    if (op->function == NULL)
    { 
        GBPR0 ("    function pointer is NULL\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    if (op->opcode != GB_USER_opcode)
    {
        if (op->opcode < GB_ONE_opcode || op->opcode >= GB_FIRST_opcode)
        { 
            GBPR0 ("    invalid opcode\n") ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    GrB_Info info ;

    info = GB_Type_check (op->ztype, "ztype", pr, f) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("    UnaryOP has an invalid ztype\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    info = GB_Type_check (op->xtype, "xtype", pr, f) ;
    if (info != GrB_SUCCESS)
    { 
        GBPR0 ("    UnaryOP has an invalid xtype\n") ;
        return (GrB_INVALID_OBJECT) ;
    }

    return (GrB_SUCCESS) ;
}

