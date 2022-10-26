//------------------------------------------------------------------------------
// GB_macrofy_input: construct a macro to load values from an input matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_input
(
    FILE *fp,
    // input:
    const char *aname,      // name of the scalar aij = ...
    const char *Aname,      // name of the input matrix
    GrB_Type xtype,         // type of aij
    GrB_Type atype,         // type of the input matrix
    int asparsity,          // sparsity format of the input matrix
    int acode,              // type code of the input (0 if iso)
    int A_iso_code          // 1 if A is iso
)
{

    int A_is_pattern = (acode == 0) ? 1 : 0 ;
    fprintf (fp, "\n// %s matrix:\n", Aname) ;
    fprintf (fp, "#define GB_%s_IS_PATTERN %d\n", Aname, A_is_pattern) ;
    fprintf (fp, "#define GB_%s_ISO %d\n", Aname, A_iso_code) ;
    GB_macrofy_sparsity (fp, Aname, asparsity) ;
    fprintf (fp, "#define GB_%s_TYPENAME %s\n", Aname, atype->name) ;

    if (A_is_pattern)
    {
        // no need to access the values of A
        fprintf (fp, "#define GB_DECLARE%s(%s)\n", Aname, aname) ;
        fprintf (fp, "#define GB_DECLARE%s_MOD(modifier,%s)\n", Aname, aname) ;
        fprintf (fp, "#define GB_GET%s(%s,%sx,p)\n", Aname, aname, Aname) ;
    }
    else
    {
        #define SLEN 256
        char macro_name [SLEN+1], xargs [SLEN+1], xexpr [SLEN+1] ;

        // declare a scalar or work array.  The modifier can be empty,
        // or (for example) "__shared__" for a CUDA shared array or scalar.
        fprintf (fp, "#define GB_DECLARE%s(%s) %s %s\n",
            Aname, aname, xtype->name, aname) ;

        // declare a scalar or work array, prefied with a modifier
        fprintf (fp, "#define GB_DECLARE%s_MOD(modifier,%s) modifier %s %s\n",
            Aname, aname, xtype->name, aname) ;

        // construct the macro:
        // #define GB_GETA(a,Ax,p) a = (xtype) Ax [iso ? 0 : p]
        // to load a value from the A matrix, and typecast it to
        // the scalar a of type xtype
        snprintf (macro_name, SLEN, "GB_GET%s", Aname) ;
        snprintf (xargs, SLEN, "%sx,p", Aname) ;
        snprintf (xexpr, SLEN, A_iso_code ? "%sx [0]" : "%sx [p]", Aname) ;
        GB_macrofy_cast (fp, macro_name, aname, xargs, xexpr, xtype, atype) ;
    }
}

