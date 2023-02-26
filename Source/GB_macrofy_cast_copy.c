//------------------------------------------------------------------------------
// GB_macrofy_cast_copy: construct a macro for copying input to output
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// constructs a macro of the form:

//      #define GB_COPY_A_TO_C(Cx,pC,Ax,pA,A_iso) Cx [pC] = (cast) Ax [pA]

// The matrices A and C are named by aname and cname, and have type atype
// and ctype.  A_iso is true if A is iso.  If either atype or ctype is NULL,
// an empty macro is created.

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_cast_copy
(
    FILE *fp,
    // input:
    const char *cname,          // name of the C matrix (typically "C")
    const char *aname,          // name of the A matrix (typically "A" or "B")
    const GrB_Type ctype,       // the type of the C matrix
    const GrB_Type atype,       // the type of the A matrix
    const bool A_iso            // true if A is iso
)
{

    fprintf (fp, "#define GB_COPY_%s_to_%s(%sx,p%s,%sx,p%s,%s_iso)",
        aname, cname, cname, cname, aname, aname, aname) ;

    if (ctype == NULL || atype == NULL)
    {
        // empty macro if atype or ctype are NULL (value not needed)
        fprintf (fp, "\n") ;
        return ;
    }

    int nargs ;
    const char *f = GB_macrofy_cast_expression (fp, ctype, atype, &nargs) ;

    char carg [256] ;
    snprintf (carg, 255, "%sx [p%s]", cname, cname) ;
    char aarg [256] ;
    if (A_iso)
    {
        snprintf (aarg, 255, "%sx [0]", aname) ;
    }
    else
    {
        snprintf (aarg, 255, "%sx [p%s]", aname, aname) ;
    }

    fprintf (fp, " ") ;
    if (f == NULL)
    {
        fprintf (fp, "%s = (%s) %s", carg, ctype->name, aarg) ;
    }
    else if (nargs == 3)
    {
        fprintf (fp, f, carg, aarg, aarg) ;
    }
    else
    {
        fprintf (fp, f, carg, aarg) ;
    }
    fprintf (fp, "\n") ;
}
