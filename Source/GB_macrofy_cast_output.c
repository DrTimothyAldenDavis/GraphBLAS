//------------------------------------------------------------------------------
// GB_macrofy_cast_output: construct a macro and defn for typecasting to output
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// constructs a macro of the form:

//      #define macro(z,x...) ... = z

// The name of the macro is given by macro_name.
// The name of the variable z is given by zarg.
// The outputs x,... are given by xargs (which may contain a comma).
// xexpr is an expression using the xargs that produce a value of type xtype.
// z has type ztype.

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_cast_output
(
    FILE *fp,
    // input:
    const char *macro_name,     // name of the macro: #define macro(z,x...)
    const char *zarg,           // name of the z argument of the macro
    const char *xargs,          // one or more x arguments
    const char *xexpr,          // an expression based on xargs
    const GrB_Type ztype,       // the type of the z input
    const GrB_Type xtype        // the type of the x output
)
{

    if (ztype == NULL || xtype == NULL)
    {
        // empty macro if xtype or ztype are NULL (value not needed)
        fprintf (fp, "#define %s(%s,%s)\n", macro_name, zarg, xargs) ;
        return ;
    }

    int nargs ;
    const char *f = GB_macrofy_cast_expression (fp, xtype, ztype, &nargs) ;

    if (f == NULL)
    {
        fprintf (fp, "#define %s(%s,%s) %s = ((%s) (%s))\n",
            macro_name, zarg, xargs, xexpr, xtype->name, zarg) ;
    }
    else
    {
        #define SLEN 1024
        char s [SLEN+1] ;
        if (nargs == 3)
        {
            snprintf (s, SLEN, f, xexpr, zarg, zarg) ;
        }
        else
        {
            snprintf (s, SLEN, f, xexpr, zarg) ;
        }
        fprintf (fp, "#define %s(%s,%s) %s\n", macro_name, zarg, xargs, s) ;
        // FIXME: use fprintf (fp, f, xexpr, zarg, zarg) ; or
        // fprintf (fp, f, xexpr, zarg) ;
        // instead of char s.
    }
}
