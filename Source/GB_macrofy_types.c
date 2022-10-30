//------------------------------------------------------------------------------
// GB_macrofy_types: construct typedefs for up to 6 types
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_types
(
    FILE *fp,
    // input:
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype,
    GrB_Type xtype,
    GrB_Type ytype,
    GrB_Type ztype
)
{

    //--------------------------------------------------------------------------
    // define complex types, if any types are GxB_FC32 or GxB_FC64
    //--------------------------------------------------------------------------

    // If the user-defined types and operators wish to use the GraphBLAS
    // typedefs GxB_FC32_t and GxB_FC64_t, they can add the string
    // "#include <GxB_complex.h>\n" to their strng definition.  See
    // Demo/Source/usercomplex.c.

    if (ctype == GxB_FC32 || ctype == GxB_FC64 ||
        atype == GxB_FC32 || atype == GxB_FC64 ||
        btype == GxB_FC32 || btype == GxB_FC64 ||
        xtype == GxB_FC32 || xtype == GxB_FC64 ||
        ytype == GxB_FC32 || ytype == GxB_FC64 ||
        ztype == GxB_FC32 || ztype == GxB_FC64)
    {
        fprintf (fp, "#include <GxB_complex.h>\n") ;
    }

    //--------------------------------------------------------------------------
    // create typedefs, checking for duplicates
    //--------------------------------------------------------------------------

    const char *defn [6] ;
    defn [0] = (ctype == NULL) ? NULL : ctype->defn ;
    defn [1] = (atype == NULL) ? NULL : atype->defn ;
    defn [2] = (btype == NULL) ? NULL : btype->defn ;
    defn [3] = (xtype == NULL) ? NULL : xtype->defn ;
    defn [4] = (ytype == NULL) ? NULL : ytype->defn ;
    defn [5] = (ztype == NULL) ? NULL : ztype->defn ;

    for (int k = 0 ; k <= 5 ; k++)
    {
        if (defn [k] != NULL)
        {
            // only print this typedef it is unique
            bool is_unique = true ;
            for (int j = 0 ; j < k && is_unique ; j++)
            {
                if (defn [j] != NULL && strcmp (defn [j], defn [k]) == 0)
                {
                    is_unique = false ;
                }
            }
            if (is_unique)
            {
                // the typedef is unique: include it in the .h file
                fprintf (fp, "%s\n\n", defn [k]) ;
            }
        }
    }
}

