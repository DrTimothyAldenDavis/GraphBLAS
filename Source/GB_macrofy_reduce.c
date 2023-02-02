//------------------------------------------------------------------------------
// GB_macrofy_reduce: construct all macros for a reduction to scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_reduce      // construct all macros for GrB_reduce to scalar
(
    FILE *fp,               // target file to write, already open
    // input:
    uint64_t rcode,         // encoded problem
    GrB_Monoid monoid,      // monoid to macrofy
    GrB_Type atype          // type of the A matrix to reduce
)
{ 

    //--------------------------------------------------------------------------
    // extract the reduction rcode
    //--------------------------------------------------------------------------

    // monoid
    int red_ecode   = GB_RSHIFT (rcode, 22, 5) ;
    int id_ecode    = GB_RSHIFT (rcode, 17, 5) ;
    int term_ecode  = GB_RSHIFT (rcode, 12, 5) ;
    bool is_term    = (term_ecode < 30) ;

    // type of the monoid
    int zcode       = GB_RSHIFT (rcode, 8, 4) ;

    // type of A
    int acode       = GB_RSHIFT (rcode, 4, 4) ;

    // zombies
    int azombies    = GB_RSHIFT (rcode, 2, 1) ;

    // format of A
    int asparsity   = GB_RSHIFT (rcode, 0, 2) ;

    //--------------------------------------------------------------------------
    // copyright, license, and describe monoid
    //--------------------------------------------------------------------------

    GB_macrofy_copyright (fp) ;
    fprintf (fp, "// monoid: (%s, %s)\n\n",
        monoid->op->name, monoid->op->ztype->name) ;

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GB_macrofy_typedefs (fp, NULL, atype, NULL, NULL, NULL, monoid->op->ztype) ;

    //--------------------------------------------------------------------------
    // construct the macros for the type names
    //--------------------------------------------------------------------------

    fprintf (fp, "// monoid type:\n") ;

    GB_macrofy_type (fp, "Z", monoid->op->ztype->name) ;

    //--------------------------------------------------------------------------
    // construct the monoid macros
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// reduction monoid:\n") ;
    GB_macrofy_monoid (fp, red_ecode, id_ecode, term_ecode, monoid) ;

    fprintf (fp, "#define GB_GETA_AND_UPDATE(z,Ax,p) \\\n") ;
    if (atype == monoid->op->ztype)
    {
        // z += Ax [p], with no typecasting.  A is never iso.
        fprintf (fp, "    GB_UPDATE(z, Ax [p]) ;    // z += Ax [p]\n") ;
    }
    else
    {
        // aij = (ztype) Ax [p] ; z += aij ; with typecasting.  A is never iso.
        fprintf (fp, "{                             \\\n"
                     "    /* z += (ztype) Ax [p] */ \\\n"
                     "    GB_DECLAREA (aij) ;       \\\n"
                     "    GB_GETA (aij, Ax, p, ) ;  \\\n"
                     "    GB_UPDATE (z, aij) ;      \\\n"
                     "}\n"
                     ) ;
    }

    //--------------------------------------------------------------------------
    // construct the macros for A
    //--------------------------------------------------------------------------

    // iso reduction is handled by GB_iso_reduce_to_scalar, which takes
    // O(log(nvals(A))) for any monoid and uses the function pointer of the
    // monoid operator.  No JIT kernel is ever required to reduce an iso matrix
    // to a scalar, even for user-defined types and monoids.

    GB_macrofy_input (fp, "a", "A", "A", true, monoid->op->ztype,
        atype, asparsity, acode, false, azombies) ;

    //--------------------------------------------------------------------------
    // reduction method
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// panel size for reduction:\n") ;
    int zsize = (int) monoid->op->ztype->size ;
    if (zsize == 1)
    {
        fprintf (fp, "#define GB_PANEL 8\n") ;
    }
    else if (zsize < 16)
    {
        fprintf (fp, "#define GB_PANEL 16\n") ;
    }
    else
    {
        fprintf (fp, "#define GB_PANEL 1\n") ;
    }

}

