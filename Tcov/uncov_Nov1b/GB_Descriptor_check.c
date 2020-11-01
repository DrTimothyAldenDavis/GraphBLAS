//------------------------------------------------------------------------------
// GB_Descriptor_check: check and print a Descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

//------------------------------------------------------------------------------
// GB_dc: check a single descriptor field
//------------------------------------------------------------------------------

static GrB_Info GB_dc
(
    int kind,                           // 0, 1, or 2
    const char *field,
    const GrB_Desc_Value v,
    const GrB_Desc_Value nondefault,    // for kind == 0
    int pr,                             // print level
    FILE *f 
)
{

    bool ok = true ;
    GrB_Info info = GrB_SUCCESS ;

    GBPR0 ("    d.%s = ", field) ;
    switch ((int) v)
    {
        case GxB_DEFAULT             : GB_cov[2222]++ ;  GBPR0 ("default   ") ; break ;
// covered (2222): 476
        case GrB_COMP                : GB_cov[2223]++ ;  GBPR0 ("complement") ; break ;
// covered (2223): 20
        case GrB_STRUCTURE           : GB_cov[2224]++ ;  GBPR0 ("structure ") ; break ;
// covered (2224): 16
        case GrB_COMP+GrB_STRUCTURE  : GB_cov[2225]++ ;  GBPR0 ("structural complement") ; break ;
// covered (2225): 18
        case GrB_TRAN                : GB_cov[2226]++ ;  GBPR0 ("transpose ") ; break ;
// covered (2226): 80
        case GrB_REPLACE             : GB_cov[2227]++ ;  GBPR0 ("replace   ") ; break ;
// covered (2227): 40
        case GxB_AxB_SAXPY           : GB_cov[2228]++ ;  GBPR0 ("saxpy     ") ; break ;
// covered (2228): 2
        case GxB_AxB_GUSTAVSON       : GB_cov[2229]++ ;  GBPR0 ("Gustavson ") ; break ;
// covered (2229): 10
        case GxB_AxB_HEAP            : GB_cov[2230]++ ;  GBPR0 ("heap      ") ; break ;
// covered (2230): 10
        case GxB_AxB_HASH            : GB_cov[2231]++ ;  GBPR0 ("hash      ") ; break ;
// covered (2231): 2
        case GxB_AxB_DOT             : GB_cov[2232]++ ;  GBPR0 ("dot       ") ; break ;
// covered (2232): 6
        default                      : GB_cov[2233]++ ;  GBPR0 ("unknown   ") ;
// covered (2233): 10
            info = GrB_INVALID_OBJECT ;
            ok = false ;
            break ;
    }

    if (ok)
    {
        if (kind == 0)
        {
            // descriptor field can be set to the default,
            // or one non-default value
            if (! (v == GxB_DEFAULT || v == nondefault))
            {   GB_cov[2234]++ ;
// covered (2234): 4
                ok = false ;
            }
        }
        else if (kind == 1)
        {
            // mask
            if (! (v == GxB_DEFAULT || v == GrB_COMP || v == GrB_STRUCTURE ||
                   v == (GrB_COMP + GrB_STRUCTURE)))
            {   GB_cov[2235]++ ;
// NOT COVERED (2235):
GB_GOTCHA ;
                ok = false ;
            }
        }
        else // kind == 2
        {
            // GxB_AxB_METHOD:
            if (! (v == GxB_DEFAULT || v == GxB_AxB_GUSTAVSON
                || v == GxB_AxB_HEAP || v == GxB_AxB_DOT
                || v == GxB_AxB_HASH || v == GxB_AxB_SAXPY))
            {   GB_cov[2236]++ ;
// covered (2236): 4
                ok = false ;
            }
        }
    }

    if (!ok)
    {   GB_cov[2237]++ ;
// covered (2237): 18
        GBPR0 (" (invalid value for this field)") ;
        info = GrB_INVALID_OBJECT ;
    }

    GBPR0 ("\n") ;

    return (info) ;
}

//------------------------------------------------------------------------------
// GB_Descriptor_check
//------------------------------------------------------------------------------

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_Descriptor_check    // check a GraphBLAS descriptor
(
    const GrB_Descriptor D,     // GraphBLAS descriptor to print and check
    const char *name,           // name of the descriptor, optional
    int pr,                     // print level
    FILE *f                     // file for output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GBPR0 ("\n    GraphBLAS Descriptor: %s ", GB_NAME) ;

    if (D == NULL)
    {   GB_cov[2238]++ ;
// covered (2238): 2
        GBPR0 ("NULL\n") ;
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // check object
    //--------------------------------------------------------------------------

    GB_CHECK_MAGIC (D, "Descriptor") ;

    GBPR0 ("\n") ;

    GrB_Info info [5] ;
    info [0] = GB_dc (0, "out     ", D->out,  GrB_REPLACE, pr, f) ;
    info [1] = GB_dc (1, "mask    ", D->mask, GxB_DEFAULT, pr, f) ;
    info [2] = GB_dc (0, "in0     ", D->in0,  GrB_TRAN,    pr, f) ;
    info [3] = GB_dc (0, "in1     ", D->in1,  GrB_TRAN,    pr, f) ;
    info [4] = GB_dc (2, "axb     ", D->axb,  GxB_DEFAULT, pr, f) ;

    for (int i = 0 ; i < 5 ; i++)
    {
        if (info [i] != GrB_SUCCESS)
        {   GB_cov[2239]++ ;
// covered (2239): 18
            GBPR0 ("    Descriptor field set to an invalid value\n") ;
            return (GrB_INVALID_OBJECT) ;
        }
    }

    int nthreads_max = D->nthreads_max ;
    double chunk = D->chunk ;

    GBPR0 ("    d.nthreads = ") ;
    if (nthreads_max <= GxB_DEFAULT)
    {   GB_cov[2240]++ ;
// covered (2240): 118
        GBPR0 ("default\n") ;
    }
    else
    {   GB_cov[2241]++ ;
// covered (2241): 2
        GBPR0 ("%d\n", nthreads_max) ;
    }
    GBPR0 ("    d.chunk    = ") ;
    if (chunk <= GxB_DEFAULT)
    {   GB_cov[2242]++ ;
// covered (2242): 118
        GBPR0 ("default\n") ;
    }
    else
    {   GB_cov[2243]++ ;
// covered (2243): 2
        GBPR0 ("%g\n", chunk) ;
    }

    if (D->use_mkl)
    {
        GBPR0 ("    d.use_mkl = true") ;
    }

    return (GrB_SUCCESS) ;
}

