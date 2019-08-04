//------------------------------------------------------------------------------
// gb_export: export a GrB_Matrix as a MATLAB matrix or GraphBLAS struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// mxArray pargout [0] = gb_export (&C, kind) ; exports C as a MATLAB matrix
// and frees the remaining content of C.

#include "gb_matlab.h"

mxArray *gb_export              // return the exported MATLAB matrix or struct
(
    GrB_Matrix *C_handle,       // GrB_Matrix to export and free
    kind_enum_t kind            // gb, sparse, or full
)
{

    if (kind == KIND_SPARSE)
    {

        //----------------------------------------------------------------------
        // export C as a MATLAB sparse matrix
        //----------------------------------------------------------------------

        // Typecast to double, if C is integer (int8, ..., uint64)

        return (gb_export_to_mxsparse (C_handle)) ;

    }
    else if (kind == KIND_FULL)
    {

        //----------------------------------------------------------------------
        // export C as a MATLAB dense matrix
        //----------------------------------------------------------------------

        // No typecasting is needed since MATLAB supports all the same types.

        GrB_Index nrows, ncols, nzmax, *Cp, *Ci ;
        int64_t nonempty ;
        void *Cx ;
        GrB_Type ctype ;
        OK (GxB_Matrix_export_CSC (C_handle, &ctype, &nrows, &ncols, &nzmax,
            &nonempty, &Cp, &Ci, &Cx, NULL)) ;
        gb_mxfree (&Cp) ;
        gb_mxfree (&Ci) ;
        return (gb_export_to_mxfull (&Cx, nrows, ncols, ctype)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // export C as a MATLAB struct containing a verbatim GrB_Matrix
        //----------------------------------------------------------------------

        // No typecasting is needed since the MATLAB struct can hold all of the
        // opaque content of the GrB_Matrix.

        return (gb_export_to_mxstruct (C_handle)) ;

    }
}

