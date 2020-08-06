//------------------------------------------------------------------------------
// GxB_Matrix_Option_set: set an option in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_transpose.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Matrix_Option_set      // set an option in a matrix
(
    GrB_Matrix A,                   // descriptor to modify
    GxB_Option_Field field,         // option to change
    ...                             // value to change it to
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_SUCCESS ;
    GB_WHERE (A, "GxB_Matrix_Option_set (A, field, value)") ;
    GB_BURBLE_START ("GxB_set") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    ASSERT_MATRIX_OK (A, "A to set option", GB0) ;

    GB_MATRIX_WAIT (A) ;
    GB_BURBLE_DENSE (A, "(A %s) ") ;

    //--------------------------------------------------------------------------
    // set the matrix option
    //--------------------------------------------------------------------------

    va_list ap ;

    switch (field)
    {

        case GxB_HYPER_SWITCH : 

            {
                va_start (ap, field) ;
                double hyper_switch = va_arg (ap, double) ;
                va_end (ap) ;
                A->hyper_switch = (float) hyper_switch ;
                // conform the matrix to its new desired sparsity structure
                info = GB_conform (A, Context) ;
            }
            break ;

        case GxB_SPARSITY :

            {
                va_start (ap, field) ;
                int sparsity = va_arg (ap, int) ;
                va_end (ap) ;
                switch (sparsity)
                {
                    case GxB_DEFAULT :
                    case GxB_HYPERSPARSE :
                    case GxB_SPARSE :
                    case GxB_BITMAP :
                    case GxB_FULL :
                        A->sparsity = sparsity ;
                        info = GB_conform (A, Context) ;
                        break ;
                    default :
                        GB_ERROR (GrB_INVALID_VALUE, "unsupported sparsity"
                            " [%d], must be one of: GxB_DEFAULT [%d]\n"
                            "GxB_HYPERSPARSE [%d], GxB_SPARSE [%d], "
                            "GxB_BITMAP [%d], or GxB_FULL [%d]\n",
                            sparsity, (int) GxB_DEFAULT,
                            (int) GxB_HYPERSPARSE, (int) GxB_SPARSE,
                            (int) GxB_BITMAP, (int) GxB_FULL) ;
                }
            }

        case GxB_FORMAT : 

            {
                va_start (ap, field) ;
                int format = va_arg (ap, int) ;
                va_end (ap) ;
                if (! (format == GxB_BY_ROW || format == GxB_BY_COL))
                { 
                    GB_ERROR (GrB_INVALID_VALUE,
                        "unsupported format [%d], must be one of:\n"
                        "GxB_BY_ROW [%d] or GxB_BY_COL [%d]", format,
                        (int) GxB_BY_ROW, (int) GxB_BY_COL) ;
                }
                // the value is normally GxB_BY_ROW (0) or GxB_BY_COL (1), but
                // any nonzero value results in GxB_BY_COL.
                bool new_csc = (format != GxB_BY_ROW) ;
                // conform the matrix to the new CSR/CSC format
                if (A->is_csc != new_csc)
                { 
                    // A = A', done in place, and change to the new format.
                    // transpose: no typecast, no op, in place of A
                    GB_BURBLE_N (GB_NNZ (A), "(transpose) ") ;
                    info = GB_transpose (NULL, NULL, new_csc, A,
                        NULL, NULL, NULL, false, Context);
                    ASSERT (GB_IMPLIES (info == GrB_SUCCESS,
                        A->is_csc == new_csc)) ;
                }
            }
            break ;

        default : 

            GB_ERROR (GrB_INVALID_VALUE,
                "invalid option field [%d], must be one of:\n"
                "GxB_HYPER_SWITCH [%d], GxB_SPARSITY [%d], or GxB_FORMAT [%d]",
                (int) field, (int) GxB_HYPER_SWITCH, (int) GxB_SPARSITY,
                (int) GxB_FORMAT) ;

    }

    GB_BURBLE_END ;
    return (info) ;
}

