//------------------------------------------------------------------------------
// GB_mex_export: test import/export
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2018, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_export (C, format, hyper, csc, dump)"

#define FREE_ALL                        \
{                                       \
    GB_MATRIX_FREE (&C) ;               \
    if (Ap != NULL) { GB_Global.nmalloc-- ; mxFree (Ap) ; Ap = NULL ; }     \
    if (Ah != NULL) { GB_Global.nmalloc-- ; mxFree (Ah) ; Ah = NULL ; }     \
    if (Ai != NULL) { GB_Global.nmalloc-- ; mxFree (Ai) ; Ai = NULL ; }     \
    if (Aj != NULL) { GB_Global.nmalloc-- ; mxFree (Aj) ; Aj = NULL ; }     \
    if (Ax != NULL) { GB_Global.nmalloc-- ; mxFree (Ax) ; Ax = NULL ; }     \
    GB_mx_put_global (true, 0) ;        \
}

#define OK(method)                      \
{                                       \
    info = method ;                     \
    if (info != GrB_SUCCESS)            \
    {                                   \
        FREE_ALL ;                      \
        return (info) ;                 \
    }                                   \
}

GrB_Vector v = NULL ;
GrB_Matrix C = NULL ;
GrB_Index *Ap = NULL ;
GrB_Index *Ah = NULL ;
GrB_Index *Ai = NULL ;
GrB_Index *Aj = NULL ;
GrB_Index nrows = 0 ;
GrB_Index ncols = 0 ;
GrB_Index nvals = 0 ;
char *Ax = NULL ;
int format = 0 ;
bool is_hyper = false ;
bool is_csc = true ;
GrB_Info info = GrB_SUCCESS ;
GrB_Descriptor desc = NULL ;
bool dump = false ;
GrB_Type type = NULL ;

//------------------------------------------------------------------------------

GrB_Info import_export (GB_Context Context)
{
    GrB_Index r = 0 ;
    GrB_Index c = 0  ;
    size_t s = 0 ;

    //--------------------------------------------------------------------------
    // export/import a vector
    //--------------------------------------------------------------------------

    if (GB_VECTOR_OK (C))
    {
        OK (GxB_Vector_export ((GrB_Vector *) (&C), &type, &nrows, &nvals,
            &Ai, &Ax, desc)) ;

        if (dump)
        {
            printf ("export standard CSC vector: %llu-by-1, nvals %llu:\n",
                nrows, nvals) ;
            OK (GB_check (type, "type", GB1)) ;
            OK (GxB_Type_size (&s, type)) ;
            GB_Type_code code = type->code ;

            for (int64_t p = 0 ; p < nvals ; p++)
            {
                printf ("  row %llu value ", Ai [p]) ;
                GB_code_check (code, Ax + p * s, stdout, Context) ;
                printf ("\n") ;
            }
        }

        OK (GxB_Vector_import ((GrB_Vector *) (&C), type, nrows, nvals,
            &Ai, &Ax, desc)) ;

        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // export/import a matrix
    //--------------------------------------------------------------------------

    switch (format)
    {

        //----------------------------------------------------------------------
        case 0 : 
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_CSR (&C, &type, &nrows, &ncols, &nvals,
                    &Ap, &Aj, &Ax, desc)) ;

            if (dump)
            {
                printf ("export standard CSR: %llu-by-%llu, nvals %llu:\n",
                    nrows, ncols, nvals) ;
                OK (GB_check (type, "type", GB1)) ;
                OK (GxB_Type_size (&s, type)) ;
                GB_Type_code code = type->code ;

                for (int64_t i = 0 ; i < nrows ; i++)
                {
                    printf ("Row %lld\n", i) ;
                    for (int64_t p = Ap [i] ; p < Ap [i+1] ; p++)
                    {
                        printf ("  col %llu value ", Aj [p]) ;
                        GB_code_check (code, Ax + p * s, stdout, Context) ;
                        printf ("\n") ;
                    }
                }
            }

            OK (GxB_Matrix_import_CSR (&C, type, nrows, ncols, nvals,
                &Ap, &Aj, &Ax, desc)) ;

            OK (GB_check (C, "C reimported", dump ? GB3 : GB0)) ;
            break ;

        //----------------------------------------------------------------------
        case 1 : 
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_CSC (&C, &type, &nrows, &ncols, &nvals,
                &Ap, &Ai, &Ax, desc)) ;

            if (dump)
            {
                printf ("export standard CSC: %llu-by-%llu, nvals %llu:\n",
                    nrows, ncols, nvals) ;
                OK (GB_check (type, "type", GB1)) ;
                OK (GxB_Type_size (&s, type)) ;
                GB_Type_code code = type->code ;

                for (int64_t j = 0 ; j < ncols ; j++)
                {
                    printf ("Col %lld\n", j) ;
                    for (int64_t p = Ap [j] ; p < Ap [j+1] ; p++)
                    {
                        printf ("  row %llu value ", Ai [p]) ;
                        GB_code_check (code, Ax + p + s, stdout, Context) ;
                        printf ("\n") ;
                    }
                }

            }

            OK (GxB_Matrix_import_CSC (&C, type, nrows, ncols, nvals,
                &Ap, &Ai, &Ax, desc)) ;

            OK (GB_check (C, "C reimported", dump ? GB3 : GB0)) ;
            break ;

        //----------------------------------------------------------------------
        case 2 : 
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_HyperCSR (&C, &type, &nrows, &ncols, &nvals,
                &r, &Ah, &Ap, &Aj, &Ax, desc)) ;

            if (dump)
            {
                printf ("export hyper CSR: %llu-by-%llu, nvals %llu, r %llu:\n",
                    nrows, ncols, nvals, r) ;
                OK (GB_check (type, "type", GB1)) ;
                OK (GxB_Type_size (&s, type)) ;
                GB_Type_code code = type->code ;

                for (int64_t k = 0 ; k < r ; k++)
                {
                    int64_t i = Ah [k] ;
                    printf ("Row %lld\n", i) ;
                    for (int64_t p = Ap [k] ; p < Ap [k+1] ; p++)
                    {
                        printf ("  col %llu value ", Aj [p]) ;
                        GB_code_check (code, Ax + p * s, stdout, Context) ;
                        printf ("\n") ;
                    }
                }
            }

            OK (GxB_Matrix_import_HyperCSR (&C, type, nrows, ncols, nvals,
                r, &Ah, &Ap, &Aj, &Ax, desc)) ;

            // printf ("Ah %p Ap %p Aj %p Ax %p\n", Ah, Ap, Aj, Ax) ;

            OK (GB_check (C, "C reimported", dump ? GB3 : GB0)) ;
            break ;

        //----------------------------------------------------------------------
        case 3 : 
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_HyperCSC (&C, &type, &nrows, &ncols, &nvals,
                &c, &Ah, &Ap, &Ai, &Ax, desc)) ;

            if (dump)
            {
                printf ("export hyper CSC: %llu-by-%llu, nvals %llu, c %llu:\n",
                    nrows, ncols, nvals, c) ;
                OK (GB_check (type, "type", GB1)) ;
                OK (GxB_Type_size (&s, type)) ;
                GB_Type_code code = type->code ;

                for (int64_t k = 0 ; k < c ; k++)
                {
                    int64_t j = Ah [k] ;
                    printf ("Col %lld\n", j) ;
                    for (int64_t p = Ap [k] ; p < Ap [k+1] ; p++)
                    {
                        printf ("  row %llu value ", Ai [p]) ;
                        GB_code_check (code, Ax + p * s, stdout, Context) ;
                        printf ("\n") ;
                    }
                }
            }

            OK (GxB_Matrix_import_HyperCSC (&C, type, nrows, ncols, nvals,
                c, &Ah, &Ap, &Ai, &Ax, desc)) ;

            OK (GB_check (C, "C reimported", dump ? GB3 : GB0)) ;
            break ;

        //----------------------------------------------------------------------
        default : 
        //----------------------------------------------------------------------

            mexErrMsgTxt ("bad format") ;
            break ;
    }
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;

    // printf ("start, nmalloc %d\n", GB_Global.nmalloc) ;

    // check inputs
    GB_WHERE (USAGE) ;
    if (nargout > 1 || nargin < 1 || nargin > 5)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get format for import/export
    GET_SCALAR (1, int, format, 0) ;

    // get hyper flag 
    GET_SCALAR (2, bool, is_hyper, false) ;

    // get csc flag 
    GET_SCALAR (3, bool, is_csc, true) ;

    // get C (make a deep copy)
    #define GET_DEEP_COPY \
    {                                                                       \
        C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;   \
        if (!is_csc)                                                        \
        {                                                                   \
            /* convert C to CSR */                                          \
            GB_transpose (NULL, NULL, false, C, NULL, NULL) ;               \
        }                                                                   \
        if (is_hyper)                                                       \
        {                                                                   \
            /* convert C to hypersparse */                                  \
            GB_to_nonhyper (C, NULL) ;                                      \
        }                                                                   \
    }
    #define FREE_DEEP_COPY GB_MATRIX_FREE (&C) ;
    GET_DEEP_COPY ;
    if (C == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("C failed") ;
    }
    mxClassID cclass = GB_mx_Type_to_classID (C->type) ;

    // printf ("here, nmalloc %d\n", GB_Global.nmalloc) ;

    // C<Mask> = op(C,A') or op(C,A)
    METHOD (import_export (Context)) ;

    // printf ("finish, nmalloc %d\n", GB_Global.nmalloc) ;

    // return C to MATLAB
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C export/import", false) ;

    // printf ("put matrix to MATLAB, C %p Ap %p Ah %p Ai %p Aj %p Ax %p\n",
    //     C, Ap, Ah, Ai, Aj, Ax) ;
    FREE_ALL ;

    // printf ("free all, nmalloc %d\n", GB_Global.nmalloc) ;
    GrB_finalize ( ) ;
}

