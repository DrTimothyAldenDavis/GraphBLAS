//------------------------------------------------------------------------------
// gbmex.h: definitions for MATLAB interface for GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"
#include "mex.h"

//------------------------------------------------------------------------------

#define ERROR(message) mexErrMsgIdAndTxt ("GraphBLAS:error", message) ;
#define USAGE(message) mexErrMsgIdAndTxt ("GraphBLAS:usage", message) ;
#define CHECK_ERROR(error,message) if (error) ERROR (message) ;
#define OK(method) CHECK_ERROR  ((method) != GrB_SUCCESS, GrB_error ( )) ;

// MATCH(s,t) compares two strings and returns true if equal
#define MATCH(s,t) (strcmp(s,t) == 0)

#define MAX(a,b) (((a) > (b)) ? (a) : (b))

// IS_SCALAR(X) is true if X is a MATLAB non-empty non-sparse numeric scalar
#define IS_SCALAR(X) (mxIsScalar (X) && mxIsNumeric (X) && !mxIsSparse (X))

//------------------------------------------------------------------------------

GrB_Type gb_mxarray_type        // return the GrB_Type of a MATLAB matrix
(
    const mxArray *X
) ;

GrB_Type gb_mxstring_to_type    // return the GrB_Type from a MATLAB string
(
    const mxArray *S        // MATLAB mxArray containing a string
) ;

int gb_mxstring_to_string   // returns length of string, or -1 if S not a string
(
    char *string,           // size maxlen
    const size_t maxlen,    // length of string
    const mxArray *S        // MATLAB mxArray containing a string
) ;

GrB_Matrix gb_get_shallow   // return a shallow copy of MATLAB sparse matrix
(
    const mxArray *X
) ;

void gb_free_shallow        // free a shallow GrB_Matrix
(
    GrB_Matrix *S_handle    // GrB_Matrix to free; set to NULL on output
) ;

GrB_Matrix gb_get_deep      // return a deep GrB_Matrix copy of a MATLAB X
(
    const mxArray *X,       // input MATLAB matrix (sparse or struct)
    GrB_Type type           // typecast X to this type (NULL if no typecast)
) ;

mxArray *gb_matrix_to_mxstruct  // return a MATLAB struct
(
    GrB_Matrix *A_Handle        // GrB_Matrix to convert to MATLAB struct
) ;

GrB_Type gb_type_to_mxstring    // return the MATLAB string from a GrB_Type
(
    const GrB_Type type
) ;

GrB_Matrix gb_typecast      // A = (type) S, where A is deep
(
    GrB_Type type,          // if NULL, copy but do not typecast
    GrB_Matrix S            // may be shallow
) ;

void gb_usage       // check usage and make sure GxB_init has been called
(
    bool ok,                // if false, then usage is not correct
    const char *message     // error message if usage is not correct
) ;

