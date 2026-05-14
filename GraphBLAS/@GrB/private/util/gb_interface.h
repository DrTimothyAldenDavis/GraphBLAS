//------------------------------------------------------------------------------
// gb_interface.h: the SuiteSparse:GraphBLAS MATLAB/Octave interface
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This interface depends heavily on internal details of the
// SuiteSparse:GraphBLAS library.  Thus, GB.h is #include'd (via GB_helper.h),
// not just GraphBLAS.h.

#ifndef GB_INTERFACE_H
#define GB_INTERFACE_H

#undef GRAPHBLAS_VANILLA
#include "GraphBLAS.h"
#include "GB_helper.h"
#include "mex.h"
#include <ctype.h>

//------------------------------------------------------------------------------
// error handling and test coverage
//------------------------------------------------------------------------------

#ifdef GBCOV

    //--------------------------------------------------------------------------
    // test coverage only, not used in production
    //--------------------------------------------------------------------------

    #define GBCOV_MAX 1000
    extern int64_t gbcov [GBCOV_MAX] ;
    extern int gbcov_max ;
    void gbcov_get (void) ;
    void gbcov_put (void) ;
    static inline void gb_wrapup (void)
    {
        gbcov_put ( ) ;
    }

#else

    //--------------------------------------------------------------------------
    // no test coverage in production
    //--------------------------------------------------------------------------

    #define gbcov_get()
    #define gbcov_put()
    #define gb_wrapup()

#endif

#ifndef FREE_WORK
#define FREE_WORK
#endif

#ifndef FREE_ALL
#define FREE_ALL FREE_WORK
#endif

//------------------------------------------------------------------------------
// basic error handling for mexFunctions and utilities
//------------------------------------------------------------------------------

#if defined ( GB_UTIL )

    // error handling for gb_* utilities
    #define ERROR2(errmsg,arg,info)                             \
    {                                                           \
        FREE_ALL ;                                              \
        mexPrintf ("File: %s, Line: %d\n", __FILE__, __LINE__) ; \
        mexPrintf ("GrB:error (%d): " errmsg "\n", info, arg) ; \
        return (info) ;                                         \
    }

    #define ERROR(errmsg,info)                                  \
    {                                                           \
        FREE_ALL ;                                              \
        mexPrintf ("File: %s, Line: %d\n", __FILE__, __LINE__) ; \
        mexPrintf ("GrB:error (%d): %s\n", info, errmsg) ;      \
        return (info) ;                                         \
    }

#else

    // error handling for mexFunctions and gbmx_* utilities
    #define ERROR2(errmsg,arg,info)                             \
    {                                                           \
        gbcov_put ( ) ;                                         \
        FREE_ALL ;                                              \
        mexPrintf ("File: %s, Line: %d\n", __FILE__, __LINE__) ; \
        mexErrMsgIdAndTxt ("GrB:error", errmsg, arg) ;          \
    }
    #define ERROR(errmsg,info)                                  \
    {                                                           \
        gbcov_put ( ) ;                                         \
        FREE_ALL ;                                              \
        mexPrintf ("File: %s, Line: %d\n", __FILE__, __LINE__) ; \
        mexErrMsgIdAndTxt ("GrB:error", errmsg) ;               \
    }

#endif

//------------------------------------------------------------------------------
// error handling for both mexFunctions and utilities
//------------------------------------------------------------------------------

#define CHECK_ERROR(error,errmsg)                           \
    if (error) ERROR (errmsg, GrB_INVALID_VALUE) ;

#define OK(method)                                          \
{                                                           \
    GrB_Info this_info = method ;                           \
    if (this_info != GrB_SUCCESS)                           \
    {                                                       \
        ERROR (gb_error_string (this_info), this_info) ;    \
    }                                                       \
}

#define OK0(method)                                                 \
{                                                                   \
    GrB_Info this_info = method ;                                   \
    if (!(this_info == GrB_SUCCESS || this_info == GrB_NO_VALUE))   \
    {                                                               \
        ERROR (gb_error_string (this_info), this_info) ;            \
    }                                                               \
}

#define OK1(C,method)                                               \
{                                                                   \
    GrB_Info this_info = method ;                                   \
    if (this_info != GrB_SUCCESS)                                   \
    {                                                               \
        const char *err1 = gb_error_string (this_info) ;            \
        mexPrintf ("%s\n", err1) ;                                  \
        const char *err2 ;                                          \
        GrB_Matrix_error (&err2, C) ;                               \
        ERROR ((err2 == NULL || err2 [0] == '\0') ? err1 : err2,    \
            this_info) ;                                            \
    }                                                               \
}

#define CHECK_NULL(p)                                               \
{                                                                   \
    if ((p) == NULL)                                                \
    {                                                               \
        ERROR ("out of memory", GrB_OUT_OF_MEMORY) ;                \
    }                                                               \
}

//------------------------------------------------------------------------------
// basic macros
//------------------------------------------------------------------------------

// MATCH(s,t) compares two strings and returns true if equal
#define MATCH(s,t) (strcmp(s,t) == 0)

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define ABS(x)   (((x) >= 0) ? (x) : (-(x)))

// largest integer representable as a double
#define FLINTMAX (((int64_t) 1) << 53)

// default maximum string length
#define LEN 256

//------------------------------------------------------------------------------
// typedefs
//------------------------------------------------------------------------------

typedef enum            // output of GrB.methods
{
    KIND_GRB = 0,       // return G.opaque containing a GrB_Matrix
    KIND_SPARSE = 1,    // return a built-in sparse matrix
    KIND_FULL = 2,      // return a built-in full matrix
    KIND_BUILTIN = 3    // return a built-in sparse or full matrix (full if all
                        // entries present, sparse otherwise)
}
kind_enum_t ;

// [I,J,X] = GrB.extracttuples (A, desc) can return I and J in three ways:
//
//      one-based double:   just like [I,J,X] = find (A)
//      one-based int64:    I and J are one-based, as built-in but int64.
//      zero-based int64:   I and J are zero-based, and int64.  This is meant
//                          for internal use in GrB methods, but it is also
//                          the
//
// The descriptor is also used for GrB.build, GrB.extract, GrB.assign, and
// GrB.subassign.  In that case, the type is determined by the input arrays I
// and J.
//
// desc.base can be one of several strings:
//
//      'default'           the default is used (one-based int)
//      'zero-based'        zero-based uint32/uint64
//      'zero-based int'    zero-based uint32/uint64
//      'one-based'         one-based uint32/uint64
//      'one-based int'     one-based uint32/uint64
//      'one-based double'  the type is double, and one-based
//      'double'            the type is double, and one-based
//
// Note that there is no option for zero-based double.

typedef enum            // type of indices
{
    BASE_DEFAULT = 0,   // one-based integers (int32/int64)
    BASE_0_INT = 1,     // indices are returned as zero-based int32/int64
    BASE_1_INT = 2,     // indices are returned as one-based int32/int64
    BASE_1_DOUBLE = 3   // one-based double, unless the dimensions are too big
                        // for a flint (max(size(A)) > flintmax).  In that
                        // case, BASE_1_INT is used.
}
base_enum_t ;

// gb_descriptor_struct: a plain struct, so that it can be statically allocated
// in the mx* portion of a mexFunction, and filled with values from the MATLAB
// desc struct.

struct gb_descriptor_struct
{
    int nondefault ;    // 0: all GrB_Descriptor options are default;
                        // so the GrB_Descriptor can be NULL
    int is_present ;    // 1: MATLAB descriptor struct is present on input;
                        // 0: not present

    // these appear in the GraphBLAS GrB_Descriptor:
    int out ;           // output descriptor
    int mask ;          // mask descriptor
    int in0 ;           // first input descriptor (A for C=A*B, for example)
    int in1 ;           // second input descriptor (B for C=A*B)
    int axb ;           // for selecting the method for C=A*B

    // these are only in the gb_descriptor:
    kind_enum_t kind ;  // how to return the output
    int fmt ;           // by row or by col
    int sparsity ;      // hypersparse/sparse/bitmap/full
    kind_enum_t base ;  // 0-based-int, 1-based int, or 1-based double

// these appear in the GraphBLAS descriptor but are not needed here:
//  int compression ;   // compression method for GxB_Matrix_serialize
//  int do_sort ;       // if nonzero, do the sort in GrB_mxm
//  int import ;        // if zero (default), trust input data
//  int row_list ;      // how to use the row index list, I
//  int col_list ;      // how to use the col index list, J
//  int val_list ;      // how to use the value list, X
} ;

typedef struct gb_descriptor_struct *gb_descriptor ;

// gb_matrix_struct: a plain struct that can be statically allocated, which
// holds either a GraphBLAS @GrB matrix or the contents of a MATLAB sparse or
// full matrix.

struct gb_matrix_struct
{
    //--------------------------------------------------------------------------
    // content for all matrices, GraphBLAS or MATLAB
    //--------------------------------------------------------------------------

    uint64_t nvals ;    // # of entries (for a GraphBLAS matrix, includes
                        // zombies but excludes pending tuples). 
    GrB_Type type ;     // type of the MATLAB matrix, as a GrB_Type
    uint64_t nrows ;    // from mxGetM
    uint64_t ncols ;    // from mxGetN
    size_t typesize ;   // size of the data type

    //--------------------------------------------------------------------------

    // Only one of the two sections are present.  This struct is memset to all
    // zero, and then only one of the two sections are filled.  If the matrix
    // is a GraphBLAS matrix, then G is non-NULL.  Otherwise, the matrix is a
    // built-in MATLAB sparse or full matrix.

    //--------------------------------------------------------------------------
    // for a GraphBLAS matrix; NULL if the matrix is a MATLAB matrix
    //--------------------------------------------------------------------------

    GrB_Matrix G ;

    //--------------------------------------------------------------------------
    // for a MATLAB matrix: populated if G is non-NULL
    //--------------------------------------------------------------------------

    // If the input is a 0-by-0 MATLAB matrix, the [p,i,x] content below is
    // NULL, and is_sparse is false.

    uint64_t *p ;       // from mxGetJc, NULL if the matrix is full
    uint64_t *i ;       // from mxGetIr, NULL if the matrix is full
    uint64_t *x ;       // from mxGetData

    //--------------------------------------------------------------------------
    // bool content for a MATLAB matrix
    //--------------------------------------------------------------------------

    bool is_sparse ;    // from mxIsSparse, for a MATLAB matrix; ignored for
                        // a GraphBLAS matrix
    bool is_empty ;     // true for an empty MATLAB matrix

    //--------------------------------------------------------------------------
    // bool content for a GraphBLAS matrix
    //--------------------------------------------------------------------------

    bool will_wait ;    // true if G has any pending work; always false for a
                        // MATLAB matrix

} ;

typedef struct gb_matrix_struct *gb_matrix ;

//------------------------------------------------------------------------------
// function prototypes
//------------------------------------------------------------------------------

void gb_at_exit ( void ) ;  // call GrB_finalize

GrB_Info gb_binaryop_ztype
(
    // output
    GrB_Type *ztype,    // the GrB_Type of the output of a binary op
    // input
    GrB_BinaryOp op
) ;

GrB_Info gb_binop_to_monoid         // return monoid from a binary op
(
    // output
    GrB_Monoid *monoid,
    // input
    GrB_BinaryOp op
) ;

GrB_Info gb_by_col
(
    // output
    GrB_Matrix *A_handle,       // return the matrix by column
    GrB_Matrix *A_copy_handle,  // copy made of A, stored by column, or NULL
    // input
    GrB_Matrix A_input          // input matrix, by row or column
) ;

GrB_Info gb_cell_to_list
(
    // output
    GrB_Vector *I_handle,
    GrB_Vector *I_to_free_handle,
    uint64_t *nI,               // # of items in the list
    int64_t *I_max,             // largest item in the list (NULL if not needed)
    // input
    struct gb_matrix_struct Cell_Matrix [3],    // contents of the Cell
    const int len,              // # of items in Cell_Matrix
    const int base_offset,      // 1 or 0
    const uint64_t n            // dimension of the matrix
) ;

GrB_Type gb_code_to_type    // return the GrB_Type from a GrB_Type_Code
(
    GrB_Type_Code code
) ;

GrB_Info gb_default_format
(
    // output
    int *fmt,               // GxB_BY_ROW or GxB_BY_COL
    // input
    uint64_t nrows,        // row vectors are stored by row
    uint64_t ncols         // column vectors are stored by column
) ;

GrB_Info gb_defaults (void) ;   // set global GraphBLAS defaults for MATLAB

GrB_Type gb_default_type        // return the default type to use
(
    const GrB_Type atype,       // type of the A matrix
    const GrB_Type btype        // type of the B matrix
) ;

const char *gb_error_string     // return an error string from a GrB_Info value
(
    GrB_Info info
) ;

GrB_Info gb_expand_scalar_to_vector
(
    GrB_Vector *V,
    GrB_Type type,
    uint64_t nvals
) ;

GrB_Info gb_expand_to_full      // C = full (A), and typecast
(
    // output
    GrB_Matrix *C_handle,
    // inputs
    const GrB_Matrix A,         // input matrix to expand to full
    GrB_Type type,              // type of C, if NULL use the type of A
    int fmt,                    // format of C
    GrB_Matrix id               // identity value, use zero if NULL
) ;

GrB_Info gb_export              // export a GrB_Matrix to MATLAB
(
    // output:
    GrB_Matrix *C_opaque,
    // input/output:
    GrB_Matrix *C_handle,       // GrB_Matrix to export, set to NULL on output
    // input:
    kind_enum_t kind            // GrB, sparse, full, or built-in
) ;

GrB_Info gb_export_to_full
(
    GrB_Matrix *C_handle    // GraphBLAS matrix to modify for export to MATLAB
) ;

GrB_Info gb_export_to_sparse
(
    // input/output
    GrB_Matrix *C_handle    // GraphBLAS matrix to modify for export to MATLAB
) ;

void gb_find_dot            // find 1st and 2nd dot ('.') in a string
(
    int32_t position [2],   // positions of one or two dots
    const char *s           // null-terminated string to search
) ;

GrB_Info gb_first_binop     // construct GrB_FIRST_[type] operator
(
    // output
    GrB_BinaryOp *op,       // return GrB_FIRST_[type] operator
    // input
    const GrB_Type type
) ;

GrB_Info gb_get_deep        // get a deep GrB_Matrix copy of a matrix
(
    // output:
    GrB_Matrix *C_handle,   // deep copy of the input matrix
    GrB_Matrix *C_shallow,  // shallow version; must be freed by caller
    // input:
    gb_matrix X             // input MATLAB or @GrB matrix
) ;

GrB_Info gb_get_descriptor
(
    // output:
    GrB_Descriptor *desc_handle,    // GraphBLAS descriptor
    // input:
    gb_descriptor gbdesc            // gb_descriptor, pointer to static struct
) ;

GrB_Info gb_get_descriptor_mxm
(
    // output:
    GrB_Descriptor *desc_handle,    // GraphBLAS descriptor
    // input:
    gb_descriptor gbdesc            // gb_descriptor, pointer to static struct
) ;

GrB_Info gb_get_first_scalar
(
    GrB_Scalar *x,          // x = find (V, 'first')
    GrB_Vector V,
    GrB_Type type
) ;

GrB_Info gb_get_format      // get the format (by row or by col)
(
    // input:
    GrB_Index cnrows,       // C is cnrows-by-cncols
    GrB_Index cncols,
    GrB_Matrix A,           // may be NULL
    GrB_Matrix B,           // may be NULL
    // input/output:
    int *fmt                // may be GxB_NO_FORMAT on input
) ;

GrB_Info gb_get_matlab_matrix    // shallow copy of MATLAB sparse matrix
(
    // output
    GrB_Matrix *A_handle,   // content of A is tagged GxB_IS_READONLY
    // input
    gb_matrix matrix        // contents of a MATLAB matrix
) ;

GrB_Info gb_get_matrix      // shallow copy of MATLAB sparse matrix,
                            // or the content of a MATLAB @GrB handle object
(
    // output
    GrB_Matrix *A_handle,   // output matrix
    GrB_Matrix *A_shallow,  // must be freed by the caller if not NULL
    // input
    gb_matrix X             // input MATLAB or @GrB matrix
) ;

GrB_Info gb_get_sparsity    // determine the sparsity of C for C = method(A,B)
(
    // input:
    GrB_Matrix A,           // may be NULL
    GrB_Matrix B,           // may be NULL
    // input/output:
    int *sparsity           // may be 0 on input
) ;

GrB_Info gb_is_all          // check two matrices for equality, given an op
(
    // output:
    bool *result,           // true if op (A,B) is all true, false otherwise
    // input:
    GrB_Matrix A,
    GrB_Matrix B,
    GrB_BinaryOp op
) ;

GrB_Info gb_is_column_vector    // determine if A is a column vector
(
    // output:
    bool *is_column_vector,
    // input:
    GrB_Matrix A                // GrB_matrix to query
) ;

GrB_Info gb_is_dense            // determine if A is dense
(
    // output:
    bool *is_dense,
    // input:
    GrB_Matrix A                // GrB_Matrix to query
) ;

GrB_Info gb_is_equal
(
    // output:
    bool *is_equal,             // true if A == B, false if A ~= B
    // input:
    GrB_Matrix A,
    GrB_Matrix B
) ;

bool gb_is_float (const GrB_Type type) ;

bool gb_is_integer (const GrB_Type type) ;

GrB_Info gb_is_scalar
(
    // output:
    bool *is_scalar,    // true if A is a 1-by-1 GrB_Matrix with 1 entry
    // input
    GrB_Matrix A
) ;

GrB_Info gb_is_vector
(
    bool *is_vector,            // true if A is a row or column vector
    GrB_Matrix A                // GrB_Matrix to query
) ;

GrB_Info gb_matrix_to_list
(
    // outputs:
    GrB_Vector *V_handle,   // list of indices or values; caller must not free
    GrB_Vector *V_to_free_handle,  // must be freed by the caller
    // inputs:
    gb_matrix matrix,
    const int base_offset   // 1 or 0
) ;

GrB_Info gb_monoid_type
(
    // output:
    GrB_Type *type,
    // input:
    GrB_Monoid op
) ;

GrB_Info gb_new       // create and empty matrix C
(
    // output
    GrB_Matrix *C_handle,
    // input
    GrB_Type type,      // type of C
    uint64_t nrows,     // # of rows
    uint64_t ncols,     // # of rows
    int fmt,            // requested format, if < 0 use default
    int sparsity        // sparsity control for C, 0 for default
) ;

GrB_Info gb_norm            // compute norm (A,kind)
(
    // output:
    double *s,              // norm of A
    // inputs:
    GrB_Matrix A,
    int64_t norm_kind       // 0, 1, 2, INT64_MAX, or INT64_MIN
) ;

GrB_UnaryOp gb_round_op (const GrB_Type type) ;

GrB_Info gb_semiring                // find semiring from (add,mult) ops
(
    // output:
    GrB_Semiring *semiring,
    // inputs:
    const GrB_BinaryOp add,         // add operator
    const GrB_BinaryOp mult         // multiply operator
) ;

GrB_Info gb_string_and_type_to_binop_or_idxunop
(
    // output:
    GrB_BinaryOp *binop,        // binary op, or NULL if idxunop
    // input:
    const char *op_name,        // name of the operator, as a string
    const GrB_Type type,        // type of the x,y inputs to the operator
    const bool type_not_given,  // true if no type present in the string
    // output:
    GrB_IndexUnaryOp *idxunop,          // idxunop from the string
    // input/output:
    int64_t *ithunk                     // thunk for idxunop
) ;

GrB_Info gb_string_and_type_to_unop  // return op from string and type
(
    // output
    GrB_UnaryOp *unop,
    // input
    const char *op_name,        // name of the operator, as a string
    const GrB_Type type,        // type of the input to the operator
    const bool type_not_given   // true if no type present in the string
) ;

GrB_Info gb_string_to_binop // return binary operator from a string
(
    // output
    GrB_BinaryOp *binop,        // binary op determined from the string
    // input/output:
    char *opstring,             // string that defines the binary operator
    // input:
    const GrB_Type atype,       // type of A
    const GrB_Type btype        // type of B
) ;

GrB_Info gb_string_to_binop_or_idxunop
(
    // output:
    GrB_BinaryOp *binop,        // binary op, or NULL if idxunop
    // input/output:
    char *opstring,                     // string defining the operator
    // input:
    const GrB_Type atype,               // type of A
    const GrB_Type btype,               // type of B
    // output
    GrB_IndexUnaryOp *idxunop,          // idxunop from the string
    // input/output
    int64_t *ithunk                     // thunk for idxunop
) ;

bool gb_string_to_format        // true if a valid format is found
(
    // input/output:
    char *format_string,
    // output:
    int *fmt,
    int *sparsity
) ;

GrB_Info gb_string_to_idxunop
(
    // outputs: one of the outputs is non-NULL and the other NULL
    GrB_IndexUnaryOp *op,       // GrB_IndexUnaryOp, if found
    bool *thunk_zero,           // true if op requires a thunk zero
    bool *op_is_positional,     // true if op is positional
    // input/output:
    int64_t *ithunk,
    // inputs:
    char *opstring,             // string defining the operator
    const GrB_Type atype        // type of A, or NULL if not present
) ;

GrB_Info gb_string_to_monoid            // return monoid from a string
(
    // output
    GrB_Monoid *monoid,
    // input
    char *opstring,                     // string defining the operator
    const GrB_Type type                 // default type if not in the string
) ;

GrB_Info gb_string_to_semiring          // return a GrB semiring from a string
(
    // output:
    GrB_Semiring *semiring,
    // input/output:
    char *semiring_string,              // string defining the semiring
    // inputs:
    const GrB_Type atype,               // type of A
    const GrB_Type btype                // type of B
) ;

GrB_Type gb_string_to_type      // return the GrB_Type from a string
(
    const char *classname
) ;

GrB_Info gb_string_to_unop              // return unary operator from a string
(
    // output
    GrB_UnaryOp *unop,                  // unary op determined by the string
    // input
    char *opstring,                     // string defining the operator
    const GrB_Type default_type         // default type if not in the string
) ;

GrB_Info gb_typecast  // C = (type) A, where C is deep
(
    // output:
    GrB_Matrix *C_handle,
    // inputs:
    GrB_Matrix A,       // may be shallow
    GrB_Type type,      // if NULL, use the type of A
    int fmt,            // format of C
    int sparsity        // sparsity control for C, if 0 use A
) ;

// allocate/free memory space in the default arena 0:
void *gb_malloc (size_t n) ;
void gb_free (void **p) ;

//------------------------------------------------------------------------------
// mx-based utilties
//------------------------------------------------------------------------------

// These methods can use mxMalloc and mexErrMsgIdAndTxt.  They cannot use any
// GraphBLAS methods (or if they do, those methods should not allocate any
// memory in the default malloc/free arena).  These methods do not return an
// error if mxMalloc fails or if mexErrMsgIdAndTxt is called.  Instead, control
// is returned directly to MATLAB.

void gbmx_abort ( void ) ;  // terminate immediately (debug assertions only)

mxArray *gbmx_export_struct ( GrB_Matrix **C_opaque_handle ) ;

void gbmx_free              // mxFree wrapper
(
    void **p_handle         // handle to pointer to be freed
) ;

int gbmx_flush ( void ) ;       // flush mexPrintf output to Command Window

GrB_Matrix gbmx_get_grb_matrix  // the content of a MATLAB @GrB handle object
(
    // input
    const mxArray *G            // must be a @GrB object
) ;

int64_t gbmx_get_int64_scalar   // return int64 value of a MATLAB scalar
(
    const mxArray *mxscalar,    // MATLAB scalar to extract
    char *name                  // name of the scalar
) ;

uint64_t *gbmx_get_integer_list (const mxArray *mxList, uint64_t *len) ;

kind_enum_t gbmx_get_kind (const mxArray *mxdesc) ;

void gbmx_get_matrix
(
    // output
    gb_matrix matrix,       // either a GraphBLAS or MATLAB matrix, statically
                            // allocated (but undefined) on input
    // input
    const mxArray *X        // @GrB object or MATLAB matrix
) ;

void gbmx_get_mxargs
(
    // input:
    int nargin,                 // # inputs for mexFunction (may be zero)
    const mxArray *pargin [ ],  // input arguments for mexFunction
    const char *usage,          // usage to print, if too many args appear
    // output:
    struct gb_matrix_struct Matrix [6], // matrix arguments
    int *nmatrices,             // # of matrix arguments
    char String [2][LEN+2],     // string arguments
    int *nstrings,              // # of string arguments
    mxArray *Cell [2],          // cell array arguments
    int *ncells,                // # of cell array arguments
    gb_descriptor gbdesc        // gb_descriptor struct
) ;

uint64_t gbmx_get_uint64_scalar // return uint64 value of a MATLAB scalar
(
    const mxArray *mxscalar,    // MATLAB scalar to extract
    char *name                  // name of the scalar
) ;

bool gbmx_mxarray_is_scalar     // true if built-in array is a scalar
(
    const mxArray *S
) ;

bool gbmx_mxarray_to_descriptor // true if descriptor present in pargin [...]
(
    // output:
    gb_descriptor gbdesc,   // statically allocated on input
    // input:
    const mxArray *mxdesc   // MATLAB struct with possible descriptor
) ;

GrB_Type gbmx_mxarray_type      // return the GrB_Type of a built-in matrix
(
    const mxArray *X
) ;

void gbmx_mxcell_to_matrices
(
    // output
    struct gb_matrix_struct Cell_Matrix [3], // matrix contents of the Cell
    int *len,                   // # of items in the Cell
    // input
    const mxArray *Cell         // built-in MATLAB cell array (at most 3 items)
) ;

mxArray *gbmx_mxclass_to_mxstring (mxClassID class, bool is_complex) ;

void gbmx_mxstring_to_string  // copy a built-in string into a C string
(
    // output:
    char *string,           // size at least maxlen+1
    // input:
    const size_t maxlen,    // length of string
    const mxArray *S,       // built-in mxArray containing a string
    const char *name        // name of the mxArray
) ;

mxArray *gbmx_new_matlab_matrix // return new MATLAB full matrix
(
    const uint64_t nrows,       // dimensions
    const uint64_t ncols,
    GrB_Type type               // type of the array
) ;

int64_t gbmx_norm_kind (const mxArray *arg) ;

void gbmx_set_double_scalar (mxArray *scalar, double value) ;

mxArray * gbmx_type_to_mxstring // return the built-in string from a GrB_Type
(
    const GrB_Type type
) ;

void gbmx_usage       // check usage and make sure GxB_init has been called
(
    bool ok,                // if false, then usage is not correct
    const char *message     // error message if usage is not correct
) ;

//------------------------------------------------------------------------------
// mexFunctions in the util folder 
//------------------------------------------------------------------------------

void gbmx_assign_mexFunction    // gbassign or gbsubassign mexFunctions
(
    int nargout,                // # output arguments for mexFunction
    mxArray *pargout [ ],       // output arguments for mexFunction
    int nargin,                 // # input arguments for mexFunction
    const mxArray *pargin [ ],  // input arguments for mexFunction
    bool do_subassign,          // true: do subassign, false: do assign
    const char *usage           // usage string to print if error
) ;

//------------------------------------------------------------------------------
// remove access to GraphBLAS polymorphic methods
//------------------------------------------------------------------------------

// The @GrB MATLAB interface does not use these macros since they require a
// C11 compiler, and thus they cannot be used for MATLAB on Windows.

#undef GrB_Monoid_new
#undef GxB_Monoid_terminal_new
#undef GrB_Scalar_setElement
#undef GrB_Scalar_extractElement
#undef GrB_Vector_build
#undef GrB_Vector_setElement
#undef GrB_Vector_extractElement
#undef GrB_Vector_extractTuples
#undef GrB_Matrix_build
#undef GrB_Matrix_setElement
#undef GrB_Matrix_extractElement
#undef GrB_Matrix_extractTuples
#undef GrB_get
#undef GrB_set
#undef GrB_wait
#undef GrB_error
#undef GrB_eWiseMult
#undef GrB_eWiseAdd
#undef GxB_eWiseUnion
#undef GrB_extract
#undef GxB_subassign
#undef GrB_assign
#undef GrB_apply
#undef GrB_select
#undef GrB_reduce
#undef GrB_kronecker
#undef GxB_resize
#undef GxB_fprint
#undef GxB_print
#undef GrB_Matrix_import
#undef GrB_Matrix_export
#undef GxB_sort
#undef GrB_free
#undef GxB_Scalar_setElement
#undef GxB_Scalar_extractElement
#undef GxB_set
#undef GxB_get
#undef GxB_select

#endif

