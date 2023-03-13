//------------------------------------------------------------------------------
// GB_stringify.h: prototype definitions construction of *.h definitions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_STRINGIFY_H
#define GB_STRINGIFY_H

//------------------------------------------------------------------------------
// dump definitions (for debugging and test coverage only)
//------------------------------------------------------------------------------

// uncomment this line to dump GB*.h files to /tmp, or compile with
// -DGB_DEBUGIFY_DEFN=1
// #undef  GB_DEBUGIFY_DEFN
// FIXME: debugify is on
#define GB_DEBUGIFY_DEFN 1

//------------------------------------------------------------------------------
// determine if the JIT is enabled at compile-time
//------------------------------------------------------------------------------

// FIXME: allow cmake to control this option.  Remove GB_DEBUGIFY_DEFN.
// Fix GBRENAME case and get the JIT working in MATLAB.

#if defined ( GB_DEBUGIFY_DEFN ) && !defined ( GBRENAME )
#define GB_JIT_ENABLED 1
#else
#define GB_JIT_ENABLED 0
#endif

//------------------------------------------------------------------------------
// print copyright and license
//------------------------------------------------------------------------------

void GB_macrofy_copyright (FILE *fp) ;

//------------------------------------------------------------------------------
// for GB_boolean_rename and related methods
//------------------------------------------------------------------------------

#include "GB_binop.h"
#include "GB_jitifyer.h"

//------------------------------------------------------------------------------
// left and right shift
//------------------------------------------------------------------------------

#define GB_LSHIFT(x,k) (((uint64_t) x) << k)
#define GB_RSHIFT(x,k,b) ((x >> k) & ((((uint64_t)0x00000001) << b) -1))

//------------------------------------------------------------------------------
// GrB_reduce
//------------------------------------------------------------------------------

uint64_t GB_encodify_reduce // encode a GrB_reduce problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    GrB_Monoid monoid,      // the monoid to enumify
    GrB_Matrix A            // input matrix to reduce
) ;

void GB_enumify_reduce      // enumerate a GrB_reduce problem
(
    // output:
    uint64_t *rcode,        // unique encoding of the entire problem
    // input:
    GrB_Monoid monoid,      // the monoid to enumify
    GrB_Matrix A            // input matrix to monoid
) ;

void GB_macrofy_reduce      // construct all macros for GrB_reduce to scalar
(
    FILE *fp,               // target file to write, already open
    // input:
    uint64_t rcode,         // encoded problem
    GrB_Monoid monoid,      // monoid to macrofy
    GrB_Type atype          // type of the A matrix to reduce
) ;

//------------------------------------------------------------------------------
// GrB_eWiseAdd, GrB_eWiseMult, GxB_eWiseUnion
//------------------------------------------------------------------------------

// FUTURE: add accumulator for eWise operations?

uint64_t GB_encodify_ewise      // encode an ewise problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const int kcode,            // kernel to encode (add, emult, rowscale, ...)
    const bool is_eWiseMult,    // if true, method is emult
    const bool is_eWiseUnion,   // if true, method is eWiseUnion
    const bool can_copy_to_C,   // if true C(i,j)=A(i,j) can bypass the op
    const bool C_iso,
    const bool C_in_iso,
    const int C_sparsity,
    const GrB_Type ctype,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_BinaryOp binaryop,
    const bool flipxy,
    const GrB_Matrix A,
    const GrB_Matrix B
) ;

bool GB_enumify_ewise       // enumerate a GrB_eWise problem
(
    // output:
    uint64_t *scode,        // unique encoding of the entire operation
    // input:
    bool is_eWiseMult,      // if true, method is emult
    bool is_eWiseUnion,     // if true, method is eWiseUnion
    bool can_copy_to_C,     // if true C(i,j)=A(i,j) can bypass the op
    // C matrix:
    bool C_iso,             // if true, C is iso on output
    bool C_in_iso,          // if true, C is iso on input
    int C_sparsity,         // sparse, hyper, bitmap, or full
    GrB_Type ctype,         // C=((ctype) T) is the final typecast
    // M matrix:
    GrB_Matrix M,           // may be NULL
    bool Mask_struct,       // mask is structural
    bool Mask_comp,         // mask is complemented
    // operator:
    GrB_BinaryOp binaryop,  // the binary operator to enumify
    bool flipxy,            // multiplier is: op(a,b) or op(b,a)
    // A and B:
    GrB_Matrix A,           // NULL for unary apply with binop, bind 1st
    GrB_Matrix B            // NULL for unary apply with binop, bind 2nd
) ;

void GB_macrofy_ewise           // construct all macros for GrB_eWise
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    GrB_BinaryOp binaryop,      // binaryop to macrofy
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype
) ;

//------------------------------------------------------------------------------
// GrB_mxm
//------------------------------------------------------------------------------

// FUTURE: add accumulator for mxm?

uint64_t GB_encodify_mxm        // encode a GrB_mxm problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const int kcode,            // kernel to encode (dot3, saxpy3, etc)
    const bool C_iso,
    const bool C_in_iso,
    const int C_sparsity,
    const GrB_Type ctype,
    const GrB_Matrix M,
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_Semiring semiring,
    const bool flipxy,
    const GrB_Matrix A,
    const GrB_Matrix B
) ;

void GB_enumify_mxm         // enumerate a GrB_mxm problem
(
    // output:              // future: may need to become 2 x uint64
    uint64_t *scode,        // unique encoding of the entire semiring
    // input:
    // C matrix:
    bool C_iso,             // C output iso: if true, semiring is ANY_PAIR_BOOL
    bool C_in_iso,          // C input iso status
    int C_sparsity,         // sparse, hyper, bitmap, or full
    GrB_Type ctype,         // C=((ctype) T) is the final typecast
    // M matrix:
    GrB_Matrix M,           // may be NULL
    bool Mask_struct,       // mask is structural
    bool Mask_comp,         // mask is complemented
    // semiring:
    GrB_Semiring semiring,  // the semiring to enumify
    bool flipxy,            // multiplier is: mult(a,b) or mult(b,a)
    // A and B:
    GrB_Matrix A,
    GrB_Matrix B
) ;

void GB_macrofy_mxm        // construct all macros for GrB_mxm
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    GrB_Semiring semiring,  // the semiring to macrofy
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype
) ;

//------------------------------------------------------------------------------
// enumify and macrofy the mask matrix M
//------------------------------------------------------------------------------

void GB_enumify_mask       // return enum to define mask macros
(
    // output:
    int *mask_ecode,            // enumified mask
    // input
    const GB_Type_code mcode,   // typecode of the mask matrix M,
                                // or 0 if M is not present
    bool Mask_struct,           // true if M structural, false if valued
    bool Mask_comp              // true if M complemented
) ;

void GB_macrofy_mask
(
    FILE *fp,               // file to write macros, assumed open already
    // input:
    int mask_ecode,         // enumified mask
    char *Mname,            // name of the mask
    int msparsity           // sparsity of the mask
) ;

//------------------------------------------------------------------------------
// enumify and macrofy a monoid
//------------------------------------------------------------------------------

void GB_enumify_monoid  // enumerate a monoid
(
    // outputs:
    int *add_ecode,     // binary op as an enum
    int *id_ecode,      // identity value as an enum
    int *term_ecode,    // terminal value as an enum
    // inputs:
    int add_opcode,     // must be a built-in binary operator from a monoid
    int zcode           // type of the monoid (x, y, and z)
) ;

void GB_macrofy_monoid  // construct the macros for a monoid
(
    FILE *fp,           // File to write macros, assumed open already
    // inputs:
    int add_ecode,      // binary op as an enum
    int id_ecode,       // identity value as an enum
    int term_ecode,     // terminal value as an enum (<= 28 is terminal)
    bool C_iso,         // true if C is iso
    GrB_Monoid monoid,  // monoid to macrofy
    bool disable_terminal_condition,    // if true, the monoid is assumed
                        // to be non-terminal.  For the (times, firstj, int64)
                        // semiring, times is normally a terminal monoid, but
                        // it's not worth exploiting in GrB_mxm.
    // output:
    const char **u_expression
) ;

void GB_macrofy_query_monoid
(
    FILE *fp,
    GrB_Monoid monoid
) ;

//------------------------------------------------------------------------------
// binary operators
//------------------------------------------------------------------------------

void GB_enumify_binop
(
    // output:
    int *ecode,         // enumerated operator, range 0 to 110; -1 on failure
    // input:
    GB_Opcode opcode,   // opcode of GraphBLAS operator to convert into a macro
    GB_Type_code zcode, // op->xtype->code of the operator
    bool for_semiring   // true for A*B, false for A+B or A.*B
) ;

void GB_macrofy_binop
(
    FILE *fp,
    // input:
    const char *macro_name,
    bool flipxy,                // if true: op is f(y,x)
    bool is_monoid_or_build,    // if true: additive operator for monoid,
                                // or binary op for GrB_Matrix_build
    bool is_ewise,              // if true: binop for ewise methods
    int ecode,
    bool C_iso,                 // if true: C is iso
    GrB_BinaryOp op,            // NULL if C is iso
    // output:
    const char **f_handle,
    const char **u_handle
) ;

//------------------------------------------------------------------------------
// operator definitions and typecasting
//------------------------------------------------------------------------------

bool GB_macrofy_defn    // return true if user-defined operator is a macro
(
    FILE *fp,
    int kind,           // 0: built-in function
                        // 1: built-in macro
                        // 2: built-in macro needed for CUDA only
                        // 3: user-defined function or macro
    const char *name,
    const char *defn
) ;

void GB_macrofy_string
(
    FILE *fp,
    const char *name,
    const char *defn
) ;

const char *GB_macrofy_cast_expression  // return cast expression
(
    FILE *fp,
    // input:
    GrB_Type ztype,     // output type
    GrB_Type xtype,     // input type
    // output
    int *nargs          // # of string arguments in output format
) ;

void GB_macrofy_cast_input
(
    FILE *fp,
    // input:
    const char *macro_name,     // name of the macro: #define macro(z,x...)
    const char *zarg,           // name of the z argument of the macro
    const char *xargs,          // one or more x arguments
    const char *xexpr,          // an expression based on xargs
    const GrB_Type ztype,       // the type of the z output
    const GrB_Type xtype        // the type of the x input
) ;

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
) ;

void GB_macrofy_cast_copy
(
    FILE *fp,
    // input:
    const char *cname,          // name of the C matrix (typically "C")
    const char *aname,          // name of the A matrix (typically "A" or "B")
    const GrB_Type ctype,       // the type of the C matrix
    const GrB_Type atype,       // the type of the A matrix
    const bool A_iso            // true if A is iso
) ;

void GB_macrofy_input
(
    FILE *fp,
    // input:
    const char *aname,      // name of the scalar aij = ...
    const char *Amacro,     // name of the macro is GB_GET*(Amacro)
    const char *Aname,      // name of the input matrix
    bool do_matrix_macros,  // if true, do the matrix macros
    GrB_Type xtype,         // type of aij
    GrB_Type atype,         // type of the input matrix
    int asparsity,          // sparsity format of the input matrix
    int acode,              // type code of the input (0 if pattern)
    int A_iso_code,         // 1 if A is iso
    int azombies            // 1 if A has zombies, 0 if A has no zombies,
                            // -1 if A can never have zombies
) ;

void GB_macrofy_output
(
    FILE *fp,
    // input:
    const char *cname,      // name of the scalar ... = cij to write
    const char *Cmacro,     // name of the macro is GB_PUT*(Cmacro)
    const char *Cname,      // name of the output matrix
    GrB_Type ctype,         // type of C, ignored if C is iso
    GrB_Type ztype,         // type of cij scalar to cast to ctype write to C
    int csparsity,          // sparsity format of the output matrix
    bool C_iso,             // true if C is iso on output
    bool C_in_iso           // true if C is iso on input
) ;

//------------------------------------------------------------------------------
// monoid identity and terminal values
//------------------------------------------------------------------------------

void GB_enumify_identity       // return enum of identity value
(
    // output:
    int *ecode,             // enumerated identity, 0 to 17 (-1 if fail)
    // input:
    GB_Opcode opcode,       // built-in binary opcode of a monoid
    GB_Type_code zcode      // type code used in the opcode we want
) ;

const char *GB_macrofy_id // return string encoding the value
(
    // input:
    int ecode,          // enumerated identity/terminal value
    size_t zsize,       // size of value
    // output:          // (optional: either may be NULL)
    bool *has_byte,     // true if value is a single repeated byte
    uint8_t *byte       // repeated byte
) ;

void GB_macrofy_bytes
(
    FILE *fp,               // file to write macros, assumed open already
    // input:
    const char *Name,       // all-upper-case name
    const char *variable,   // variable to declaer
    const char *type_name,  // name of the type
    const uint8_t *value,   // array of size nbytes
    size_t nbytes,
    bool is_identity        // true for the identity value
) ;

void GB_enumify_terminal       // return enum of terminal value
(
    // output:
    int *ecode,                 // enumerated terminal, 0 to 31 (-1 if fail)
    // input:
    GB_Opcode opcode,           // built-in binary opcode of a monoid
    GB_Type_code zcode          // type code used in the opcode we want
) ;

//------------------------------------------------------------------------------
// sparsity structure
//------------------------------------------------------------------------------

void GB_enumify_sparsity    // enumerate the sparsity structure of a matrix
(
    // output:
    int *ecode,             // enumerated sparsity structure:
                            // 0:hyper, 1:sparse, 2:bitmap, 3:full
    // input:
    int sparsity            // 0:no matrix, 1:GxB_HYPERSPARSE, 2:GxB_SPARSE,
                            // 4:GxB_BITMAP, 8:GxB_FULL
) ;

void GB_macrofy_sparsity    // construct macros for sparsity structure
(
    // input:
    FILE *fp,
    const char *matrix_name,    // "C", "M", "A", or "B"
    int sparsity
) ;

//------------------------------------------------------------------------------
// typedefs, type name and size
//------------------------------------------------------------------------------

void GB_macrofy_typedefs
(
    FILE *fp,
    // input:
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype,
    GrB_Type xtype,
    GrB_Type ytype,
    GrB_Type ztype
#if 0
    const char *ctype_defn,
    const char *atype_defn,
    const char *btype_defn,
    const char *xtype_defn,
    const char *ytype_defn,
    const char *ztype_defn
#endif
) ;

void GB_macrofy_type
(
    FILE *fp,
    // input:
    const char *what,       // typically X, Y, Z, A, B, or C
    const char *what2,      // typically "_" or "2"
    const char *name        // name of the type
) ;

void GB_macrofy_query_defn
(
    FILE *fp,
    GB_Operator op0,    // monoid op, select op, unary op, etc
    GB_Operator op1,    // binaryop for a semring
    GrB_Type type0,
    GrB_Type type1,
    GrB_Type type2
) ;

void GB_macrofy_query_version
(
    FILE *fp
) ;

//------------------------------------------------------------------------------
// unary ops
//------------------------------------------------------------------------------

bool GB_enumify_apply       // enumerate an apply or tranpose/apply problem
(
    // output:
    uint64_t *scode,        // unique encoding of the entire operation
    // input:
    // C matrix:
    int C_sparsity,         // sparse, hyper, bitmap, or full.  For apply
                            // without transpose, Cx = op(A) is computed where
                            // Cx is just C->x, so the caller uses 'full' when
                            // C is sparse, hyper, or full.
    bool C_is_matrix,       // true for C=op(A), false for Cx=op(A)
    GrB_Type ctype,         // C=((ctype) T) is the final typecast
    // operator:
        const GB_Operator op,       // unary/index-unary to apply; not binaryop
        bool flipij,                // if true, flip i,j for user idxunop
    // A matrix:
    const GrB_Matrix A              // input matrix
) ;

void GB_enumify_unop    // enumify a GrB_UnaryOp or GrB_IndexUnaryOp
(
    // output:
    int *ecode,         // enumerated operator, range 0 to 255
    bool *depends_on_x, // true if the op depends on x
    bool *depends_on_i, // true if the op depends on i
    bool *depends_on_j, // true if the op depends on j
    bool *depends_on_y, // true if the op depends on y
    // input:
    bool flipij,        // if true, then the i and j indices are flipped
    GB_Opcode opcode,   // opcode of GraphBLAS operator to convert into a macro
    GB_Type_code xcode  // op->xtype->code of the operator
) ;

void GB_macrofy_unop
(
    FILE *fp,
    // input:
    const char *macro_name,
    bool flipij,                // if true: op is f(z,x,j,i,y) with ij flipped
    int ecode,
    GB_Operator op              // GrB_UnaryOp or GrB_IndexUnaryOp
) ;

void GB_macrofy_apply           // construct all macros for GrB_apply
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    // operator:
        const GB_Operator op,       // unary/index-unary to apply; not binaryop
    GrB_Type ctype,
    GrB_Type atype
) ;

uint64_t GB_encodify_apply      // encode an apply problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const int kcode,            // kernel to encode
    const int C_sparsity,
    const bool C_is_matrix,     // true for C=op(A), false for Cx=op(A)
    const GrB_Type ctype,
    const GB_Operator op,
    const bool flipij,
    const GrB_Matrix A
) ;

//------------------------------------------------------------------------------
// builder kernel
//------------------------------------------------------------------------------

uint64_t GB_encodify_build      // encode an build problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const int kcode,            // kernel to encode
    const GrB_Type ttype,       // type of Tx array
    const GrB_Type stype,       // type of Sx array
    const GrB_BinaryOp dup      // operator for summing up duplicates
) ;

bool GB_enumify_build       // enumerate a GB_build problem
(
    // output:
    uint64_t *build_code,   // unique encoding of the entire operation
    // input:
    GrB_Type ttype,         // type of Tx
    GrB_Type stype,         // type of Sx
    // operator:
    GrB_BinaryOp dup        // operator for duplicates
) ;

void GB_macrofy_build           // construct all macros for GB_build
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t build_code,        // unique encoding of the entire problem
    GrB_Type ttype,             // type of Tx
    GrB_Type stype,             // type of Sx
    GrB_BinaryOp dup            // dup binary operator to macrofy
) ;

//------------------------------------------------------------------------------
// select kernel
//------------------------------------------------------------------------------

uint64_t GB_encodify_select     // encode an select problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const int kcode,            // kernel to encode
    const bool C_iso,
    const bool in_place_A,
    const GrB_IndexUnaryOp op,
    const bool flipij,
    const GrB_Matrix A
) ;

bool GB_enumify_select      // enumerate a GrB_selectproblem
(
    // output:
    uint64_t *scode,        // unique encoding of the entire operation
    // input:
    bool C_iso,
    bool in_place_A,
    // operator:
    GrB_IndexUnaryOp op,    // the index unary operator to enumify
    bool flipij,            // if true, flip i and j
    // A matrix:
    GrB_Matrix A
) ;

void GB_macrofy_select          // construct all macros for GrB_select
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    // operator:
    const GrB_IndexUnaryOp op,
    GrB_Type atype
) ;

GrB_Info GB_select_bitmap_jit      // select bitmap
(
    int8_t *Cb,
    int64_t *cnvals_handle,
    const bool C_iso,
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int nthreads
) ;

#endif

