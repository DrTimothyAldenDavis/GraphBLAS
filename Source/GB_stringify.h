//------------------------------------------------------------------------------
// GB_stringify.h: prototype definitions construction of *.h definitions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_STRINGIFY_H
#define GB_STRINGIFY_H

//------------------------------------------------------------------------------
// print copyright and license
//------------------------------------------------------------------------------

void GB_macrofy_copyright (FILE *fp) ;

//------------------------------------------------------------------------------
// dump definitions (for debugging and test coverage only)
//------------------------------------------------------------------------------

// uncomment this line to dump GB*.h files to /tmp, or compile with
// -DGB_DEBUGIFY_DEFN=1
// #undef  GB_DEBUGIFY_DEFN
#define GB_DEBUGIFY_DEFN 1

//------------------------------------------------------------------------------
// for GB_boolean_rename and related methods
//------------------------------------------------------------------------------

#include "GB_binop.h"

//------------------------------------------------------------------------------
// left and right shift
//------------------------------------------------------------------------------

#define GB_LSHIFT(x,k) (((uint64_t) x) << k)
#define GB_RSHIFT(x,k,b) ((x >> k) & ((((uint64_t)0x00000001) << b) -1))

//------------------------------------------------------------------------------
// GrB_reduce
//------------------------------------------------------------------------------

void GB_enumify_reduce      // enumerate a GrB_reduce problem
(
    // output:
    uint64_t *rcode,        // unique encoding of the entire problem
    // input:
    GrB_Monoid reduce,      // the monoid to enumify
    GrB_Matrix A
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

void GB_enumify_ewise         // enumerate a GrB_eWise problem
(
    // output:
    uint64_t *scode,        // unique encoding of the entire operation
    // input:
    // C matrix:
    bool C_iso,             // if true, operator is ignored
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
    GrB_Matrix A,
    GrB_Matrix B
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

void GB_enumify_mxm         // enumerate a GrB_mxm problem
(
    // output:
    uint64_t *scode,        // unique encoding of the entire semiring
    // input:
    // C matrix:
    bool C_iso,             // if true, semiring is ignored
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
// GrB_select
//------------------------------------------------------------------------------

void GB_enumify_select
(
    // output:
    uint64_t *select_code,      // unique encoding of the selector
    // input:
    bool C_iso,                 // true if C is iso
    GB_Opcode opcode,           // selector opcode
    const GB_Operator op,       // user operator, NULL for resize/nonzombie
    const bool flipij,          // if true, flip i and j for user operator
    GrB_Matrix A,               // input matrix
    bool in_place_A             // true if select is done in-place
) ;

void GB_typify_select           // determine x,y,z types for select
(
    // outputs:
    GrB_Type *xtype,            // x,y,z types for select operator
    GrB_Type *ytype,
    GrB_Type *ztype,
    // inputs:
    GB_Opcode opcode,           // selector opcode
    const GB_Operator op,       // user operator, NULL in some cases
    GrB_Type atype              // the type of the A matrix
) ;

void GB_macrofy_select
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t select_code,       // unique encoding of the selector
    GB_Opcode opcode,           // selector opcode
    const GB_Operator op,       // user operator, NULL for resize/nonzombie
    GrB_Type atype
) ;

char *GB_namify_select          // determine the select op name
(
    // inputs:
    GB_Opcode opcode,           // selector opcode
    const GB_Operator op        // user operator, NULL in some cases
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
    int term_ecode,     // terminal value as an enum (< 30 is terminal)
    GrB_Monoid monoid   // monoid to macrofy (NULL if not used)
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
    bool flipxy,                // if true: op is f(y,x), multipicative only
    bool is_monoid,             // if true: additive operator for monoid
    int ecode,
    GrB_BinaryOp op
) ;

//------------------------------------------------------------------------------
// operator definitions and typecasting
//------------------------------------------------------------------------------

void GB_macrofy_defn
(
    FILE *fp,
    int kind,
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
    int A_iso_code          // 1 if A is iso
) ;

void GB_macrofy_output
(
    FILE *fp,
    // input:
    const char *cname,      // name of the scalar ... = cij to write
    const char *Cmacro,     // name of the macro is GB_PUT*(Cmacro)
    const char *Cname,      // name of the output matrix
    GrB_Type ctype,         // type of C
    GrB_Type ztype,         // type of cij scalar to write to C
    int csparsity,          // sparsity format of the output matrix
    bool C_iso              // true if C is iso
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

const char *GB_charify_identity_or_terminal // return string encoding the value
(
    // input:
    int ecode                   // enumerated identity/terminal value
) ;

void GB_macrofy_bytes
(
    FILE *fp,               // file to write macros, assumed open already
    // input:
    const char *Name,       // all-upper-case name
    const char *variable,   // variable to declaer
    const char *type_name,  // name of the type
    const uint8_t *value,   // array of size nbytes
    size_t nbytes
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
    const char *name,       // name of the type
    size_t size             // size of the type
) ;

size_t GB_padify_typesize   // pad the size of a type
(
    size_t size
) ;

//------------------------------------------------------------------------------
// GB_namify_problem: name a problem
//------------------------------------------------------------------------------

void GB_namify_problem
(
    // output:
    char *problem_name,     // of size at least 256 + 8*GxB_MAX_NAME_LEN
    // input:
    const uint64_t scode,
    const bool builtin,     // true if all objects are builtin
    const char *opname1,    // each string has size at most GxB_MAX_NAME_LEN
    const char *opname2,
    const char *typename1,
    const char *typename2,
    const char *typename3,
    const char *typename4,
    const char *typename5,
    const char *typename6
) ;

//------------------------------------------------------------------------------
// GB_debugify_*: dump the definition file to /tmp
//------------------------------------------------------------------------------

void GB_debugify_mxm
(
    // C matrix:
    bool C_iso,             // if true, operator is ignored
    int C_sparsity,         // sparse, hyper, bitmap, or full
    GrB_Type ctype,         // C=((ctype) T) is the final typecast
    // M matrix:
    GrB_Matrix M,
    bool Mask_struct,
    bool Mask_comp,
    // semiring:
    GrB_Semiring semiring,
    bool flipxy,
    // A and B matrices:
    GrB_Matrix A,
    GrB_Matrix B
) ;

void GB_debugify_ewise
(
    // C matrix:
    bool C_iso,             // if true, operator is ignored
    int C_sparsity,         // sparse, hyper, bitmap, or full
    GrB_Type ctype,         // C=((ctype) T) is the final typecast
    // M matrix:
    GrB_Matrix M,
    bool Mask_struct,
    bool Mask_comp,
    // operator:
    GrB_BinaryOp binaryop,
    bool flipxy,
    // A and B matrices:
    GrB_Matrix A,
    GrB_Matrix B
) ;

void GB_debugify_reduce     // enumerate and macrofy a GrB_reduce problem
(
    // input:
    GrB_Monoid monoid,      // the monoid to enumify
    GrB_Matrix A
) ;

void GB_debugify_select
(
    bool C_iso,                 // true if C is iso
    GB_Opcode opcode,           // selector opcode
    const GB_Operator op,       // user operator, NULL for resize/nonzombie
    const bool flipij,          // if true, flip i and j for user operator
    GrB_Matrix A,               // input matrix
    int64_t ithunk,             // (int64_t) Thunk, if Thunk is NULL
    const GrB_Scalar Thunk,     // optional input for select operator
    bool in_place_A             // true if select is done in-place
) ;

#endif

