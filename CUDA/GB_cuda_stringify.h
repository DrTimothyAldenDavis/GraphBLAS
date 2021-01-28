//------------------------------------------------------------------------------
// GB_cuda_stringify.h: prototype definitions for using C helpers 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd only in the GraphBLAS/CUDA/GB_cuda*.cu source files.

#ifndef GB_CUDA_STRINGIFY_H
#define GB_CUDA_STRINGIFY_H

// length of strings for building semiring code and names
#define GB_CUDA_STRLEN 2048

//------------------------------------------------------------------------------
// GB_cuda_stringify_mask: define macros that access the mask matrix M
//------------------------------------------------------------------------------

void GB_cuda_stringify_mask     // return string to define mask macros
(
    // output:
    char **mask_macros,         // string that defines the mask macros
    // input:
    const GB_Type_code mcode,   // typecode of the mask matrix M,
                                // or 0 if M is not present
    bool Mask_struct,           // true if M structural, false if valued
    bool Mask_comp              // true if M complemented
) ;

void GB_cuda_enumify_mask       // return enum to define mask macros
(
    // output:
    int *mask_ecode,            // enumified mask
    // input
    const GB_Type_code mcode,   // typecode of the mask matrix M,
                                // or 0 if M is not present
    bool Mask_struct,           // true if M structural, false if valued
    bool Mask_comp              // true if M complemented
) :

void GB_cuda_macrofy_mask       // return enum to define mask macros
(
    // output:
    char **mask_macros,         // string that defines the mask macros
    // input
    int mask_ecode              // enumified mask
) ;

//------------------------------------------------------------------------------
// GB_cuda_stringify_semiring: build all strings for a semiring
//------------------------------------------------------------------------------

void GB_cuda_stringify_semiring     // build a semiring (name and code)
(
    // input:
    GrB_Semiring semiring,  // the semiring to stringify
    bool flipxy,            // multiplier is: mult(a,b) or mult(b,a)
    GrB_Type ctype,         // the type of C
    GrB_Type atype,         // the type of A
    GrB_Type btype,         // the type of B
    GrB_Type mtype,         // the type of M, or NULL if no mask
    bool Mask_struct,       // mask is structural
    bool mask_in_semiring_name, // if true, then the semiring_name includes
                                // the mask_name.  If false, then semiring_name
                                // is independent of the mask_name
    // output: (all of size at least GB_CUDA_LEN+1)
    char *semiring_name,    // name of the semiring
    char *semiring_code,    // List of types and macro defs
    char *mask_name         // definition of mask data load
) ;

//------------------------------------------------------------------------------
// GB_cuda_stringify_binop and supporting methods
//------------------------------------------------------------------------------

void GB_cuda_stringify_binop
(
    // output:
    char *binop_macro,  // string with the #define macro
    // input:
    const char *macro_name,   // name of macro to construct
    GB_Opcode opcode,   // opcode of GraphBLAS operator to convert into a macro
    GB_Type_code zcode, // op->xtype->code of the operator
    bool for_semiring,  // if true: op is a multiplier in a semiring
    bool flipxy         // if true, use mult(y,x) else mult(x,y)
) ;

void GB_cuda_enumify_binop
(
    // output:
    int *ecode,         // enumerated operator, range 0 to 110; -1 on failure
    // input:
    GB_Opcode opcode,   // opcode of GraphBLAS operator to convert into a macro
    GB_Type_code zcode, // op->xtype->code of the operator
    bool for_semiring   // true for A*B, false for A+B or A.*B
) ;

void GB_cuda_charify_binop
(
    // output:
    char **op_string,   // string defining the operator (NULL if failure)
    // input:
    int ecode           // from GB_cuda_enumify_binop
) ;

void GB_cuda_macrofy_binop
(
    // output:
    char *binop_macro,          // string with the #define macro
    // input:
    const char *macro_name,     // name of macro to construct
    char *op_string,            // string defining the operator
    bool flipxy                 // if true, use mult(y,x) else mult(x,y)
) ;

//------------------------------------------------------------------------------
// GB_cuda_stringify_identity and supporting methods
//------------------------------------------------------------------------------

void GB_cuda_stringify_identity     // return string for identity value
(
    // output:
    char *identity_macro,    // string with the #define macro
    // input:
    GB_Opcode opcode,       // must be a built-in binary operator from a monoid
    GB_Type_code zcode      // type code of the binary operator
) ;

void GB_cuda_enumify_identity       // return enum of identity value
(
    // output:
    int *ecode,             // enumerated identity, 0 to 17 (-1 if fail)
    // input:
    GB_Opcode opcode,       // built-in binary opcode of a monoid
    GB_Type_code zcode      // type code used in the opcode we want
) ;

void GB_cuda_charify_identity_or_terminal
(
    // output:
    char **value_string,        // string encoding the value
    // input:
    int ecode                   // enumerated identity/terminal value
) ;

void GB_cuda_macrofy_identity
(
    // output:
    char *identity_macro,       // string with #define macro
    // input:
    const char *value_string    // string defining the identity value
) ;

//------------------------------------------------------------------------------
// GB_cuda_stringify_terminal and supporting methods
//------------------------------------------------------------------------------

void GB_cuda_stringify_terminal         // return strings to check terminal
(
    // outputs:
    bool *is_monoid_terminal,           // true if monoid is terminal
    char *terminal_expression_macro,    // #define for terminal expression macro
    char *terminal_statement_macro,     // #define for terminal statement macro
    // inputs:
    const char *terminal_expression_macro_name,     // name of expression macro
    const char *terminal_statement_macro_name,      // name of statement macro
    GB_Opcode opcode,    // must be a built-in binary operator from a monoid
    GB_Type_code zcode   // type code of the binary operator
) ;

void GB_cuda_enumify_terminal       // return enum of terminal value
(
    // output:
    bool *is_monoid_terminal,   // true if monoid is terminal
    int *ecode,                 // enumerated terminal, 0 to 17 (-1 if fail)
    // input:
    GB_Opcode opcode,           // built-in binary opcode of a monoid
    GB_Type_code zcode          // type code used in the opcode we want
) ;

void GB_cuda_charify_terminal_expression    // string for terminal expression
(
    // output:
    char *terminal_expression,          // string with terminal expression
    // input:
    char *terminal_string,              // string with terminal value
    bool is_monoid_terminal,            // true if monoid is terminal
    int ecode                           // ecode of monoid operator
) ;

void GB_cuda_charify_terminal_statement // string for terminal statement
(
    // output:
    char *terminal_statement,           // string with terminal statement
    // input:
    char *terminal_string,              // string with terminal value
    bool is_monoid_terminal,            // true if monoid is terminal
    int ecode                           // ecode of monoid operator
) ;

void GB_cuda_macrofy_terminal_expression    // macro for terminal expression
(
    // output:
    char *terminal_expression_macro,
    // intput:
    const char *terminal_expression_macro_name,
    const char *terminal_expression
) ;

void GB_cuda_macrofy_terminal_statement     // macro for terminal statement
(
    // output:
    char *terminal_statement_macro,
    // intput:
    const char *terminal_statement_macro_name,
    const char *terminal_statement
) ;

//------------------------------------------------------------------------------
// GB_cuda_stringify_load: return a string to load/typecast macro
//------------------------------------------------------------------------------

void GB_cuda_stringify_load         // return a string to load/typecast macro
(
    // output:
    char *load_macro,               // string with #define macro to load value
    // input:
    const char *load_macro_name,    // name of macro to construct
    bool is_pattern                 // if true, load/cast does nothing
) ;

//------------------------------------------------------------------------------
// GB_cuda_stringify_opcode: name of unary/binary opcode
//------------------------------------------------------------------------------

const char *GB_cuda_stringify_opcode    // name of unary/binary opcode
(
    GB_Opcode opcode    // opcode of GraphBLAS unary or binary operator
) ;

//------------------------------------------------------------------------------
// GB_stringify_sparsity: define macros for sparsity structure
//------------------------------------------------------------------------------

void GB_stringify_sparsity  // construct macros for sparsity structure
(
    // output:
    char *sparsity_macros,  // macros that define the sparsity structure
    // intput:
    char *matrix_name,      // "C", "M", "A", or "B"
    GrB_Matrix A
) ;

void GB_enumify_sparsity    // enumerate the sparsity structure of a matrix
(
    // output:
    int *ecode,             // enumerated sparsity structure
    // input:
    GrB_Matrix A
) ;

void GB_macrofy_sparsity    // construct macros for sparsity structure
(
    // output:
    char *sparsity_macros,  // macros that define the sparsity structure
    // input:
    char *matrix_name,      // "C", "M", "A", or "B"
    int ecode
) ;

//------------------------------------------------------------------------------
// for GB_binop_flip and related methods
//------------------------------------------------------------------------------

#include "GB_binop.h"

#endif

