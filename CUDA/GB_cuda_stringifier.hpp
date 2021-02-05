// Class to manage both stringify functions from semiring, ops and monoids to char buffers
// Also provides a iostream callback to deliver the buffer to jitify as if read from a file

// (c) Nvidia Corp. 2020 All rights reserved 
// SPDX-License-Identifier: Apache-2.0

// Implementations of string callbacks
#pragma once
#include <iostream>
#include "GB.h"
#include "GB_binop.h"
#include "GB_stringify.h"

// Define function pointer we will use later
//std::istream* (*file_callback)(std::string, std::iostream&);

// Define a factory class for building any buffer of text
class GB_cuda_stringifier {
  char callback_buffer[2048];
  char *callback_string;
  const char *include_filename;

  public:

//------------------------------------------------------------------------------
// load string: set string and file name to mimic
//------------------------------------------------------------------------------
    void load_string(const char *fname, char *input)
    {
        callback_string = input; 
        include_filename =  fname;
    }

//------------------------------------------------------------------------------
// callback: return string as if it was read from a file 
//------------------------------------------------------------------------------

    std::istream* callback( std::string filename, std::iostream& tmp_stream) 
    {
        if ( filename == std::string(this->include_filename) )
        {
           tmp_stream << this->callback_string; 
           return &tmp_stream;
        }
        else 
        {
           return nullptr;
        }
    }

    void enumify_semiring {}
    (
        // output:
        uint64_t *scode,        // unique encoding of the entire semiring
        // input:
        GrB_Semiring semiring,  // the semiring to enumify
        bool flipxy,            // multiplier is: mult(a,b) or mult(b,a)
        GrB_Type ctype,         // the type of C
        GrB_Type mtype,         // the type of M, or NULL if no mask
        GrB_Type atype,         // the type of A
        GrB_Type btype,         // the type of B
        bool Mask_struct,       // mask is structural
        bool Mask_comp,         // mask is complemented
        int C_sparsity,         // sparsity structure of C
        int M_sparsity,         // sparsity structure of M
        int A_sparsity,         // sparsity structure of A
        int B_sparsity          // sparsity structure of B
    )
    {
        C_sparsity = GB_sparsity (C) ;
        ...
    }

    void macrofy_semiring {}
    (
        // output:
        char *semiring_macros,  // List of types and macro defs
        // input:
        uint64_t scode          // unique encoding of the entire semiring
    )
    {

    }


};

