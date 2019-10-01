//------------------------------------------------------------------------------
// gb_get_mxargs: get input arguments to a GraphBLAS mexFunction 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// gb_get_mxargs collects all the input arguments for the 12 foundational
// GraphBLAS operations.  The user-level view is described below.  For
// the private mexFunctions, the descriptor always appears as the last
// argument.  The matrix arguments are either MATLAB sparse or full matrices,
// GraphBLAS matrices.  To call the mexFunction, the opaque content of the
// GraphBLAS matrices has been extracted, so they appear here as GraphBLAS
// structs.

/*

% GraphBLAS has 12 foundational operations, listed below.  All have similar
% parameters.  The full set of input parameters is listed in the order in which
% they appear in the GraphBLAS C API, except that for the MATLAB interface, Cin
% and Cout are different matrices.  They combine into a single input/output
% matrix in the GraphBLAS C API.  In the MATLAB interface, many of the
% parameters become optional, and they can appear in different order.

%   Cout = GrB.mxm       (Cin, M, accum, op, A, B,  desc)
%   Cout = GrB.kronecker (Cin, M, accum, op, A, B,  desc)
%   Cout = GrB.eadd      (Cin, M, accum, op, A, B,  desc)
%   Cout = GrB.emult     (Cin, M, accum, op, A, B,  desc)

%   Cout = GrB.select    (Cin, M, accum, op, A, thunk,  desc)

%   Cout = GrB.vreduce   (Cin, M, accum, op, A,     desc)
%   Cout = GrB.apply     (Cin, M, accum, op, A,     desc)

%   Cout = GrB.assign    (Cin, M, accum, A, I, J,   desc)
%   Cout = GrB.subassign (Cin, M, accum, A, I, J,   desc)
%   Cout = GrB.extract   (Cin, M, accum, A, I, J,   desc)

%   Cout = GrB.trans     (Cin, M, accum, A,         desc)

%   Cout = GrB.reduce    (Cin,    accum, op, A,     desc)

% For the MATLAB interface, the parameters divide into 4 classes: matrices,
% strings, cells, and a single optional struct (the descriptor).  The order of
% parameters between the matrices, strings, and cell classes is arbitrary.  The
% order of parameters within a class is important; for example, if a method
% takes 4 matrix inputs, then they must appear in the order Cin, M, A, and then
% B.  However, if a single string appears as a parameter, it can appear
% anywhere within the list of 4 matrices.

% (1) Cin, M, A, B are matrices.  If the method takes up to 4 matrices
%       (mxm, kronecker, select (with operator requiring a thunk parameter),
%       eadd, emult), then they appear in this order:
%       with 2 matrix inputs: A, B
%       with 3 matrix inputs: Cin, A, B
%       with 4 matrix inputs: Cin, M, A, B
%       For the GrB.select, B is the thunk scalar.
%
%   If the method takes up to 3 matrices (vreduce, apply, assign, subassign,
%       extract, trans, or select without thunk):
%       with 1 matrix input:  A
%       with 2 matrix inputs: Cin, A
%       with 3 matrix inputs: Cin, M, A
%       Note that assign and subassign require Cin.
%
%   If the method takes up to 2 input matrices (just the reduce method):
%       with 1 matrix input:  A
%       with 2 matrix inputs: Cin, A

% (2) accum and op are strings.  The accum string is always optional.  If
%       the method has an op parameter, then it is a required input.
%
%       If the method has both parameters, and just one string appears, it is
%       the op, which is a semiring for mxm, a unary operator for apply, a
%       select operator for the select method, and a binary operator for all
%       other methods.  If 2 strings appear, the first one is the accum the
%       second is the op.  If the accum appears then Cin must also appear as a
%       matrix input.
%
%       If the method has no op (assign, subassign, extract, trans), but just
%       an accum parameter, then 0 or 1 strings may appear in the parameter
%       list.  If a string appears, it is the accum.

% (3) I and J are cell arrays.  For details, see the assign, subassign, and
%       extract methods; a short summary appears below.  Both are optional:
%       with no cell inputs: default for I and J
%       with 1  cell inputs: I, default for J
%       with 2  cell inputs: I, J
%
%       Each cell array may appear with 0, 1, 2, or 3 items:
%           0: { }                  ":" in MATLAB notation
%           1: { list }             a list of integer indices
%           2: { start,fini }       start:fini in MATLAB notation
%           3: { start,inc,fini }   start:inc:fini in MATLAB notation

% (4) The descriptor is an optional struct.  If present, it must always
%       appears last, after all other parameters.

% Some valid uses are shown below, along with their equivalent in GraphBLAS
% notation.  For the first three mxm examples, the four matrices C, M, A, and B
% must appear in that order, and the two strings '+' and '+,*' must appear in
% that order, but the matrices and strings may be interleaved arbitrarily.

%   C = GrB.mxm (C, M, '+', '+.*', A, B)        C<M> += A*B
%   C = GrB.mxm (C, M, '+', A, '+.*', B)        C<M> += A*B
%   C = GrB.mxm ('+', '+,*', C, M, A, B)        C<M> += A*B

%   C = GrB.mxm ('+.*', A, B)                   C = A*B
%   C = GrB.mxm (A, '+.*', B)                   C = A*B
%   C = GrB.mxm (C, M, A, '+.*', B)             C<M> = A*B

%   C = GrB.emult (C, M, '+', A, '*', B)        C<M> += A.*B
%   C = GrB.emult (A, '*', B)                   C = A.*B

%   C = GrB.assign (C, M, '+', A, I, J)         C(I,J)<M> += A
%   C = GrB.assign (C, I, J, M, '+', A)         C(I,J)<M> += A

%   C = GrB.assign (C, A, I, J)                 C(I,J) = A
%   C = GrB.assign (C, I, J, A)                 C(I,J) = A
%   C = GrB.assign (C, A)                       C = A
%   C = GrB.assign (C, M, A)                    C<M> = A
%   C = GrB.assign (C, M, '+', A)               C<M> += A
%   C = GrB.assign (C, '+', A, I)               C (I,:) += A

%   C = GrB.extract (C, M, '+', A, I, J)        C<M> += A(I,J)
%   C = GrB.extract (A, I, J)                   C = A(I,J)
%   C = GrB.extract (I, J, A)                   C = A(I,J)
%   C = GrB.extract (A)                         C = A
%   C = GrB.extract (C, M, A)                   C<M> = A
%   C = GrB.extract (C, M, '+', A)              C<M> += A
%   C = GrB.extract (C, '+', A, I)              C += A(I,:)

%   C = GrB.apply (C, M, '|', '~', A)           C<M> |= ~A
%   C = GrB.apply ('~', A)                      C = ~A

%   c = GrB.reduce (c, '+', 'max', A)           c += max (A)
%   c = GrB.reduce ('max', A)                   c = max (A)
%   c = GrB.reduce (A, 'max')                   c = max (A)
%   c = GrB.reduce (c, 'max', A)                c = max (A)

*/

#include "gb_matlab.h"

void gb_get_mxargs
(
    // input:
    int nargin,                 // # input arguments for mexFunction
    const mxArray *pargin [ ],  // input arguments for mexFunction
    const char *usage,          // usage to print, if too many args appear

    // output:
    const mxArray *Matrix [4],  // matrix arguments
    int *nmatrices,             // # of matrix arguments
    const mxArray *String [2],  // string arguments
    int *nstrings,              // # of string arguments
    const mxArray *Cell [2],    // cell array arguments
    int *ncells,                // # of cell array arguments
    GrB_Descriptor *desc,       // last argument is always the descriptor
    base_enum_t *base,          // desc.base
    kind_enum_t *kind,          // desc.kind
    GxB_Format_Value *fmt       // desc.format
)
{

    //--------------------------------------------------------------------------
    // get the descriptor
    //--------------------------------------------------------------------------

    (*desc) = gb_mxarray_to_descriptor (pargin [nargin-1], kind, fmt, base) ;

    //--------------------------------------------------------------------------
    // find the remaining arguments
    //--------------------------------------------------------------------------

    (*nmatrices) = 0 ;
    (*nstrings) = 0 ;
    (*ncells) = 0 ;

    for (int k = 0 ; k < (nargin-1) ; k++)
    {
        if (mxIsCell (pargin [k]))
        {
            // I or J index arguments
            if ((*ncells) >= 2)
            { 
                ERROR ("only 2D indexing is supported") ;
            }
            Cell [(*ncells)++] = pargin [k] ;
        }
        else if (mxIsChar (pargin [k]))
        {
            // accum operator, unary op, binary op, monoid, or semiring
            if ((*nstrings) >= 2)
            { 
                ERROR (usage) ;
            }
            String [(*nstrings)++] = pargin [k] ;
        }
        else
        {
            // a matrix argument is C, M, A, or B
            if ((*nmatrices) >= 4)
            { 
                // at most 4 matrix inputs are allowed
                ERROR (usage) ;
            }
            Matrix [(*nmatrices)++] = pargin [k] ;
        }
    }
}

