function Cout = gbmxm (Cin, M, accum, semiring, A, B, desc)
%GBMXM sparse matrix-matrix multiplication
%
% The GraphBLAS operation GrB_mxm (C, M, accum, semiring, A, B, desc)
% modifies C in place, computing the following in GraphBLAS notation.
%
%   C<#M,replace> = accum (C, A*B)
%
% C is both an input and output matrix.  In this MATLAB to GraphBLAS, it is
% split into Cin (the value of C on input) and Cout (the value of C on output).
% M is the optional mask matrix, and #M is either M or !M depending on whether
% or not the mask is complemented via the desc.mask option.  The replace option
% is determined by desc.out.  A and/or B may optionally be transposed via the
% descriptor fields desc.in0 and desc.in1, respectively.
%
% Usage:
%
%   Cout = gbmxm (semiring, A, B)
%   Cout = gbmxm (semiring, A, B, desc)
%
%   Cout = gbmxm (Cin, accum, semiring, A, B)
%   Cout = gbmxm (Cin, accum, semiring, A, B, desc)
%
%   Cout = gbmxm (Cin, M, semiring, A, B)
%   Cout = gbmxm (Cin, M, semiring, A, B, desc)
%
%   Cout = gbmxm (Cin, M, accum, semiring, A, B)
%   Cout = gbmxm (Cin, M, accum, semiring, A, B, desc)
%
% Not all inputs are required.
%
% Cin is an optional input matrix.  If Cin is not present or is an empty matrix
% (Cin = [ ]) then it is implicitly a matrix with no entries, of the right size
% (which depends on A, B, and the descriptor).
%
% M is the optional mask matrix.  If not present, or if empty, then no mask is
% used.  If present, M must have the same size as C.
%
% If accum is not present, then the operation becomes C<...> = A*B.  Otherwise,
% accum (C,A*B) is computed.  The accum operator acts like a sparse matrix
% addition (see gbadd).
%
% The semiring is a required string defining the semiring to use, in the form
% 'add.mult.type', where '.type' is optional.  For example, '+.*.double' is the
% conventional semiring for numerical linear algebra, used in MATLAB for
% C=A*B when A and B are double.  If A or B are complex, then the '+.*.complex'
% semiring is used.  GraphBLAS has many more semirings it can use.  See 'help
% gbsemiring' for more details.
%
% A and B are the input matrices.  They are transposed on input if
% desc.in0 = 'transpose' (which transposes A), and/or
% desc.in1 = 'transpose' (which transposes B).
%
% The descriptor desc is optional.  If not present, all default settings are
% used.  Fields not present are treated as their default values.  See
% 'help gbdescriptor' for more details.
%
% See also gbdescriptor, gbadd, mtimes.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('gbmxm mexFunction not found; use gbmake to compile GraphBLAS') ;

