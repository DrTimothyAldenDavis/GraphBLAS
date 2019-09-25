function C = expand (scalar, S)
%GB.EXPAND expand a scalar into a GraphBLAS matrix.
% C = gb.expand (scalar, S) expands the scalar into a matrix with the
% same size and pattern as S, as C = scalar*spones(S).  C has the same
% type as the scalar.  The numerical values of S are ignored; only the
% pattern of S is used.  The inputs may be either GraphBLAS and/or
% MATLAB matrices/scalars, in any combination.  C is returned as a
% GraphBLAS matrix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

% FUTURE: this is slow.  Use a built-in mexFunction.

% FUTURE: as much as possible, replace scalar expansion with binary operators
% used in a unary apply, when it becomes part of the C API.

C = gb.gbkron (['1st.' gb.type(scalar)], scalar, S) ;

