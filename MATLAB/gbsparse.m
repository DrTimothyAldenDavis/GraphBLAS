function A = gbsparse (X)
%GBSPARSE convert a GraphBLAS matrix into a MATLAB sparse matrix
%
% Usage:  A = gbsparse (X)
%
% The input X is a GraphBLAS struct or a MATLAB sparse matrix.  The output A is
% a MATLAB sparse matrix.  The matrix X can have any MATLAB type: logical,
% int*, uint*, single, double, or complex.  If X is logical, double, complex
% (if supported), then A has the same type.  Otherwise, X is typecasted to
% double.
%
% See also gb.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('gbsparse mexFunction not found; use gbmake to compile GraphBLAS') ;

