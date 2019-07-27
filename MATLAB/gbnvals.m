function nvals = gbnvals (X)
%GBNVALS number of entries in a GraphBLAS or MATLAB sparse matrix
%
% Usage
%   nvals = gbnvals (X)
%
% gbnvals returns the number of entries in a MATLAB sparse matrix or GraphBLAS
% sparse matrix.  For a MATLAB sparse matrix, gbnvals (X) is the same as
% nnz (X).  No explicit zeros appear in a MATLAB sparse matrix.  For a
% GraphBLAS sparse matrix, explicit zeros may appear, and thus gbnvals (X)
% may be more than the number of nonzeros. 
%
% See also nnz.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('gbnvals mexFunction not found; use gbmake to compile GraphBLAS') ;

