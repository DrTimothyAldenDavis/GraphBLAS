function [m n] = gbsize (X)
%GBSIZE size of a GraphBLAS or MATLAB sparse matrix
%
% Usage
%   [m n] = gbsize (X)
%
% gbsize returns the dimensions m and n for an m-by-n MATLAB sparse matrix or
% GraphBLAS sparse matrix.
%
% See also size.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('gbsize mexFunction not found; use gbmake to compile GraphBLAS') ;
