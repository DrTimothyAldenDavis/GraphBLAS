function [I J X] = gbfind (A, onebased)
%GBFIND extract a list of entries from a matrix
%
% Usage:
%
%   [I J X] = gbfind (A)            % extract 1-based indices; I and J double
%   [I J X] = gbfind (A, 0) ;       % extract 0-based indices; I and J uint64
%   [I J X] = gbfind (A, 1) ;       % extract 1-based indices; I and J double
%
% gbfind extracts all the tuples from either a MATLAB sparse matrix A or a
% GraphBLAS matrix A.  If A is a MATLAB sparse matrix, [I J X] = gbfind (A)
% is identical to [I J X] = find (A).
%
% An optional second argument determines the type of I and J.  It defaults to
% 1, and in this case, I and J are double, and reflect 1-based indices, just
% like the MATLAB statement [I J X] = find (A).  If zero, then I and J are
% returned as uint64 arrays, containing 0-based indices.
%
% This function corresponds to the GrB_*_extractTuples_* functions
% in GraphBLAS.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('gbfind mexFunction not found; use gbmake to compile GraphBLAS') ;
