function C = sprand (G)
%SPRAND  sparse uniformly distributed GraphBLAS random matrix.
% C = sprand (G) has the same pattern as A, but uniformly
%   distributed random entries.  If the same random seed is used,
%   and if G and A have the same pattern, sprand (G) and the MATLAB
%   sprand (A) produce the same result.
%
% The entries in C will greater than zero and less than one.
%
% See also GrB/sprandn, GrB/sprandsym, GrB.random.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% TODO

C = GrB.random (G) ;

