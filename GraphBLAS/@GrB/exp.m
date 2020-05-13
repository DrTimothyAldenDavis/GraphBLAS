function C = exp (G)
%EXP exponential of the entries of a GraphBLAS matrix
% C = exp (G) computes e^x of each entry x of a GraphBLAS matrix G.
% Since e^0 is nonzero, C is a full matrix.
%
% See also GrB/exp, GrB/expm1, GrB/exp2, GrB/log, GrB/log10, GrB/log2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (~isfloat (G))
    G = GrB (G, 'double') ;
end

C = GrB.apply ('exp', full (G)) ;

