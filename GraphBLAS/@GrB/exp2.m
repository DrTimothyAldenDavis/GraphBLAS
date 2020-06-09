function C = exp2 (G)
%EXP2 base 2 exponential.
% C = exp2 (G) is 2^x for each entry x of the matrix G.
% Since 2^0 is nonzero, C is a full matrix.
%
% See also GrB/pow2, GrB/exp, GrB/expm1, GrB/exp2, GrB/log, GrB/log10,
% GrB/log2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
type = gbtype (G) ;

if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gbapply ('exp2', gbfull (G, type))) ;

