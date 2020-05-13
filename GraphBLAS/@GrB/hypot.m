function C = hypot (A, B)
%HYPOT robust computation of the square root of sum of squares.
% C = hypot (A,B) computes sqrt (abs (A).^2 + abs (B).^2) accurately.
% If A and B are matrices, the pattern of C is the set union of A and B.
% If one of A or B is a nonzero scalar, the scalar is expanded into a
% dense matrix the size of the other matrix, and the result is a full
% matrix.
%
% The input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  C is returned as a GraphBLAS matrix.
%
% See also GrB/abs, GrB/norm, GrB/sqrt, GrB/plus, GrB.eadd.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (~isreal (A))
    A = abs (A) ;
elseif (~isfloat (A))
    A = GrB (A, 'double') ;
end

if (~isreal (B))
    B = abs (B) ;
elseif (~isfloat (A))
    B = GrB (B, 'double') ;
end

C = gb_eadd (A, 'hypot', B) ;

