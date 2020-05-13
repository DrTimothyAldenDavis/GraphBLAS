function C = tan (G)
%TAN tangent.
% C = tan (G) computes the tangent of each entry of a GraphBLAS matrix G.
%
% See also GrB/tanh, GrB/atan, GrB/atanh, GrB/atan2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (~isfloat (G))
    G = GrB (G, 'double') ;
end

C = GrB.apply ('tan', G) ;

