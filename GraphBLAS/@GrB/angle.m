function C = angle (G)
%ANGLE phase angle of the entries of a GraphBLAS matrix
% C = angle (G) computes the phase angle of each entry of a GraphBLAS
% matrix G.
%
% See also GrB/abs.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isreal (G))
    C = zeros (size (G), 'like', G) ;
else
    C = GrB.apply ('carg', G) ;
end

