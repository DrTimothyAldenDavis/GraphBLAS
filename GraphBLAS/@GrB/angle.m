function C = angle (G)
%ANGLE phase angle of the entries of a GraphBLAS matrix
% C = angle (G) computes the phase angle of each entry of a GraphBLAS
% matrix G.
%
% See also GrB/abs.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
type = gbtype (G) ;

if (contains (type, 'complex'))
    C = GrB (gbapply ('carg', G)) ;
else
    % C is all zero
    [m, n] = gbsize (G) ;
    C = GrB (gbnew (m, n, type)) ;
end

