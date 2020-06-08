function C = sign (G)
%SIGN Signum function.
% C = sign (G) computes the signum function for each entry in the
% GraphBLAS matrix G.  For each element of G, sign(G) returns 1 if the
% element is greater than zero, 0 if it equals zero, and -1 if it is less
% than zero.  For the complex case, C = G ./ abs (G).
%
% See also GrB/abs.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

Q = G.opaque ;
type = gbtype (Q) ;

if (isequal (type, 'logical'))
    C = G ;
elseif (~gb_isfloat (type))
    C = GrB (gbnew (gbapply ('signum.single', Q), type)) ;
else
    C = GrB (gbapply ('signum', Q)) ;
end

