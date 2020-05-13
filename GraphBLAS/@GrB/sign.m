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

% remove explicit zeros
G = GrB.prune (G) ;
type = GrB.type (G) ;

if (contains (type, 'uint') || isequal (type, 'logical'))

    % logical or unsigned integer
    C = spones (G, type) ;

elseif (contains (type, 'int'))

    % signed integer
    C = spones (GrB.select (G, '>0')) - spones (GrB.select (G, '<0')) ;

elseif (contains (type, 'complex'))

    % single or double complex
    C = GrB.emult (G, '/', abs (G)) ;

else

    % single or double real
    C = GrB.emult ('copysign', spones (G, type), G) ;

end

