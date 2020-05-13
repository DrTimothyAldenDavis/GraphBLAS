function C = acosh (G)
%ACOSH inverse hyperbolic cosine.
% C = acosh (G) computes the inverse hyperbolic cosine of each entry of a
% GraphBLAS matrix G.  Since acosh (0) is nonzero, the result is a full
% matrix.  C is complex if any (G < 1).
%
% See also GrB/cos, GrB/acos, GrB/cosh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isreal (G))
    if (any (G < 1, 'all'))
        if (isequal (GrB.type (G), 'single'))
            G = GrB (G, 'single complex') ;
        else
            G = GrB (G, 'double complex') ;
        end
    elseif (~isfloat (G))
        G = GrB (G, 'double') ;
    end
end

C = GrB.apply ('acosh', full (G)) ;

