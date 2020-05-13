function C = acos (G)
%ACOS inverse cosine.
% C = acos (G) computes the inverse cosine of each entry of a GraphBLAS
% matrix G.  Since acos (0) is nonzero, the result is a full matrix.
% C is complex if any (abs(G) > 1).
%
% See also GrB/cos, GrB/cosh, GrB/acosh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isreal (G))
    if (any (abs (G) > 1, 'all'))
        if (isequal (GrB.type (G), 'single'))
            G = GrB (G, 'single complex') ;
        else
            G = GrB (G, 'double complex') ;
        end
    elseif (~isfloat (G))
        G = GrB (G, 'double') ;
    end
end

C = GrB.apply ('acos', full (G)) ;

