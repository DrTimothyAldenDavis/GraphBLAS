function C = asin (G)
%ASIN inverse sine.
% C = asin (G) computes the inverse sine of each entry of a GraphBLAS
% matrix G.  C is complex if any entry in any (abs(G) > 1)
%
% See also GrB/sin, GrB/sinh, GrB/asinh.

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

C = GrB.apply ('asin', G) ;

