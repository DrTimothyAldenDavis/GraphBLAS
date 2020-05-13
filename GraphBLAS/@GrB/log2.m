function [F, E] = log2 (G)
%LOG2 Base-2 logarithm of the entries of a GraphBLAS matrix
% C = log2 (G) computes the base-2 logarithm of each entry of a GraphBLAS
% matrix G.  Since log2 (0) is nonzero, the result is a full matrix.
% If any entry in G is negative, the result is complex.
%
% [F,E] = log2 (G) returns F and E so that G = F.*(2.^E), where entries
% in abs (F) are either in the range [0.5,1), or zero if the entry in G is
% zero.  F and E are both sparse, with the same pattern as G.  If G is
% complex, [F,E] = log2 (real (G)).
%
% See also GrB/pow2, GrB/log, GrB/log1p, GrB/log10, GrB/exp.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (nargout == 1)

    % C = log2 (G)
    if (isreal (G))
        if (GrB.issigned (G) && any (G < 0, 'all'))
            if (isequal (GrB.type (G), 'single'))
                G = GrB (G, 'single complex') ;
            else
                G = GrB (G, 'double complex') ;
            end
        elseif (~isfloat (G))
            G = GrB (G, 'double') ;
        end
    end
    F = GrB.apply ('log2', full (G)) ;

    % same behavior as log
    if (~isreal (F) && nnz (imag (F) == 0))
        F = real (F) ;
    end

else

    % [F,E] = log2 (G)
    if (~isfloat (G))
        G = GrB (G, 'double') ;
    elseif (~isreal (G))
       G = real (G) ;
    end
    F = GrB.apply ('frexpx', G) ;
    E = GrB.apply ('frexpe', G) ;

end

