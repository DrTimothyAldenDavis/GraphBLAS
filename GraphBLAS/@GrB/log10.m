function C = log10 (G)
%LOG10 Base-10 logarithm of the entries of a GraphBLAS matrix
% C = log10 (G) computes the base-10 logarithm of each entry of a GraphBLAS
% matrix G.  Since log10 (0) is nonzero, the result is a full matrix.
% If any entry in G is negative, the result is complex.
%
% See also GrB/log, GrB/log1p, GrB/log2, GrB/exp.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

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

C = GrB.apply ('log10', full (G)) ;

% same behavior as log
if (~isreal (C) && nnz (imag (C) == 0))
    C = real (C) ;
end

