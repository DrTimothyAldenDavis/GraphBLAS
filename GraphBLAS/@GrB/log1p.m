function C = log1p (G)
%LOG1P natural logarithm of the entries of a GraphBLAS matrix
% C = log1p (G) computes the log(1+x) for each entry of a GraphBLAS
% matrix G.  If any entry in G is < -1, the result is complex.
%
% See also GrB/log, GrB/log2, GrB/log10, GrB/exp.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% FUTURE: the GxB_LOG1P_FC* ops are not accurate for the complex case

if (isreal (G))
    if (GrB.issigned (G) && any (G < -1, 'all'))
        if (isequal (GrB.type (G), 'single'))
            G = GrB (G, 'single complex') ;
        else
            G = GrB (G, 'double complex') ;
        end
    elseif (~isfloat (G))
        G = GrB (G, 'double') ;
    end
end

C = GrB.apply ('log1p', G) ;

