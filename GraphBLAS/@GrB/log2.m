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

G = G.opaque ;

if (nargout == 1)

    % C = log2 (G)
    F = GrB (gb_to_real_if_imag_zero (gb_trig ('log2', gbfull (G)))) ;

else

    % [F,E] = log2 (G)
    type = gbtype (G) ;
    if (~gb_isfloat (type))
        G = gbnew (G, 'double') ;
    elseif (contains (type, 'complex'))
        G = gbapply ('creal', G) ;
    end
    F = GrB (gbapply ('frexpx', G)) ;
    E = GrB (gbapply ('frexpe', G)) ;

end

