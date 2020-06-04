function C = imag (G)
%IMAG complex imaginary part.
% C = imag (G) returns the imaginary part of the GraphBLAS matrix G.
%
% See also GrB/conj, GrB/real.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
type = gbtype (G) ;

if (contains (type, 'complex'))
    % C = imag (G) where G is complex
    C = GrB (gbapply ('cimag', G)) ;
else
    % G is real, so C = zeros (m,n)
    [m, n] = gbsize (G) ;
    C = GrB (gbnew (m, n, type)) ;
end

