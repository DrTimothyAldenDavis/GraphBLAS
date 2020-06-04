function U = triu (G, k)
%TRIU upper triangular part of a GraphBLAS matrix.
% U = triu (G) returns the upper triangular part of G.
%
% U = triu (G,k) returns the entries on and above the kth diagonal of X,
% where k=0 is the main diagonal.
%
% See also GrB/tril.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (nargin < 2)
    k = 0 ;
elseif (isobject (k))
    k = gb_get_scalar (k) ;
end

if (isobject (G))
    G = G.opaque ;
end

U = GrB (gbselect ('triu', G, k)) ;

