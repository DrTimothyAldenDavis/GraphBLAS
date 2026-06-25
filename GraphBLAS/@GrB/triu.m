function U = triu (G, k)
%TRIU upper triangular part of a matrix.
% U = triu (G) returns the upper triangular part of G.
%
% U = triu (G,k) returns the entries on and above the kth diagonal of X,
% where k=0 is the main diagonal.
%
% See also GrB/tril.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (nargin < 2)
    k = 0 ;
else
    k = gb_get_scalar (k) ;
end

U = GrB (gbmex_select (ghb, 'triu', G, k)) ;

