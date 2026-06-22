function L = tril (G, k)
%TRIL lower triangular part of a matrix.
% L = tril (G) returns the lower triangular part of G.
%
% L = tril (G,k) returns the entries on and below the kth diagonal of G,
% where k=0 is the main diagonal.
%
% See also GrB/triu.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

if (nargin < 2)
    k = 0 ;
else
    k = gb_get_scalar (k) ;
end

L = GrB (gbmex_select (ghb, 'tril', G, k)) ;

