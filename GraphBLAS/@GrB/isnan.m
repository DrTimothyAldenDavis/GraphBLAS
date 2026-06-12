function C = isnan (G)
%ISNAN true for NaN elements.
% C = isnan (G) is a logical C matrix with C(i,j)=true if G(i,j) is NaN.
%
% See also GrB/isinf, GrB/isfinite.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n, type] = gbmex_size (G) ;

if (gb_isfloat (type) && gbmex_nvals (G) > 0)   % FIXME remove nvals
    C = GrB (gbmex_apply ('isnan', G)) ;
else
    % C is all false
    C = GrB (m, n, 'logical') ;
end

