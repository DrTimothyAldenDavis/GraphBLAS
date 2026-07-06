function C = gb_ones (ghb, varargin)
%GB_ONES implements GrB.ones and GhB.ones.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

for k = 1:nargin-1
    if (gb_is_grb (varargin {k}))
        varargin {k} = struct (varargin {k}) ;
    end
end

[m, n, type] = gb_parse_args (ghb, 'ones', varargin {:}) ;
C = gb_scalar_to_full (ghb, m, n, type, gbmex_format, 1) ;

