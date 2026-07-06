function C = gb_zeros (ghb, varargin)
%GB_ZEROS implements GrB.zeros and GhB.zeros.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

for k = 1:nargin-1
    if (gb_is_grb (varargin {k}))
        varargin {k} = struct (varargin {k}) ;
    end
end

[m, n, type] = gb_parse_args ('zeros', varargin {:}) ;
C = gzb (ghb, m, n, type) ;

