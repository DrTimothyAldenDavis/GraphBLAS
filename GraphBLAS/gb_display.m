function gb_display (ghb, name, A, level)
%GB_DISPLAY display the contents of a matrix.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 4)
    k = 2 ;
else
    k = gb_get_scalar (ghb, level) ;
end

if (~isempty (name))
    fprintf ('\n%s =\n', name) ;
end

gbmex_disp (ghb, A, k) ;

if (k > 1)
    fprintf ('\n') ;
end

