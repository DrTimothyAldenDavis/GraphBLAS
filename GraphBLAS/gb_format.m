function [f,s,iso] = gb_format (arg)
%GB_FORMAT: implments GrB.format and GhB.format.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    % f = GrB.format ; get the global format
    if (nargout > 1)
        error ('GrB:error', 'usage: f = GrB.format') ;
    end
    f = gbmex_format ;
else
    % f = GrB.format (A) ; get the format of A (built-in or GraphBLAS)
    % f = GrB.format (f) ; set the global format for all matrices.
    if (nargout <= 1)
        f = gbmex_format (arg) ;
    else
        [f,s,iso] = gbmex_format (arg) ;
    end
end

