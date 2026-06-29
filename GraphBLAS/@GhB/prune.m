function C = prune (G, id)
%GRB.PRUNE remove explicit values from a matrix.
% C = GrB.prune (G) removes any explicit zeros from G.
% C = GrB.prune (G, id) removes entries equal to the given scalar id.
%
% See also GrB/full, GrB.select, GrB.prune.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

if (nargin == 1)
    C = gb_prune (ghb, G) ;
else
    C = gb_prune (ghb, G, id) ;
end

