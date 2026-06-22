function e = nvals (G)
%NVALS the number of entries in a matrix.
% e = GrB.nvals (G) is the number of entries in a GraphBLAS matrix G.  A
% GraphBLAS matrix G may have explicit zero entries, and these are
% included in the count e.  Thus, nnz (G) <= GrB.nvals (G).
%
% See also GrB.entries, GrB.prune, GrB/nonzeros, GrB/size, GrB/numel,
% GrB/nnz.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

if (isobject (G))
    e = gbmex_nvals (G) ;
else
    % for a MATLAB/Octave matrix: nvals is the same as nnz
    e = nnz (G) ;
end

