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

if (isobject (G))
    % this is faster:
    e = gbnvals (G.opaque) ;
    % the mexFunction can read a @GrB object, but this is 2x slower:
    % e = gbnvals (G) ;
else
    % for a MATLAB/Octave matrix: nvals is the same as nnz
    e = nnz (G) ;
end

