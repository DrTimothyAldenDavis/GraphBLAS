function C = gb_sparse (ghb, G)
%GB_SPARSE implements GrB/sparse and GhB/sparse.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[~, sparsity] = gbmex_format (G) ;

switch (sparsity)
    case { 'hypersparse', 'sparse' }
        % nothing to do; G is already sparse or hypersparse
        C = gzb (ghb, G) ;
    case { 'bitmap', 'full' }
        % convert G to sparse or hypersparse
        C = gzb (ghb, G, 'sparse/hypersparse') ;
end

