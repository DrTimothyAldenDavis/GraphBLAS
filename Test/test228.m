function test228
%TEST228 test serialize/deserialize for all sparsity formats

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('test228 C = serialize (A) for all sparsity formats and all types\n') ;

[~, ~, ~, types, ~, ~] = GB_spec_opsall ;
types = types.all ;

rng ('default') ;

for k1 = 1:length (types)
    atype = types {k1} ;
    fprintf ('%s ', atype) ;
    for d = [0.5 inf]
        A = GB_spec_random (10, 10, d, 128, atype) ;
        for A_sparsity = 0:15
            A.sparsity = A_sparsity ;
            for method = [-1 0 1000 2000:2009]
                C = GB_mex_serialize (A, method) ;      % default: fast
                GB_spec_compare (A, C) ;
                C = GB_mex_serialize (A, method, 0) ;   % fast
                GB_spec_compare (A, C) ;
                C = GB_mex_serialize (A, method, 502) ; % secure
                GB_spec_compare (A, C) ;
            end
        end
    end
end

fprintf ('\n') ;
fprintf ('test228: all tests passed\n') ;

