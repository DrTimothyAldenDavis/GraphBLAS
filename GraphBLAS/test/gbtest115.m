function gbtest115
%GBTEST115 test serialize/deserialize

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

rng ('default') ;

types = gbtest_types ;
modes = { 'fast', 'secure' } ;
compression_methods = { 'none', 'lz4', 'lz4hc' } ;

for k = 1:length(types)
    type = types {k} ;
    A = GrB (GrB.random (5, 10, 0.4, 'range', [0 10]), type) ;

    % defaults
    blob = GrB.serialize (A) ;
    B = GrB.deserialize (blob) ;
    assert (isequal (A, B)) ;

    for k1 = 1:2
        mode = modes {k1} ;
        for k2 = 1:3
            method = compression_methods {k2} ;

            % default level
            blob = GrB.serialize (A, method) ;
            B = GrB.deserialize (blob, mode) ;
            assert (isequal (A, B)) ;

            % levels 0:9 for lz4hc
            if (k2 == 3)
                for level = 0:9
                    blob = GrB.serialize (A, method, level) ;
                    B = GrB.deserialize (blob, mode) ;
                    assert (isequal (A, B)) ;
                end
            end

            % with typecasting
            blob = GrB.serialize (A, method) ;
            B = GrB.deserialize (blob, mode, 'double') ;
            assert (isequal (GrB (A, 'double'), B)) ;
        end
    end
end

fprintf ('\ngbtest115: all tests passed\n') ;


