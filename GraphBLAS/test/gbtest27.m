function gbtest27
%GBTEST27 test conversion to full

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

types = gbtest_types ;

rng ('default') ;
for k1 = 1:length (types)

    atype = types {k1} ;
    fprintf ('\n================================================ %s\n', atype) ;
    A = 100 * sprand (3, 3, 0.5) ;
    H = GrB (A, atype) %#ok<*NOPRT>
    G = full (H) %#ok<*NASGU>

    for k2 = 1:length (types)

        gtype = types {k2} ;
        fprintf ('\n------------ %s:\n', gtype) ;
        G = full (H, gtype)
        K = full (G, atype)
        for id = [0 1 inf]
            C = full (H, gtype, id)
        end
    end
end

fprintf ('gbtest27: all tests passed\n') ;

