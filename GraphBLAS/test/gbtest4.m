function gbtest4
%GBTEST4 list all 1865 possible semirings
% This count excludes operator synonyms

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

types = gbtest_types ;
ops = gbtest_binops ;

nsemirings = 0 ;

for k1 = 1:length (ops)
    add = ops {k1} ;
    for k2 = 1:length (ops)
        mult = ops {k2} ;
        for k3 = 1:length (types)
            type = types {k3} ;

            s = [add '.' mult] ;
            semiring = [s '.' type] ;

            try
                fprintf ('\n================================ %s\n', semiring) ;
                gb.semiringinfo (semiring) ;
                gb.semiringinfo (s, type) ;
                nsemirings = nsemirings + 1 ;
            catch me
            end
        end
    end
end

assert (nsemirings == 1865)
gb.semiringinfo

fprintf ('\ngbtest4: all tests passed\n') ;

