function gbtest4
%GBTEST4 list all 1918 possible semirings
% This count excludes operator synonyms

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

types = gbtest_types ;
ops = gbtest_binops ;

nsemirings = 0 ;
npair = 0 ;

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
                GrB.semiringinfo (semiring) ;
                GrB.semiringinfo (s, type) ;
                nsemirings = nsemirings + 1 ;
                ok = true ;
            catch
                % this is an error, but it is expected since not all
                % combinations operators and types can be used to construct a
                % valid semiring.
                ok = false ;
            end
        end
    end
end

nsemirings
assert (nsemirings == 1918)
GrB.semiringinfo

fprintf ('\ngbtest4: all tests passed\n') ;

