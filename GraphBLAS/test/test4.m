function test4
%TEST4 list all 1865 possible semirings
% This count excludes operator synonyms

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

types = test_types ;
ops = test_binops ;

nsemirings = 0 ;

for k1 = 1:length (ops)
    add = ops {k1} ;
    for k2 = 1:length (ops)
        mult = ops {k2} ;
        for k3 = 1:length (types)
            type = types {k3} ;

            semiring = [add '.' mult '.' type] ;

            try
                gb.semiring (semiring) ;
                nsemirings = nsemirings + 1 ;
                fprintf ('%s\n', semiring) ;
            catch me
            end
        end
    end
end

nsemirings

fprintf ('\ntest4: all tests passed\n') ;

