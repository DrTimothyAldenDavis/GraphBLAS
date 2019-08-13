function test2
%TEST2 test binary operators

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

optype = test_types ;
opnames = test_binops ;

for k1 = 1:length(opnames)

    opname = opnames {k1} ;
    fprintf ('\n=================================== %s\n', opname) ;

    for k2 = 0:length(optype)

        op = opname ;
        if (k2 > 0)
            op = [op '.' optype{k2}] ;
        end
        fprintf ('\nop: [%s]\n', op) ;
        if (k2 > 0)
            gb.binopinfo (op)
        else
            gb.binopinfo (op, 'double')
        end
    end
end

fprintf ('test2: all tests passed\n') ;

