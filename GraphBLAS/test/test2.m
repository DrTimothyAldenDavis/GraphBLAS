function test2
%TEST2 test

optype = list_types ;
opnames = list_binops ;

X = 1 ;

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

