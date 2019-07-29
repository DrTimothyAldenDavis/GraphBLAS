
clear

optype = list_types ;

% TODO: 'complex' 

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

    end

end
