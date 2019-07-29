%TEST4 list all 1865 possible semirings
% This count excludes operator synonyms

types = list_types ;
ops = list_binops ;

nsemirings = 0 ;

for k1 = 1:length (ops)
    add = ops {k1} ;
    for k2 = 1:length (ops)
        mult = ops {k2} ;
        for k3 = 1:length (types)
            type = types {k3} ;

            semiring = [add '.' mult '.' type] ;

            try
                gbsemiring (semiring) ;
                nsemirings = nsemirings + 1 ;
                fprintf ('%s\n', semiring) ;
            catch me
            end
        end
    end
end

nsemirings

