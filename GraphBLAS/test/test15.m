function test15
%TEST15 unary operators

types = list_types ;
ops = { 'identity', '~', '-', '1', 'minv', 'abs' }

for k1 = 1:length (ops)
    for k2 = 1:length (types)
        op = [ops{k1} '.' types{k2}] ;
        fprintf ('\nop: %s\n', op) ;
        gb.unopinfo (op) ;
    end
end

