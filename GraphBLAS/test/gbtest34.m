function gbtest34
%GBTEST34 test repmat

rng ('default') ;

for m1 = 0:5
    for n1 = 0:5
        for m2 = 0:5
            for n2 = 0:5
                A = rand (m1, n1) ;
                C = repmat (A, m2, n2) ;
                G = gb (A) ;
                H = repmat (G, m2, n2) ;
                assert (isequal (C, double (H))) ;
            end
        end
    end
end

fprintf ('gbtest34: all tests passed\n') ;

