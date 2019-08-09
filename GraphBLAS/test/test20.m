function test20
%TEST20 test bandwidth, isdiag, ceil, floor, round, fix

for trial = 1:40
    fprintf ('.') ;
    for m = 0:10
        for n = 0:10
            A = 100 * sprand (m, n, 0.5) ;
            G = gb (A) ;
            [lo1, hi1] = bandwidth (A) ;
            [lo2, hi2] = bandwidth (G) ;
            assert (isequal (lo1, lo2)) ;
            assert (isequal (hi1, hi2)) ;
            d1 = isdiag (A) ;
            d2 = isdiag (G) ;
            assert (isequal (d1, d2)) ;

            assert (isequal (ceil  (A), sparse (ceil  (G)))) ;
            assert (isequal (floor (A), sparse (floor (G)))) ;
            assert (isequal (round (A), sparse (round (G)))) ;
            assert (isequal (fix   (A), sparse (fix   (G)))) ;
        end
    end
end

fprintf ('\ntest20: all tests passed\n') ;

