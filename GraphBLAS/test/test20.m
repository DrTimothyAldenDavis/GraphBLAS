function test20

% TODO ... in progress (currently fails)
for trial = 1:40
    for m = 1:10
        for n = 1:10
            A = sprand (m, n, 0.5) ;
            A
            G = gb (A) ;
            [lo1, hi1] = bandwidth (A)
            [lo2, hi2] = bandwidth (G)
            assert (isequal (lo1, lo2)) ;
            assert (isequal (hi1, hi2)) ;
            d1 = isdiag (A)
            d2 = isdiag (G)
            assert (isequal (d1, d2)) ;
        end
    end
end

fprintf ('test20: all tests passed\n') ;

