function test19
%TEST19 test mpower

rng ('default') ;
A = rand (4) ;
G = gb (A) ;

for k = 0:10
    C1 = A^k ;
    C2 = G^k ;
    err = sparse(C1) - sparse (C2) ;
    assert (norm (err,1) < 1e-12)
end

fprintf ('test19: all tests passed\n') ;
    
