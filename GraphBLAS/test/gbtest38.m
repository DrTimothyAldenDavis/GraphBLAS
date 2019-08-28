function gbtest38
%GBTEST38 test sqrt, eps, ceil, floor, round, fix, real, conj, ...
% isfinite, isinf, isnan, spfun, eig

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;

for trial = 1:40

    A = 1e3 * rand (3) ;
    B = single (A) ;

    G = gb (A) ;
    H = gb (B) ;

    err = norm (sqrt (A) - sqrt (G), 1) ; assert (err < 8 * eps ('double')) ;
    err = norm (sqrt (B) - sqrt (H), 1) ; assert (err < 8 * eps ('single')) ;

    assert (isequal (eps (A), double (eps (G)))) ;
    assert (isequal (eps (B), single (eps (H)))) ;

    assert (isequal (ceil (A), double (ceil (G)))) ;
    assert (isequal (ceil (B), single (ceil (H)))) ;

    assert (isequal (floor (A), double (floor (G)))) ;
    assert (isequal (floor (B), single (floor (H)))) ;

    assert (isequal (round (A), double (round (G)))) ;
    assert (isequal (round (B), single (round (H)))) ;

    assert (isequal (fix (A), double (fix (G)))) ;
    assert (isequal (fix (B), single (fix (H)))) ;

    assert (isequal (real (A), double (real (G)))) ;
    assert (isequal (real (B), single (real (H)))) ;

    assert (isequal (conj (A), double (conj (G)))) ;
    assert (isequal (conj (B), single (conj (H)))) ;

    C = A ;
    C (1,1) = inf ;
    C (2,2) = nan ;
    G = gb (C) ;

    assert (isequal (isfinite (C), double (isfinite (G)))) ;
    assert (isequal (isnan    (C), double (isnan    (G)))) ;

    A = sprand (10, 10, 0.5) ;
    G = gb (A) ;
    assert (isequal (spfun (@exp, A), double (spfun (@exp, G)))) ;

    A = rand (10) ;
    G = gb (A) ;
    assert (isequal (eig (A), double (eig (G)))) ;

end

fprintf ('gbtest38: all tests passed\n') ;
