function gbtest47
%GBTEST47 test gb.entries, gb.nonz, numel

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;

A = magic (4) ;
c0 = nnz (A) ;
c1 = gb.nonz (A) ;
c2 = gb.nonz (gb (A)) ;
assert (c0 == c1) ;
assert (c1 == c2) ;

c1 = gb.nonz (A, 0) ;
c2 = gb.nonz (gb (A), 0) ;
assert (c0 == c1) ;
assert (c1 == c2) ;

A = sparse (A) ;
c0 = nnz (A) ;

c1 = gb.nonz (A) ;
c2 = gb.nonz (gb (A)) ;
assert (c0 == c1) ;
assert (c1 == c2) ;

c1 = gb.nonz (A, 0) ;
c2 = gb.nonz (gb (A), 0) ;
assert (c0 == c1) ;
assert (c1 == c2) ;

A = sparse (A) ;
c0 = nnz (A ~= 1) ;
c1 = gb.nonz (A, 1) ;
c2 = gb.nonz (gb (A), 1) ;
assert (c0 == c1) ;
assert (c1 == c2) ;

try
    x = vpa (1) ;
    have_symbolic = true ;
catch me
    have_symbolic = false ;
end

for trial = 1:40
    fprintf ('.') ;

    A = rand (4) ;
    A (A > .5) = 0 ;
    G = gb (A) ;

    c1 = gb.entries (A) ;
    c2 = gb.entries (G) ;
    assert (c1 == c2) ;
    assert (c1 == numel (A)) ;

    c1 = gb.nonz (A) ;
    c2 = gb.nonz (G) ;
    assert (c1 == c2) ;
    assert (c1 == nnz (A)) ;

    B = sparse (A) ;
    G = gb (B) ;

    c1 = gb.entries (B) ;
    c2 = gb.entries (G) ;
    assert (c1 == c2) ;
    assert (c1 == nnz (B)) ;

    c1 = gb.nonz (B) ;
    c2 = gb.nonz (G) ;
    assert (c1 == c2) ;
    assert (c1 == nnz (B)) ;

    c1 = gb.nonz (B, 0) ;
    c2 = gb.nonz (G, 0) ;
    assert (c1 == c2) ;
    assert (c1 == nnz (B)) ;

    x1 = gb.entries (B, 'list') ;
    x2 = gb.entries (G, 'list') ;
    assert (isequal (x1, x2)) ;

    x1 = gb.nonz (B, 'list') ;
    x2 = gb.nonz (G, 'list') ;
    x0 = unique (nonzeros (B)) ;
    assert (isequal (x0, x2)) ;
    assert (isequal (x1, x2)) ;

    d1 = gb.entries (B, 'row') ;
    d2 = gb.entries (G, 'row') ;
    d3 = length (find (sum (spones (B), 2))) ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gb.nonz (B, 'row') ;
    d2 = gb.nonz (G, 'row') ;
    d3 = length (find (sum (spones (B), 2))) ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gb.entries (B, 'row', 'list') ;
    d2 = gb.entries (G, 'row', 'list') ;
    d3 = find (sum (spones (B), 2)) ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gb.nonz (B, 'row', 'list') ;
    d2 = gb.nonz (G, 'row', 'list') ;
    d3 = find (sum (spones (B), 2)) ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gb.entries (B, 'col') ;
    d2 = gb.entries (G, 'col') ;
    d3 = length (find (sum (spones (B), 1))) ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gb.nonz (B, 'col') ;
    d2 = gb.nonz (G, 'col') ;
    d3 = length (find (sum (spones (B), 1))) ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gb.entries (B, 'col', 'list') ;
    d2 = gb.entries (G, 'col', 'list') ;
    d3 = find (sum (spones (B), 1))' ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    d1 = gb.nonz (B, 'col', 'list') ;
    d2 = gb.nonz (G, 'col', 'list') ;
    d3 = find (sum (spones (B), 1))' ;
    assert (isequal (d1, d2)) ;
    assert (isequal (d1, d3)) ;

    % requires vpa in the Symbolic toolbox:
    if (have_symbolic)
        Huge = gb (2^30, 2^30) ;
        e = numel (Huge) ;
        assert (logical (e == 2^60)) ;
    end

end

fprintf ('\ngbtest47: all tests passed\n') ;

