function gbtest43
%GBTEST43 test error handling

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

ok = true ;
G = gb (magic (5)) ;

try
    x = prod (G, 'gunk') ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    x = min (G, [ ], 'gunk') ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    x = max (G, [ ], 'gunk') ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    x = min (G, [ ], 1, G) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    x = max (G, [ ], 1, G) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    x = sum (G, 'gunk') ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    x = all (G, 'gunk') ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    x = any (G, 'gunk') ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    [x, y] = bandwidth (G, 'lower') ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    x = bandwidth (G, 'gunk') ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    x = gb.eye (1, 2, 3) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    x = gb.eye ([1, 2, 3]) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    G.stuff = 3 ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    G (2,1,1) = 3 ;
    ok = false
catch me
    me
end
assert (ok) ;

H = gb (rand (4,3)) ;
try
    C = H^H ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = G^H ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = G^(-1) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = G(1,2).stuff(3,4) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = G.stuff ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = G (3:4) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = G (1,2,2) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = spones (G, 'gunk') ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = spones (G, G) ;
    ok = false
catch me
    me
end
assert (ok) ;

G = gb (magic (2), 'int16') ;
try
    C = eps (G)
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = gb.entries (G, 'gunk')
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = gb.entries (G, 'all', 'degree')
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = norm (G, 0) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = norm (G, 42) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = norm (G, -inf) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = norm (G, 2) ;
    ok = false
catch me
    me
end
assert (ok) ;

v = gb (rand (4,1)) ;
try
    C = norm (G, 42) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = reshape (v, 42, 42) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = reshape (v, [2 2 2]) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = zeros (3, 3, 'crud', G) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = zeros ([3, 3], 'crud', G) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = zeros (3, 3, 'like', G, 'gunk') ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = reshape (v, [2 2], 2) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = norm (v, 'gunk') ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = norm (v, 3) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = norm (G) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    v = gb.bfs (v, 1) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    v = gb.bfs (G, 1, 'gunk') ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    % G must be symmetric
    v = gb.bfs (G, 1, 'symmetric', 'check') ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    v = [G v] ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    v = [G ; v] ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = complex (gb (1), gb (1)) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = gb.empty (4, 4) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = gb.empty (0, 4, 4) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    C = gb.empty ([0, 4, 4]) ;
    ok = false
catch me
    me
end
assert (ok) ;

try
    c = gb.tricount (v) ;
    ok = false
catch me
    me
end
assert (ok) ;

fprintf ('gbtest43: all tests passed\n') ;

