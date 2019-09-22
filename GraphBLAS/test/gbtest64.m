function gbtest64
%GBTEST64 test gb.pagerank

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

load west0479 ;
W = abs (west0479) ;
W (1,:) = 0 ;

A = digraph (W) ;
G = gb (W) ;
R = gb (W, 'by row') ;

r1 = centrality (A, 'pagerank') ;
r2 = gb.pagerank (G) ;
assert (norm (r1-r2) < 1e-12) ;

r1 = centrality (A, 'pagerank') ;
r2 = gb.pagerank (R) ;
assert (norm (r1-r2) < 1e-12) ;

r1 = centrality (A, 'pagerank', 'Tolerance', 1e-8) ;
r2 = gb.pagerank (G, struct ('tol', 1e-8)) ;
assert (norm (r1-r2) < 1e-12) ;

lastwarn ('') ;
warning ('off', 'MATLAB:graphfun:centrality:PageRankNoConv') ;
warning ('off', 'gb:pagerank') ;

r1 = centrality (A, 'pagerank', 'MaxIterations', 2) ;
[msg, id] = lastwarn ;
assert (isequal (id, 'MATLAB:graphfun:centrality:PageRankNoConv')) ;

r2 = gb.pagerank (G, struct ('maxit', 2)) ;
[msg, id] = lastwarn ;
assert (isequal (id, 'gb:pagerank')) ;
assert (norm (r1-r2) < 1e-12) ;

lastwarn ('') ;

r1 = centrality (A, 'pagerank', 'FollowProbability', 0.5) ;
r2 = gb.pagerank (G, struct ('damp', 0.5)) ;
assert (norm (r1-r2) < 1e-12) ;

r1 = gb.pagerank (G, struct ('weighted', true)) ;
r2 = gb.pagerank (R, struct ('weighted', true)) ;
assert (norm (r1-r2) < 1e-12) ;

fprintf ('gbtest64: all tests passed\n') ;

