
% A = mread ('cover.mtx') ;
clear all
gb.threads (8)
% gb.chunk (1024*1024)

%%MatrixMarket matrix coordinate pattern general
%%GraphBLAS GrB_BOOL
% Matrix from the cover of "Graph Algorithms in the Language of Linear
% Algebra", Kepner and Gilbert.  Note that cover shows A'.  This is A.
% 7 7 12
ij = [
4 1
1 2
4 3
6 3
7 3
1 4
7 4
2 5
7 5
3 6
5 6
2 7 ] ;


source = 1 ;

A = sparse (ij (:,1), ij (:,2), ones (12,1), 7, 7) ;
A = gb (A, 'by row') ;

H = gbgraph (A, 'logical', 'by row') 

v = bfs (H, source)
[v pi] = bfs (H, source)

G = graph (H) ;
v2 = bfsearch (G, source) ;

levels = v (v2)

Prob = ssget (2294) ;
A = Prob.A ;
A = A+A' ;
H = gbgraph (A, 'logical', 'by row') ;
tic
v = bfs (H, source)
toc
nnz (v)

G = graph (H) ;
tic 
v2 = bfsearch (G, source) ;
toc
length (v2)

tic 
t = bfsearch (G, source, 'allevents') ;
toc

fprintf ('compute the bfs tree from source node:\n') ;
tic
[v, pi] = bfs (H, source) ;
gb_time = toc

tic 
p = shortestpathtree (G, source) ;
matlab_time = toc

