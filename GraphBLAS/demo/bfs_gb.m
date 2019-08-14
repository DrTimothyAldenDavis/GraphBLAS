function v = bfs_gb (A, s)
%BFS_GB a simple breadth-first-search using the MATLAB interface to GraphBLAS
%
% v = bfs_gb (A, s)
%
% A is a square binary matrix, corresponding to the adjacency matrix of a
% graph, with A(i,j)=1 denoting the edge (i,j).  Self loops are permitted, and
% A may be unsymmetric.  s is a scalar input with the source node.  The output
% v is the level each node in the graph, where v(s)=1 (the first level), v(j)=2
% if there is an edge (s,j) (the 2nd level), etc.  v(j)=k if node j is in the
% kth level, where the shortest path (in terms of # of edges) from  s to j has
% length k+1.  The source node s defaults to 1.

if (nargin < 2)
    s = 1 ;
end

[m n] = size (A) ;
if (m ~= n)
    error ('A must be square') ;
end

v = gb (zeros (n,1)) ;
q = gb (n, 1, 'logical') ;
q (s) = true ;
if (~isequal (gb.type (A), 'logical'))
    A = gb (A, 'logical') ;
end
d.mask = 'complement' ;
d.out = 'replace' ;
d.in0 = 'transpose' ;

for level = 1:n
    % v<q> = level; assign level to all nodes in the queue q
    v = gb.assign (v, q, level) ;
    if (~full (any (q)))
        % break if q is empty
        break ;
    end
    % q<~v,replace> = A'*q, using the boolean semiring
    q = gb.mxm (q, v, '|.&.logical', A, q, d) ;
end

