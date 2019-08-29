function H = plot (G, varargin)
% H = plot (G, varargin) plots the graph of the GraphBLAS matrix G.
% If G is symmetric, G is plotted as an undirected graph.
% If G is square and unsymmetric, G is plotted as a directed graph.
% If G is rectangular, the bipartite graph is plotted.

% TODO test

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[m n] = size (G) ;
if (m == n)
    if (issymmetric (G))
        plot (graph (logical (G)), varargin {:}) ;
        title (sprintf (...
            'GraphBLAS: undirected graph, %d nodes %d edges\n', n, nnz (G))) ;
    else
        plot (digraph (logical (G)), varargin {:}) ;
        title (sprintf (...
            'GraphBLAS: directed graph, %d nodes %d edges\n', n, nnz (G))) ;
    end
else
    G = [gb(m,m) G ; G' gb(n,n)] ;
    plot (graph (logical (G)), varargin {:}) ;
    title (sprintf (...
        'GraphBLAS: bipartite graph, [%d %d] nodes, %d edges\n', ...
        m, n, nnz (G))) ;
end

