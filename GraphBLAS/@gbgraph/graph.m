function C = graph (G, uplo)
%GRAPH convert a gbgraph into a MATLAB undirected graph.
% C = graph (G) converts a gbgraph G into an undirected MATLAB graph.  G may be
% directed or undirected.  If it is directed, then the undirected graph is
% constructed with the adjacency matrix of G+G'.  No weights are added if G is
% logical.
%
% A second argument modifies how a directed gbgraph G is converted into the
% undirected graph C, where C = graph (G, 'upper') uses the upper triangular
% part of G, and C = graph (G, 'lower') uses the lower triangular part of G.

% TODO test

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

type = gb.type (G) ;
n = numnodes (G) ;

if (nargin > 1)

    % C = graph (G, uplo)
    if (isequal (type, 'logical'))
        C = graph (logical (G), uplo) ;
    elseif (isequal (type, 'double'))
        C = graph (double (G), uplo) ;
    else
        % all other types (int*, uint*, single)
        if (isequal (uplo, 'lower'))
            [i, j, x] = gb.extracttuples (tril (G)) ;
        elseif (isequal (uplo, 'upper'))
            [i, j, x] = gb.extracttuples (triu (G)) ;
        else
            error ('usage: graph(G,''lower'') or graph(G,''upper'')') ;
        end
        C = graph (i, j, x, n) ;
    end

else

    % C = graph (G)
    if (isdirected (G))
        G = G + G' ;
    end

    if (isequal (type, 'logical'))
        C = graph (logical (G)) ;
    elseif (isequal (type, 'double'))
        C = graph (double (G)) ;
    else
        % all other types (int*, uint*, single)
        [i, j, x] = gb.extracttuples (tril (G)) ;
        C = graph (i, j, x, n) ;
    end

end

