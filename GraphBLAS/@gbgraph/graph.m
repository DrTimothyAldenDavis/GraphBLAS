function C = graph (H, uplo)
%GRAPH convert a gbgraph into a MATLAB undirected graph.
% C = graph (H) converts a gbgraph H into an undirected MATLAB graph.  H may be
% directed or undirected.  If it is directed, then the undirected graph is
% constructed with the adjacency matrix of H+H'.  No weights are added if H is
% logical.
%
% A second argument modifies how a directed gbgraph H is converted into the
% undirected graph C, where C = graph (H, 'upper') uses the upper triangular
% part of H, and C = graph (H, 'lower') uses the lower triangular part of H.

% TODO test

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

type = gb.type (H) ;

if (nargin > 1)

    % C = graph (H, uplo)
    if (isequal (type, 'logical'))
        C = graph (logical (H), uplo) ;
    elseif (isequal (type, 'double'))
        C = graph (double (H), uplo) ;
    else
        % all other types (int*, uint*, single)
        if (isequal (uplo, 'lower'))
            [i, j, x] = gb.extracttuples (tril (H)) ;
        elseif (isequal (uplo, 'upper'))
            [i, j, x] = gb.extracttuples (triu (H)) ;
        else
            error ('usage: graph(H,''lower'') or graph(H,''upper'')') ;
        end
        C = graph (i, j, x) ;
    end

else

    % C = graph (H)
    if (isdirected (H))
        H = H + H' ;
    end

    if (isequal (type, 'logical'))
        C = graph (logical (H)) ;
    elseif (isequal (type, 'double'))
        C = graph (double (H)) ;
    else
        % all other types (int*, uint*, single)
        [i, j, x] = gb.extracttuples (tril (H)) ;
        C = graph (i, j, x) ;
    end

end

