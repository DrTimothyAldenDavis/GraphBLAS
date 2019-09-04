classdef gbgraph < gb
%GBGRAPH a directed or undirected graph, based on GraphBLAS.
%
% gbgraph is a graph object based on a GraphBLAS adjacency matrix.  To
% construct a gbgraph, use one of the following methods, where X is a square
% matrix (either a GraphBLAS matrix, or a MATLAB sparse or full matrix),
% a MATLAB graph object, or a MATLAB digraph object.
%
%   G = gbgraph (X) ;
%   G = gbgraph (X, kind, format)
%
% One or two optional strings may given, in any order.  With an optional string
% argument of 'undirected' or 'directed', the kind of graph can be specified.
% The default is to construct an undirected graph if G is symmetric, or a
% directed graph otherwise.  When constructing an undirected graph, G must be
% symmetric.  Another optional string allows the format of G to be specified:
% 'by row' or 'by col'.  With two strings, both parameters can be specified.
%
% A gbgraph has different methods available to it than the MATLAB graph and
% digraph.  Multigraphs, node properties, and edge properties of the MATLAB
% graph and digraph objects are not yet supported.  However, GraphBLAS allows
% for arbitrary linear algebra operations on its gbgraph objects.  This cannot
% be done with the MATLAB graph and digraph objects.
%
% Example:
%
%   A = sprand (5, 5, 0.5) ;
%   A = A+A'
%   G1 = gbgraph (A)
%   G2 = gbgraph (A, 'directed', 'by row')
%
% See also graph, digraph, gb.

% TODO tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

properties
    graphkind = [ ] ;    % a string, either 'undirected' or 'directed'

    % TODO add node and edge properties?
end

methods

    function G = gbgraph (X, varargin)
    %GBGRAPH create a GraphBLAS directed or undirected graph.
    % Creates a directed or undirected graph, based on a square adjacency
    % matrix X, or a MATLAB graph X or digraph.
    %
    % Usage:
    %
    %   G = gbgraph (X) ;
    %   G = gbgraph (X, kind, format, type) ;

    % get the input arguments
    type = '' ;
    kind = '' ;
    format = '' ;
    check = true ;
    for k = 1:nargin-1
        arg = varargin {k} ;
        switch (arg)
            case { 'undirected', 'directed' }
                kind = arg ;
            case { 'by row', 'by col' }
                format = arg ;
            case { 'double', 'single', 'logical', 'complex', ...
                   'int8', 'int16', 'int32', 'int64', ...
                   'uint8', 'uint16', 'uint32', 'uint64' }
                type = arg ;
            case { true, false }
                % for internal use in @gbgraph only:
                check = arg ;
            otherwise
                error ('unknown input parameter') ;
        end
    end

    % check the inputs, if called by the user
    if (check)

        if (isa (X, 'graph'))

            % C = gbgraph (X, ...) where X is a MATLAB undirected graph
            X = adjacency (X, 'weighted') ;
            if (isempty (kind))
                % C will be an undirected gbgraph, by default
                kind = 'undirected' ;
            end

        elseif (isa (X, 'digraph'))

            % C = gbgraph (X, ...) where X is a MATLAB directed graph
            X = adjacency (X, 'weighted') ;
            if (isempty (kind))
                % C will be a directed gbgraph, by default
                kind = 'directed' ;
            else
                % C will be an directed or undirected gbgraph, per kind input
                if (isequal (kind, 'undirected'))
                    % if converting from a digraph to an undirected gbgraph, X
                    % must be symmetric.
                    if (~issymmetric (X))
                        error ('X must be symmetric') ;
                    end
                end
            end

        elseif (isa (X, 'gbgraph'))

            % C = gbgraph (X, ...) where X is gbgraph.
            if (isempty (kind))
                % C will be an undirected or directed gbgraph, the same as X
                kind = X.graphkind ;
            else
                % C will be an directed or undirected gbgraph, per kind input
                if (isequal (kind, 'undirected') & isdirected (X))
                    % if converting from a directed gbgraph X to an undirected
                    % gbgraph G, X must be symmetric.
                    if (~issymmetric (X))
                        error ('X must be symmetric') ;
                    end
                end
            end

        else

            % C = gbgraph (X) where X is a matrix (GraphBLAS or MATLAB)
            [m n] = size (X) ;
            if (m ~= n)
                error ('X must be square') ;
            end
            if (isempty (kind))
                % C = gbgraph (X), either directed or undirected
                if (issymmetric (X))
                    % If X is symmetric, construct an undirected graph.
                    kind = 'undirected' ;
                else
                    % otherwise, construct a directed graph
                    kind = 'directed' ;
                end
            else
                % C = gbgraph (X, kind) ;
                % Construct C according to the kind input.  For this case,
                % if C is an undirected graph then X must be symmetric.
                if (isequal (kind, 'undirected') & ~issymmetric (X))
                    error ('X must be symmetric') ;
                end
            end

        end
    end

    % determine the matrix type and format
    if (isempty (format))
        format = gb.format (X) ;
    end
    if (isempty (type))
        type = gb.type (X) ;
    end

    % create the graph
    G = G@gb (X, type, format) ;
    G.graphkind = kind ;
    end

    %---------------------------------------------------------------------------
    % overloaded methods
    %---------------------------------------------------------------------------

    % methods in @gbgraph that overload the MATLAB graph and digraph methods:

    s = isequal (G1, G2) ;
    d = degree (G) ;
    C = digraph (G) ;
    display (G) ;
    disp (G, level) ;
    c = edgecount (G, s, t) ;
    C = graph (G, uplo) ;
    d = indegree (G) ;
    L = laplacian (G, type) ;
    [e d] = numedges (G) ;
    n = numnodes (G) ;
    d = outdegree (G) ;
    [handle titlehandle] = plot (G, varargin) ;
    S = subgraph (G, I) ;
    C = flipedge (G, varargin) ;
    s = ismultigraph (G) ;
    C = adjacency (G, arg) ;
    C = incidence (G, type) ;
    C = reordernodes (G, order) ;

    % FUTURE::
    %
    % methods for both classes graph and digraph not yet implemented:
    %
    %    addedge addnode bfsearch centrality conncomp dfsearch distances
    %    findedge findnode isisomorphic isomorphism maxflow nearest outedges
    %    rmedge rmnode shortestpath shortestpathtree simplify
    %
    %    gbgraph/bfs is like graph/bfsearch and graph/shortestpathtree.
    %
    % methods for class graph (not in digraph class) not yet implemented:
    %
    %    bctree biconncomp minspantree neighbors
    %
    % methods for class digraph (not in graph class) not yet implemented:
    %
    %    condensation inedges isdag predecessors successors toposort
    %    transclosure transreduction

    %---------------------------------------------------------------------------
    % additional methods in gbgraph
    %---------------------------------------------------------------------------

    % The following methods in @gbgraph are in addition to methods defined for
    % the MATLAB graph and digraph.

    r = pagerank (G, opts) ;
    C = prune (G, varargin) ;
    G = pruneself (G) ;
    e = nself (G) ;
    C = ktruss (G, k) ;
    [v, parent] = bfs_pushpull (G, GT, s) ;
    [v, parent] = bfs (G, s) ;  % like bfsearch and shortestpathtree
    k = kind (G) ;
    s = tricount (G, order) ;
    s = isdirected (G) ;
    s = isundirected (G) ;
    G = byrow (G) ;
    G = bycol (G) ;

    % FUTURE::
    %
    %   dnn, ... (see LAGraph)

end

end

