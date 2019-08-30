classdef gbgraph < gb
%GBGRAPH a directed or undirected graph, based on GraphBLAS.
%
% gbgraph is a graph object based on a GraphBLAS adjacency matrix.  To
% construct a gbgraph, use one of the following methods, where G is a square
% matrix (either a GraphBLAS matrix, or a MATLAB sparse or full matrix),
% a MATLAB graph object, or a MATLAB digraph object.
%
%   H = gbgraph (G) ;
%   H = gbgraph (G, 'undirected') ;
%   H = gbgraph (G, 'directed') ;
%
% If the 'undirected' option is specified, G must be symmetric, and an
% undirected graph is created.  If not specified, H is constructed as an
% undirected graph if G is symmetric, or as a directed graph otherwise.
%
% Multigraphs, node properties, and edge properties of the MATLAB graph and
% digraph objects are not yet supported.  However, GraphBLAS allows for
% arbitrary linear algebra operations on its gbgraph objects.  This cannot be
% done with the MATLAB graph and digraph objects.

% TODO tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

properties
    graphkind = [ ] ;    % a string, either 'undirected' or 'directed'

    % TODO add node names here
end

methods

    function H = gbgraph (G, kind, check)
    %GBGRAPH create a GraphBLAS directed or undirected graph.
    % Creates a directed or undirected graph, based on a square adjacency
    % matrix G, or a MATLAB graph G or digraph.  The matrix G may be a
    % GraphBLAS matrix or any MATLAB matrix (sparse or full).
    %
    % Usage:
    %
    %   H = gbgraph (G) ;
    %   H = gbgraph (G, 'undirected') ;
    %   H = gbgraph (G, 'directed') ;
    %
    % With an optional second argument, the kind of graph can be specified.
    % The default is to construct an undirected graph if G is symmetric, or
    % a directed graph otherwise.  When constructing an undirected graph,
    % G must be symmetric.
    %
    % See also graph, digraph, gb.

    if (nargin < 3)
        % for internal use only; the default is to always check the inputs
        check = true ;
    end

    if (check)

        % check the kind input parameter
        if (nargin == 2)
            if (~ (isequal (kind, 'undirected') | isequal (kind, 'directed')))
                error ('unknown kind') ;
            end
        end

        if (isa (G, 'graph'))

            % C = gbgraph (G, ...) where G is a MATLAB undirected graph
            G = adjacency (G, 'weighted') ;
            if (nargin < 2)
                % C will be an undirected gbgraph
                kind = 'undirected' ;
            else
                % C will be an directed or undirected gbgraph, per kind input
                ;
            end

        elseif (isa (G, 'digraph'))

            % C = gbgraph (G, ...) where G is a MATLAB directed graph
            G = adjacency (G, 'weighted') ;
            if (nargin < 2)
                % C will be a directed gbgraph
                kind = 'directed' ;
            else
                % C will be an directed or undirected gbgraph, per kind input
                if (isequal (kind, 'undirected'))
                    % if converting from a digraph to an undirected gbgraph, G
                    % must be symmetric.
                    if (~issymmetric (G))
                        error ('G must be symmetric') ;
                    end
                end
            end

        elseif (isa (G, 'gbgraph'))

            % C = gbgraph (G, ...) where G is gbgraph.
            if (nargin < 2)
                % C will be an undirected or directed gbgraph, the same as G
                kind = G.graphkind ;
            else
                % C will be an directed or undirected gbgraph, per kind input
                if (isequal (kind, 'undirected') & ...
                    isequal (G.graphkind, 'directed'))
                    % if converting from a directed gbgraph to an undirected
                    % gbgraph, G must be symmetric.
                    if (~issymmetric (G))
                        error ('G must be symmetric') ;
                    end
                end
            end

        else

            % C = gbgraph (G) where G is a matrix (GraphBLAS or MATLAB)
            [m n] = size (G) ;
            if (m ~= n)
                error ('G must be square') ;
            end
            if (nargin < 2)
                % C = gbgraph (G)
                if (issymmetric (G))
                    % If G is symmetric, construct an undirected graph.
                    kind = 'undirected' ;
                else
                    % otherwise, construct a directed graph
                    kind = 'directed' ;
                end
            else
                % C = gbgraph (G, kind) ;
                % Construct C according to the kind input.  For this case,
                % if C is an undirected graph then G must be symmetric.
                if (isequal (kind, 'undirected') & ~issymmetric (G))
                    error ('G must be symmetric') ;
                end
            end

        end
    end

    % create the graph
    H = H@gb (G) ;
    H.graphkind = kind ;
    end

    %---------------------------------------------------------------------------
    % overloaded methods
    %---------------------------------------------------------------------------

    % methods in @gbgraph that overload the MATLAB graph and digraph methods:

    d = degree (H) ;
    C = digraph (H) ;
    display (H) ;
    disp (H, level) ;
    c = edgecount (H, s, t) ;
    C = graph (H, uplo) ;
    d = indegree (H) ;
    L = laplacian (H, type) ;
    [e d] = numedges (H) ;
    n = numnodes (H) ;
    d = outdegree (H) ;
    [handle titlehandle] = plot (H, varargin) ;
    S = subgraph (H, I) ;
    C = flipedge (H, varargin) ;
    s = ismultigraph (H) ;
    G = adjacency (H, arg) ;
    G = incidence (H, type) ;
    C = reordernodes (H, order) ;

    % FUTURE::
    %
    % methods for both classes graph and digraph not yet implemented:
    %
    %    addedge addnode bfsearch centrality conncomp dfsearch distances
    %    findedge findnode isisomorphic isomorphism maxflow nearest outedges
    %    rmedge rmnode shortestpath shortestpathtree simplify
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

    k = kind (H) ;

    % FUTURE::
    %
    %   tricount, ktruss, dnn, ... (see LAGraph)

end

end

