classdef gbgraph < gb

    % methods for both classes graph and digraph not implemented here:
    %
    %    addedge addnode adjacency bfsearch centrality conncomp dfsearch
    %    distances findedge findnode incidence isisomorphic ismultigraph
    %    isomorphism maxflow nearest numedges numnodes outedges plot
    %    reordernodes rmedge rmnode shortestpath shortestpathtree simplify

    % methods for class graph (not in digraph class) not implemented here:
    %
    %    bctree biconncomp minspantree neighbors

    % methods for class digraph (not in graph class) not implemented here:
    %
    %    condensation flipedge inedges isdag predecessors successors toposort
    %    transclosure transreduction

properties
    graphkind = [ ] ;    % 'graph' or 'digraph'
end

methods

    function H = gbgraph (G, graphkind)
    %
    %   H = gbgraph (G) ;
    %   H = gbgraph (G, graphkind) ;

    if (nargin < 1)
        error ('not enough input arguments') ;
    end
    [m n] = size (G) ;
    if (m ~= n)
        error ('G must be square') ;
    end
    if (nargin < 2)
        if (issymmetric (G))
            graphkind = 'graph' ;
        else
            graphkind = 'digraph' ;
        end
    else

        if (isequal (graphkind, 'graph'))
            if (~issymmetric (G))
                error ('G must be symmetric') ;
            end
        elseif (isequal (graphkind, 'digraph'))
            ;
        else
            error ('unknown graphkind') ;
        end

    end

    H@gb (G) ;
    H.graphkind = graphkind ;
    end

end


methods (Static)

    function k = kind (H)
    k = H.graphkind ;
    end

    function H = setkind (H, kind)
    %SETKIND change the kind of the graph H.

    if (isequal (graphkind, 'graph'))
        if (~issymmetric (H))
            error ('H must be symmetric') ;
        end
    elseif (isequal (graphkind, 'digraph'))
        ;
    else
        error ('unknown graphkind') ;
    end

    H.graphkind = kind ;
    end

end

end

