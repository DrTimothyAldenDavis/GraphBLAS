function [v, parent] = bfs (G, s)
%BFS breadth-first search of a gbgraph.
% v = bfs (G, s) computes the breadth-first search of the gbgraph G.  The
% breadth-first search starts at node s.  The output v is a sparse vector of
% size n-by-1, with the level of each node, where v(s)=1, and v(i)=k if the
% path with the fewest edges from from s to i has k-1 edges.  If i is not
% reachable from s, then v(i) is implicitly zero and does not appear in the
% pattern of v.
%
% [v, parent] = bfs (G, s) also computes the parent vector, representing the
% breadth-first search tree.  parent(s)=s denotes the root of the tree, and
% parent(c)=p if node p is the parent of c in the tree.  The parent vector is
% sparse, and parent (i) is not present if i is not found in the breadth-first
% search.
%
% For best performance, if G is a directed graph, it should be stored by row.
% If G is an undirected graph, then it can be stored in either format.  The
% edge weights of the graph G are ignored. 
%
% See also graph/bfsearch, graph/shortestpathtree.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

d = struct ('out', 'replace', 'mask', 'complement') ;

% determine the method to use, and convert G if necessary
if (isundirected (G))
    if (gb.isbycol (G))
        % G is stored by column but undirected, so use q*G' instead of q*G
        d.in1 = 'transpose' ;
    end
else
    if (gb.isbycol (G))
        % this can be costly
        G = gbgraph (G, 'by row') ;
    end
end

% determine the integer type to use, and initialize v as a full vector
n = numnodes (G) ;
if (n < intmax ('int32'))
    int_type = 'int32' ;
else
    int_type = 'int64' ;
end
v = full (gb (1, n, int_type)) ;

if (nargout == 1)

    % just compute the level of each node
    q = gb (1, n, 'logical') ;
    q (s) = 1 ;

    for level = 1:n
        % assign the current level: v<q> = level
        v (q) = level ;
        if (~any (q))
            % quit if q is empty
            break ;
        end
        % move to the next level:  q<~v,replace> = q*G,
        % using the boolean semiring
        q = gb.mxm (q, v, '|.&.logical', q, G, d) ;
    end

else

    % compute both the level and the parent
    parent = full (gb (1, n, int_type)) ;
    parent (s) = s ;    % denotes the root of the tree
    q = gb (1, n, int_type) ;
    q (s) = s ;
    id = gb (1:n, int_type) ;

    for level = 1:n
        % assign the current level: v<q> = level
        v = gb.assign (v, q, level) ;
        if (~any (q))
            % quit if q is empty
            break ;
        end
        % move to the next level:  q<~v,replace> = q*G,
        % using the min-first semiring
        q = gb.mxm (q, v, 'min.1st', q, G, d) ;
        % assign parents
        parent = gb.assign (parent, q, q) ;
        % q(i) = i for all entries in q
        q = gb.assign (q, q, id) ;
    end

    % remove zeros from parent
    parent = gb.prune (parent) ;

end

% remove zeros from v
v = gb.prune (v) ;

