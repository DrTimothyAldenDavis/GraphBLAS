function [v, parent] = bfs_pushpull (G_byrow, G_bycol, s)
%BFSPUSHPULL breadth-first search of a gbgraph.
% [v, parent] = bfs_pushpull (G_byrow, G_bycol, s)
%
% in progress ...

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

d = struct ('out', 'replace', 'mask', 'complement') ;

if (gb.isbycol (G_byrow) || gb.isbyrow (G_bycol))
    error ('G_byrow must be stored by row and G_bycol by col') ;
end

% determine the integer type to use, and initialize v as a full vector
n = numnodes (G_byrow) ;
if (n < intmax ('int32'))
    int_type = 'int32' ;
else
    int_type = 'int64' ;
end
v = full (gb (1, n, int_type)) ;

% average out-degree:
deg = full (double (sum (outdegree (G_byrow)))) / n ;
nvisited = 0 ;

% if (nargout == 1)

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

        % select push/pull method
        nq = nnz (q) ;
        nvisited = nvisited + nq ;
        pushwork = deg * nq ;
        expected = n / nvisited ;
        per_dot = min (deg, expected) ;
        binarysearch = 3 * (1 + ceil (log2 (nq)));
        pullwork = (n - nvisited) * per_dot * binarysearch ;

        fprintf ('level %12d nq %12d visited %12d pullwork %10.4e pushwork %10.4e : ', ...
            level, nq, nvisited, pullwork, pushwork) ;

        if (pushwork < pullwork)
            % push step to move to the next level:  q<~v,replace> = q*G
            % using the boolean semiring
            fprintf ('push\n') ;
            q = gb.mxm (q, v, '|.&.logical', q, G_byrow, d) ;
        else
            % pull step: 
            fprintf ('pull\n') ;
            q = gb.mxm (q, v, '|.&.logical', q, G_bycol, d) ;
        end
    end

%{
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
%}

% remove zeros from v
v = gb.prune (v) ;

