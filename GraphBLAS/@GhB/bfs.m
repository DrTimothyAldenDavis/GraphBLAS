function [out1, out2] = bfs (A, AT, degree, s, varargin)
%GHB.BFS breadth-first search of a graph, using its adjacency matrix.
% v = GhB.bfs (A, s) performs the breadth-first search of the directed
% graph represented by the square adjacency matrix A.  The breadth-first
% search starts at node s.  The output v is a sparse vector of size n-by-1,
% with the level of each node, where v(s)=1, and v(i)=k if the path with
% the fewest edges from from s to i has k-1 edges.  If i is not reachable
% from s, then v(i) is implicitly zero and does not appear in the pattern
% of v.
%
% [v, parent] = GhB.bfs (A, s) also computes the parent vector,
% representing the breadth-first search tree.  parent(s)=s denotes the root
% of the tree, and parent(c)=p if node p is the parent of c in the tree.
% The parent vector is sparse, and parent (i) is not present if i is not
% found in the breadth-first search.
%
% Optional string arguments can be provided, after A and s:
%
%   'undirected' or 'symmetric':  A is assumed to be symmetric, and
%       represents an undirected graph.  Results are undefined if A is
%       unsymmetric, and 'check' is not specified.
%
%   'directed' or 'unsymmetric':  A is assumed to be unsymmetric, and
%       presents a directed graph.  This is the default.
%
%   'check': with the 'undirected' or 'symmetric' option, A is checked to
%       ensure that it is symmetric.  The default is not to check.
%
% A must be square.  Only the pattern, spones (A), is considered; the values of
% its entries (the edge weights of the graph) are ignored.  A and AT (if
% provided) must be held by row; that is, GhB.format (A) must report 'by row'.
%
%   [v, parent] = GhB.bfs (A, s, AT, degree, ...)
%
% Example:
%
%   A = bucky ;
%   s = 1 ;
%   [v pi] = GhB.bfs (A, s)
%   figure (1) ;
%   subplot (1,2,1) ; plot (graph (A)) ;
%   pi2 = full (double (pi)) ;
%   pi2 (s) = 0 ;
%   subplot (1,2,2) ; treeplot (pi2) ; title ('BFS tree') ;
%   n = size (A,1) ;
%   for level = 1:n
%       level
%       inlevel = find (v == level)
%       parents = full (double (pi (inlevel)))
%       if (isempty (inlevel))
%           break ;
%       end
%   end
%
% See also graph/bfsearch, graph/shortestpathtree, treeplot.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

%-------------------------------------------------------------------------------
% get inputs
%-------------------------------------------------------------------------------

narginchk (2, 6) ;

if (~isscalar (s) || GhB.nvals (s) ~= 1)
    error ('GrB:error', 'source node s must be a scalar') ;
end

[m, n] = size (A) ;
if (m ~= n)
    error ('GrB:error', 'A must be square') ;
end
[m2, n2] = size (AT) ;
if (n ~= n2 || n ~= m2)
    error ('GrB:error', 'AT has the wrong size') ;
end
bycol = GhB.isbycol (A) ;
if (bycol ~= GhB.isbycol (AT))
    error ('GrB:error', 'A and AT must the same format (by row or by col)') ;
end

% get the string arguments
kind = 'directed' ;
check = false ;
nvar = length (varargin) ;
compute_parent = (nargout == 2) ;
monoid = 'any' ;
for k = 1:nvar
    arg = varargin {k} ;
    if (ischar (arg))
        arg = lower (arg) ;
        switch arg
            case { 'undirected', 'symmetric' }
                kind = 'undirected' ;
            case { 'directed', 'unsymmetric' }
                kind = 'directed' ;
            case { 'parent', 'anyparent' }
                % use the 'any' monoid which is fast but non-deterministic
                compute_parent = true ;
            case { 'minparent' }
                monoid = 'min' ;
                compute_parent = true ;
            case { 'maxparent' }
                monoid = 'max' ;
                compute_parent = true ;
            case { 'check' }
                check = true ;
            otherwise
                error ('GrB:error', 'unknown option') ;
        end
    end
end

compute_level = (nargout == 1 && ~compute_parent) || (nargout == 2) ;

% extensive checks, if requested
if (check)
    if (~isequal (A, AT'))
        error ('GrB:error', 'A must equal AT''') ;
    end
    if (isequal (kind, 'undirected') && ~issymmetric (A))
        error ('GrB:error', 'A must be symmetric') ;
    end
    if (~isequal (degree, GhB.entries (A, 'row', 'degree')))
        error ('GrB:error', 'degree is incorrect') ;
    end
end

%-------------------------------------------------------------------------------
% initializations
%-------------------------------------------------------------------------------

% create the descriptors

desc_s.mask = 'structural' ;

desc_rs.out = 'replace' ;
desc_rs.mask = 'structural' ;

desc_rsc.out = 'replace' ;
desc_rsc.mask = 'structural complement' ;

desc_rsct.out  = 'replace' ;
desc_rsct.mask = 'structural complement' ;
desc_rsct.in0 = 'transpose' ;

% determine the integer type to use
int_type = 'int64' ;
if (n < intmax ('int32'))
    int_type = 'int32' ;
end

do_push = true ;                % start with a push
last_nq = 0 ;                   % # nodes in prior frontier
nq = 1 ;                        % # nodes in the current frontier
unexplored = GhB.nvals (A) ;    % # of edges not yet explored via pull
any_pull = false ;              % true if any pull has ever been done

if (compute_parent)
    parent = GhB (n, 1, int_type, 'bitmap/full') ;  % parent = sparse (n,1)
    GhB.subassign (parent, { s }, s) ;              % parent (s) = s
end

if (compute_level)
    v = GhB (n, 1, int_type, 'bitmap/full') ;       % v = sparse (n,1)
    GhB.assign (v, { s }, 1) ;                      % v (s) = 1
end

if (compute_level && ~compute_parent)
    % just compute the level
    q = GhB (n, 1, 'logical') ;     % set frontier, q = sparse (n,1)
    GhB.assign (q, { s }, true) ;   % q (s) = 1
    semiring = 'any.pair.logical' ; % deterministic, when just computing level
    mask = v ;                      % use v as the mask below
else
    % compute just the parent, or both level and parent
    q = GhB (n, 1, int_type) ;      % q = sparse (n,1)
    GhB.subassign (q, { s }, s) ;   % q (s) = s ; (source s is its own parent)
    semiring = [monoid '.secondi1.' int_type] ;
    mask = parent ;                 % use v as the mask below
end

% push/pull parameters
alpha = 8 ;
beta1 = 8 ;
beta2 = 512 ;

%-------------------------------------------------------------------------------
% do the BFS
%-------------------------------------------------------------------------------

for level = 2:n

    %---------------------------------------------------------------------------
    % decide to push or pull
    %---------------------------------------------------------------------------

    if (do_push)
        % check for switch from push to pull
        growing = (nq > last_nq) ;
        switch_to_pull = false ;
        if (unexplored < n)
            % very little of the graph is left; disable the pull
            push_pull = false ;
        elseif (any_pull)
            % at least one pull has been done already; no longer keeping track
            % of the # unexplored nodes
            switch_to_pull = (growing && nq > (n/beta1)) ;
        else
            % count the # of outgoing edges from the current frontier, q.
            % only do this if no pull has yet occured.
            % w(i) = outdegree of node i if it is in the queue, as w<q> = degree
            w = GhB (n, 1, int_type) ;
            GhB.assign (w, q, degree, desc_rs) ;
            edges_in_frontier = double (sum (w)) ;
            unexplored = unexplored - edges_in_frontier ;
            switch_to_pull = growing && ...
                (edges_in_frontier > (unexplored / alpha)) ;
        end
        if (switch_to_pull)
            % switch from push to pull
            do_push = false ;
        end
    else
        % check for switch from pull to push
        shrinking = (nq < last_nq) ;
        if (shrinking && (nq <= (n / beta2)))
            % switch from pull to push
            do_push = true ;
        end
    end
    if (~do_push)
        % at least one pull has been done
        any_pull = true ;
    end
    last_nq = nq ;

    %---------------------------------------------------------------------------
    % convert q to the right format for push/pull
    %---------------------------------------------------------------------------

    [~,sparsity,~] = GhB.format (q) ;
    if (do_push && (~isequal (sparsity, 'sparse')))
        q = GhB (q, 'sparse') ;
    elseif (~do_push && (~isequal (sparsity, 'bitmap')))
        q = GhB (q, 'bitmap') ;
    end

    %---------------------------------------------------------------------------
    % q<~v,replace> = A'*q
    %---------------------------------------------------------------------------

    if (do_push == bycol)
        % push if A,AT are held by col; pull if A,AT are held by row
        GhB.mxm (q, mask, semiring, AT, q, desc_rsc) ;
    else
        % pull if A,AT are held by col; push if A,AT are held by row
        GhB.mxm (q, mask, semiring, A, q, desc_rsct) ;
    end

    %---------------------------------------------------------------------------
    % quit if q is empty
    %---------------------------------------------------------------------------

    nq = GhB.nvals (q) ;
    if (nq == 0), break, end

    %---------------------------------------------------------------------------
    % update parent and level
    %---------------------------------------------------------------------------

    if (compute_parent)
        % assign parents: parent<q> = q
        GhB.assign (parent, q, q, desc_s) ;
        done = (GhB.nvals (parent) == n) ;
    end

    if (compute_level)
        % assign the current level: v<q> = level
        GhB.subassign (v, q, level, desc_s) ;
        done = (GhB.nvals (v) == n) ;
    end

    %---------------------------------------------------------------------------
    % quit if all nodes have been reached
    %---------------------------------------------------------------------------

    if (done), break, end

end

%-------------------------------------------------------------------------------
% return results
%-------------------------------------------------------------------------------

if (compute_level && compute_parent)
    out1 = v ;
    out2 = parent ;
elseif (compute_level)
    out1 = v ;
else
    out1 = parent ;
end

