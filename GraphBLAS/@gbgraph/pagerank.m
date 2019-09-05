function r = pagerank (G, opts)
%PAGERANK PageRank of a gbgraph.
% r = pagerank (G) computes the PageRank of a gbgraph G.
% r = pagerank (G, options) allows for non-default options to be selected.
% For compatibility with MATLAB, defaults are identical to the MATLAB pagerank
% method in @graph/centrality and @digraph/centrality:
%
%   opts.tol = 1e-4         stopping criterion
%   opts.maxit = 100        maximum # of iterations to take
%   opts.damp = 0.85        dampening factor
%   opts.weighted = false   true: use edgeweights of G; false: use spones(G)
%   opts.type = 'double'    compute in 'single' or 'double' precision
%
% G can be stored by row or by column, but storing it by column is faster
% for computing the pagerank.

% check inputs and set defaults
if (nargin < 2)
    opts = struct ;
end
if (~isfield (opts, 'tol'))
    opts.tol = 1e-4 ;
end
if (~isfield (opts, 'maxit'))
    opts.maxit = 100 ;
end
if (~isfield (opts, 'damp'))
    opts.damp = 0.85 ;
end
if (~isfield (opts, 'weighted'))
    opts.weighted = false ;
end
if (~isfield (opts, 'type'))
    opts.type = 'double' ;
end

% get options
tol = opts.tol ;
maxit = opts.maxit ;
damp = opts.damp ;
type = opts.type ;
weighted = opts.weighted ;

% make sure G is stored by column, and of the right type
n = numnodes (G) ;
if (weighted)
    % use the weighted edges of G
    d = outdegree (G) ;
    G = gb (G, type, 'by col') ;
else
    % use the pattern of G 
    G = bycol (G) ;
    G = spones (G, type) ;
    d = gb.vreduce ('+', G) ;
end

% G is now a gb matrix, and no longer a gbgraph

d = full (cast (d, type)) ;

% d (i) = outdegree of node i, or 1 if i is a sink
sinks = find (d == 0) ;
any_sinks = length (sinks) > 0 ;
if (any_sinks > 0)
    d (sinks) = 1 ;
end

% place explicit zeros on the diagonal of G so that r remains full
G = G + gb.build (1:n, 1:n, zeros (n, 1, type), n, n) ;

% teleport factor
tfactor = cast ((1 - damp) / n, type) ;

% sink factor
dn = cast (damp / n, type) ;

% use G' in gb.mxm, and return the result as a MATLAB full vector
desc.in0 = 'transpose' ;
desc.kind = 'full' ;

% initial PageRank: all nodes have rank 1/n
r = ones (n, 1, type) / n ;

% prescale d with damp so it doesn't have to be done in each iteration
d = d / damp ;

% compute the PageRank
for iter = 1:maxit
    rold = r ;
    teleport = tfactor ;
    if (any_sinks)
        % add the teleport factor from all the sinks
        teleport = teleport + dn * sum (r (sinks)) ;
    end
    % r = damp * G' * (r./d) + teleport
    r = r ./ d ;
    r = gb.mxm ('+.*', G, r, desc) ;
    r = r + teleport ;
    if (norm (r - rold, inf) < tol)
        % convergence has been reached
        return ;
    end
end

warning ('gbgraph:pagerank', 'pagerank failed to converge') ;

