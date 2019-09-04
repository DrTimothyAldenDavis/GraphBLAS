function r = pagerank (G, opts)
%PAGERANK PageRank of a gbgraph.
% r = pagerank (G) computes the PageRank of a gbgraph G.
% r = pagerank (G, options) allows for non-default options to be selected.
% Defaults are identical to the MATLAB pagerank method in @graph/centrality:
%
%   opts.tol = 1e-4         stopping criterion
%   opts.maxit = 100        maximum # of iterations to take
%   opts.damp = 0.85        dampening factor
%   opts.weighted = false   true: use edgeweights of G; false: use spones(G)

% check inputs and get options
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

tol = opts.tol ;
maxit = opts.maxit ;
damp = opts.damp ;

n = numnodes (G) ;
d = full (double (outdegree (G))) ;

if (opts.weighted)
    % use the weighted edges of G, but typecast to double first
    G = gb (G, 'double', 'by col') ;
else
    % use the pattern of G
    G = spones (G, 'double', 'by col') ;
end

G
d

% nodes with no out-going edges:
sinks = find (d == 0) ;
d (sinks) = 1 ;

teleport = (1 - damp) / n ;

desc.in0 = 'transpose' ;

% initial PageRank
r = ones (n, 1) / n ;

if (length (sinks) > 0)

    % compute the PageRank
    for iter = 1:maxit
        rold = r ;
        % r = damp * G * (r ./ d) + ((damp / n) * sum (r (sinks)) + (1-damp) / n) ;
        sum (r (sinks))

        r = damp* gb.mxm ('+.*', G, r./d, desc) + damp/n*sum(r(sinks)) + teleport ;
        if (norm (r - rold, inf) < tol)
            break ;
        end
    end

else

    % compute the PageRank, no sinks
    for iter = 1:maxit
        rold = r ;
        % r = damp * G * (r ./ d) + ((damp / n) * sum (r (sinks)) + (1-damp) / n) ;
        r = damp* gb.mxm ('+.*', G, r./d, desc) + teleport ;
        if (norm (r - rold, inf) < tol)
            break ;
        end
    end

end

%{
        cnew = ones(n, 1)/n;
        for ii=1:maxit
            c = cnew;
            cnew = damp*A*(c./d) + damp/n*sum(c(snks)) + (1 - damp)/n;
            if norm(c - cnew, inf) <= tol
                break;
            end
        end
        
        if ~(norm(c - cnew, inf) <= tol)
            warning(message('MATLAB:graphfun:centrality:PageRankNoConv'));
        end
        c = cnew;
        
end
%}
