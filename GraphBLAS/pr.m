
% A = mread ('cover.mtx') ;
clear all
% gb.threads (8)
% gb.chunk (1024*1024)

%%MatrixMarket matrix coordinate pattern general
%%GraphBLAS GrB_BOOL
% Matrix from the cover of "Graph Algorithms in the Language of Linear
% Algebra", Kepner and Gilbert.  Note that cover shows A'.  This is A.
% 7 7 12
ij = [
4 1
1 2
4 3
6 3
7 3
1 4
7 4
2 5
7 5
3 6
5 6
2 7 ] ;


source = 1 ;

A = sparse (ij (:,1), ij (:,2), ones (12,1), 7, 7) ;
G = gbgraph (A, 'by col') 

H = digraph (A)

c1 = centrality (H, 'pagerank')
% c2 = pagerank (G)


tol = 1e-4  ;
maxit = 100 ;
damp = 0.85 ;
weighted = false ;

n = numnodes (G) ;
d = full (double (outdegree (G))) ;

if (weighted)
    % use the weighted edges of G, but typecast to double first
    G = gb (G, 'double') ;
else
    % use the pattern of G
    G = spones (G, 'double') ;
end

G
d

% nodes with no out-going edges:
sinks = find (d == 0) ;
d (sinks) = 1 ;

% initial PageRank
r = ones (n, 1) / n ;
r

if (length (sinks) > 0)

    % compute the PageRank
    for iter = 1:maxit
        rold = r ;
        % r = damp*G*(r ./ d) + ((damp / n) * sum (r (sinks)) + (1-damp) / n) ;
        r = damp*G*(r./d) + damp/n*sum(r(sinks)) + (1 - damp)/n;
        if (norm (r - rold, inf) < tol)
            break ;
        end
    end

else

    % compute the PageRank, no sinks
    for iter = 1:maxit
        rold = r ;
        % r = damp*G*(r ./ d) + ((damp / n) * sum (r (sinks)) + (1-damp) / n) ;
        r = damp*G'*(r./d) + (1 - damp)/n;
        r

        c = full (double (rold)) ;
        c = damp*G'*(c./d) + (1 - damp)/n;
        c
        r-c
        pause
        if (norm (r - rold, inf) < tol)
            break ;
        end
    end

        % Iterative computation
        cnew = ones(n, 1)/n;
        for ii=1:maxit
            c = cnew;
            cnew = damp*G*(c./d) + damp/n*sum(c(sinks)) + (1 - damp)/n;
            cnew
            pause
            if norm(c - cnew, inf) <= tol
                break;
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


