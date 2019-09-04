
clear all
C = gb.mxm ('+.*', 1,1) ;  % warmup
fprintf ('# of threads in GraphBLAS: %d\n', gb.threads) ;

f = [
2459        % DIMACS10/road_usa (the GAP "Road" matrix)
2454        % LAW/sk-2005 (the GAP "web" matrix)
2796        % SNAP/twitter7 (the GAP "twitter" matrix)
] ;

root = '/raid/hyper/gapbs/benchmark/out/' ;
source_files = {
'bfs-road_source.txt'
'bfs-web_source.txt'
'bfs-twitter_source.txt'
} ;
% also:
% bfs-kron_source.txt
% bfs-urand_source.txt


for k = 2:3
    id = f (k) ;
    Prob = ssget (id)
    A = Prob.A ;
    A = spones (A) ;
    G = gbgraph (A+A', 'undirected') ;

    tic ;
    c = tricount (G) ;
    t = toc ;
    fprintf ('Triangle count: %g  time: %g\n', c, t) ;

    G_row = gbgraph (A, 'logical', 'directed', 'by row') ;
    G_col = gbgraph (A, 'logical', 'directed', 'by col') ;

    % get the BFS sources
    sources = load ([root source_files{k}]) ;
    ttotal = 0 ;
    ntrials = 1 % length (sources) ;
    for i = 1:ntrials
        s = sources (i) ;

        if (k == 3)
            % twitter is relabelled in the SuiteSparse Collection
            s = find (Prob.aux.nodeid == s) ;
        end

        tic ;
        % [v, parent] = bfs (G, s) ;
        v = bfs_pushpull (G_row, G_col, s) ;
        t = toc ;
        ttotal = ttotal + t ;

        depth = full (double (max (v))) ;

        fprintf ('Source: %12d reach %12d depth %12d time %g\n', ...
            s, length (v), depth, t) ;
    end
    fprintf ('Avg BFS time %g\n', ttotal / ntrials) ;

end

