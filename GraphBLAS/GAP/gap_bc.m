function gap_bc
%GAP_BC run centrality for the GAP benchmark

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

diary on
rng ('default') ;

% warmup, to make sure GrB library is loaded
C = GrB (1) * GrB (1) + 1 ;
clear C

index = ssget ;
f = find (index.nrows == index.ncols & index.nnz > 5e6 & index.isReal) ;
[~,i] = sort (index.nnz (f)) ;
matrices = f (i) ;

% smaller test matrices:
matrices = { 'HB/west0067', 'SNAP/roadNet-CA', ...
    'GAP/GAP-road', ...
    'GAP/GAP-web', ...
    'GAP/GAP-urand', ...
    'GAP/GAP-twitter', ...
    'GAP/GAP-kron' }

matrices = { 'HB/west0067', 'SNAP/roadNet-CA' , ...
    'SNAP/com-Orkut', 'LAW/indochina-2004' }

% the GAP test matrices:
matrices = {
    'GAP/GAP-kron'
    'GAP/GAP-urand'
    'GAP/GAP-twitter'
    'GAP/GAP-web'
    'GAP/GAP-road'
    } ;

[status, result] = system ('hostname') ;
clear status
if (isequal (result (1:5), 'hyper'))
    fprintf ('hypersparse: %d threads\n', GrB.threads (40)) ;
elseif (isequal (result (1:5), 'slash'))
    fprintf ('slash: %d threads\n', GrB.threads (8)) ;
else
    fprintf ('default: %d threads\n', GrB.threads) ;
end
clear result

for k = 1:length(matrices)

    %---------------------------------------------------------------------------
    % get the GAP problem
    %---------------------------------------------------------------------------

try

    id = matrices (k) ;
    GrB.burble (0) ;
    t1 = tic ;
    clear A Prob
    Prob = ssget (id, index) ;
    sources = Prob.aux.sources ;
    A = GrB (Prob.A, 'by row', 'logical') ;
    name = Prob.name ;
    clear Prob
    A = spones (A) ;
    AT = A' ;
    n = size (A,1) ;
    fprintf ('\n%s: nodes: %g million  nvals: %g million\n', ...
        name, n / 1e6, nnz (A) / 1e6) ;
    t1 = toc (t1) ;
    fprintf ('load time: %g sec\n', t1) ;

    %---------------------------------------------------------------------------
    % compute the centrality for each batch of 4
    %---------------------------------------------------------------------------

    fprintf ('\ngap_centrality  tests:\n') ;

    good = '~/LAGraph/Test/BetweennessCentrality/batch_%02d_%d.mtx' ;

    tot = 0 ;
    for k = 1:4:length(sources)
        src = sources (k:k+4) ;

        tstart = tic ;
        c = gap_centrality (src, A, AT) ;
        t = toc (tstart) ;
        tot = tot + t ;
        fprintf ('trial: %2d GrB centrality time: %8.3f\n', trial, t) ;

        % check result
        cgood = load (sprintf (good, k-1, n)) ;
        err = norm (cgood - c) ;
        fprintf ('err: %g\n', err) ;
    end

    fprintf ('avg GrB centrality time:  %10.3f (%d trials)\n', ...
        tot/ntrials, ntrials) ;

    diary off
    diary on

catch me
    k
    disp (me.message)
end

end


