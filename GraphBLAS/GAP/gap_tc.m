%-------------------------------------------------------------------------------
% gap_tc: run tricount for the GAP benchmark
%-------------------------------------------------------------------------------

clear
rng ('default') ;

% warmup, to make sure GrB library is loaded
C = GrB (1) * GrB (1) + 1 ;
clear C

% the GAP test matrices:
matrices = {
    'GAP/GAP-road'
    'GAP/GAP-web'
    'GAP/GAP-urand'
    'GAP/GAP-twitter'
    'GAP/GAP-kron'
    } ;

matrices = { 'HB/west0067', 'SNAP/roadNet-CA' } ;
    % 'SNAP/com-Orkut', 'LAW/indochina-2004', ...

% smaller test matrices:
matrices = { 'HB/west0067', 'SNAP/roadNet-CA', ...
    'GAP/GAP-road', ...
    'GAP/GAP-web', ...
    'GAP/GAP-urand', ...
    'GAP/GAP-twitter', ...
    'GAP/GAP-kron' }

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

    GrB.burble (0) ;
    t1 = tic ;
    clear A Prob
    Prob = ssget (matrices {k}) ;
    A = GrB (Prob.A, 'by row', 'logical') ;
    name = Prob.name ;
    clear Prob
    A = A|A' ;
    n = size (A,1) ;
    fprintf ('\n%s: nodes: %g million  nvals: %g million\n', ...
        name, n / 1e6, nnz (A) / 1e6) ;
    t1 = toc (t1) ;
    fprintf ('load time: %g sec\n', t1) ;

    ntrials = 1 ; % TODO 3 ;

    %---------------------------------------------------------------------------
    % triangle count
    %---------------------------------------------------------------------------

    fprintf ('\nGrB.tricount  tests:\n') ;

    tot = 0 ;
    for trial = 1:ntrials
        tstart = tic ;
        s = GrB.tricount (A) ;
        t = toc (tstart) ;
        tot = tot + t ;
        fprintf ('trial: %2d GrB.tricount  time: %8.3f\n', trial, t) ;
    end
    fprintf ('avg GrB.tricount time:  %10.3f (%d trials)\n', ...
        tot/ntrials, ntrials) ;
    fprintf ('% triangles: %d\n', full (s)) ;

    %---------------------------------------------------------------------------
    % triangle count with permutations
    %---------------------------------------------------------------------------

    [c times best] = tric (A, s) ;

    clear A
end

