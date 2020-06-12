function mxm_demo
%MXM_DEMO performance test of real and complex A*B
% Requires the ssget interface to the SuiteSparse Matrix Collection.
% See https://sparse.tamu.edu.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved.  http://suitesparse.com   See GraphBLAS/Doc/License.txt.

ncores = feature ('numcores') ;
rng ('default') ;
Prob = ssget (936)
% GrB.burble (true) ;

try
    system ('hostname') ;
catch
end
v = ver ('matlab') ;
fprintf ('MATLAB version: %s release: %s\n', v.Version, v.Release) ;

% warmup
G = GrB (1) ;
G = G*G ;
clear G

for nth = [1 ncores 2*ncores]

    % tell MATLAB and GraphBLAS to use nth threads:
    maxNumCompThreads (nth) ;
    GrB.threads (nth) ;

    fprintf ('\n-------------------------------------------------\n') ;
    if (nth == 1)
        fprintf ('Testing single-threaded performance of C=A*B:\n') ;
    else
        fprintf ('Testing performance of C=A*B using %d threads:\n', nth) ;
    end
    fprintf ('-------------------------------------------------\n') ;

    types = { 'single', 'double', 'single complex', 'double complex' } ;

    for k = 1:4

        type = types {k} ;

        % GrB.random (A, ...) is like sprand (A), but it allows
        % the range of values to be specified, as an array of size 2.
        % The output matrix has the same type as the range parameter.
        switch (type)
            case 'single'
                fprintf ('\n=== MATLAB: double (real) ') ;
                range = single ([-1 1]) ;
            case 'double'
                fprintf ('\n=== MATLAB: double (real) ') ;
                range = double ([-1 1]) ;
            case 'single complex'
                fprintf ('\n=== MATLAB: double complex ') ;
                range = complex (single ([-1 1])) ;
            case 'double complex'
                fprintf ('\n=== MATLAB: double complex ') ;
                range = complex (double ([-1 1])) ;
        end

        GA = GrB.random (Prob.A, 'range', range) ;
        GB = GrB.random (Prob.A, 'range', range) ;
        fprintf ('vs GraphBLAS: %s\n', GrB.type (GA)) ;

        % create MATLAB versions of GA and GB.  The overloaded "double"
        % function converts GA and GB to double or double complex,
        % just like the built-in.  MATLAB doesn't have sparse 'single'
        % or 'single complex', so all of these test use 'double' or
        % 'double complex'.
        A = double (GA) ;
        B = double (GB) ;

        tm_total = 0 ;
        tg_total = 0 ;
        ntrials = 4 ;

        for trial = 1:ntrials
            tic
            C1 = A*B ;
            tm = toc ;
            tic
            C2 = GA*GB ;
            tg = toc ;
            err = norm (C1-C2,1) / norm (C1, 1) ;
            fprintf ('trial %d: MATLAB: %10.4f GrB: %10.4f', trial, tm, tg);
            fprintf (' speedup: %10.2f err: %g\n', tm / tg, err) ;
            tm_total = tm_total + tm ;
            tg_total = tg_total + tg ;
        end

        tm = tm_total / ntrials ;
        tg = tg_total / ntrials ;

        fprintf ('average: MATLAB: %10.4f GrB: %10.4f', tm, tg) ;
        fprintf (' speedup: %10.2f\n', tm / tg) ;
    end
end

% restore # of threads to their defaults
maxNumCompThreads (ncores) ;
GrB.clear ;

