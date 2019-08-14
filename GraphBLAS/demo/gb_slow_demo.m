function gb_slow_demo (bnz)
%GB_SLOW_DEMO Extreme performance differences between GraphBLAS and MATLAB.
%
% Usage:
%       gb_slow_demo                % uses a default bnz = 6000
%       gb_slow_demo (10000)
%
% The GraphBLAS operations used in gbdemo are perhaps 3x to 50x faster than the
% corresponding MATLAB operations, depending on how many cores your computer
% has.  Here's an example where GraphBLAS is asymptotically far faster than
% MATLAB R2019a: a simple assignment C(I,J) = A for a large matrix C.
%
% The matrix C is constructed via C = kron (B,B) where nnz (B) is roughly the
% bnz provided on input (with a default of bnz = 6000), so that C will have
% about bnz^2 entries, or 36 million.
%
% Please be patient ... when the problem becomes large, MATLAB will take a very
% long time.  If you have enough memory, and want to see higher speedups in
% GraphBLAS, increase bnz.  With the default bnz = 6000, this test takes about
% 4GB of RAM.
%
% See also gb.assign, subsasgn.

nthreads = gb.threads ;
help gb_slow_demo
fprintf ('\n# of threads used in GraphBLAS: %d\n\n', nthreads) ;

if (nargin < 1)
    bnz = 6000 ;
end

for n = [1000:1000:6000]

    rng ('default') ;

    tic
    B = sprandn (n, n, bnz / n^2) ;
    C = kron (B, B) ;
    cn = size (C,1) ;
    k = 5000 ;
    anz = 50000 ;
    I = randperm (cn, k) ;
    J = randperm (cn, k) ;
    A = sprandn (k, k, anz / k^2) ;
    G = gb (C) ;
    t_setup = toc ;

    fprintf ('\nC(I,J)=A where C is %g million -by- %g million with %g million entries\n', ...
        cn /1e6, cn /1e6, nnz (C) / 1e6) ;
    fprintf ('    A is %d-by-%d with %d entries\n', k, k, nnz (A)) ;
    fprintf ('    setup time:     %g sec\n', t_setup) ;

    tic
    G (I,J) = A ;
    gb_time = toc ;

    fprintf ('    GraphBLAS time: %g sec\n', gb_time) ;
    fprintf ('    Starting MATLAB ... \n') ;

    tic
    C (I,J) = A ;
    matlab_time = toc ;

    fprintf ('    MATLAB time:    %g sec\n', matlab_time) ;
    fprintf ('    Speedup of GraphBLAS over MATLAB: %g\n', ...
        matlab_time / gb_time) ;

    % check the result
    tic
    assert (isequal (C, sparse (G))) ;
    t_check = toc ;
    fprintf ('    check time:     %g sec\n', t_check) ;

end

