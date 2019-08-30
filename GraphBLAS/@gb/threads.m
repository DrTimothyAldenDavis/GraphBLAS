function nthreads = threads (varargin)
%GB.THREADS get/set the max number of threads to use in GraphBLAS
%
% Usage:
%   nthreads = gb.threads ;      % get the current maximum # of threads
%   gb.threads (nthreads) ;      % set the maximum # of threads
%
% gb.threads gets and/or sets the maximum number of threads to use in
% GraphBLAS.  By default, if GraphBLAS has been compiled with OpenMP, it
% uses the number of threads returned by omp_get_max_threads.  Otherwise,
% it can only use a single thread.
%
% Changing the number of threads with gb.threads(nthreads) causes all
% subsequent GraphBLAS operations to use at most the given number of
% threads.  GraphBLAS may use fewer threads, if the problem is small (see
% gb.chunk).  The setting is kept for the remainder of the current MATLAB
% session, or until 'clear all' or gb.clear is used, at which point the
% setting reverts to the default number of threads.
%
% MATLAB can detect the number of physical and logical cores via an
% undocumented builtin function: ncores = feature('numcores'), or via
% maxNumCompThreads.
%
% Example:
%
%   feature ('numcores') ;          % print info about cores
%   ncores = feature ('numcores') ; % get # of logical cores MATLAB uses
%   ncores = maxNumCompThreads ;    % same as feature ('numcores')
%   gb.threads (2*ncores) ;         % GraphBLAS will use <= 2*ncores threads
%
% See also feature, maxNumCompThreads, gb.chunk.

nthreads = gbthreads (varargin {:}) ;

