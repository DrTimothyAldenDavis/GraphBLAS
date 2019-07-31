function nthreads = gbthreads (nthreads)
%GBTHREADS get/set the maximum number of threads to use in SuiteSparse:GraphBLAS
%
% Usage:
%   nthreads = gbthreads ;      % get the current maximum # of threads
%   gbthreads (nthreads) ;      % set the maximum # of threads
%
% gbthreads gets and/or sets the maximum number of threads to use in GraphBLAS.
% By default, if GraphBLAS has been compiled with OpenMP, it uses the number of
% threads returned by omp_get_max_threads.  Otherwise, it can only use a
% single thread.
%
% Changing the number of threads with gbthreads(nthreads) causes all subsequent
% GraphBLAS operations to use at most the given number of threads.  GraphBLAS
% may use fewer threads, if the problem is small.  The setting persists for the
% current MATLAB session, or until 'clear all' is used, at which point the
% setting reverts to the default number of threads.
%
% MATLAB can detect the number of physical and logical cores via an
% undocumented builtin function: ncores = feature('numcores') ;
% or via maxNumCompThreads
%
% Example:
%
%   feature ('numcores') ;          % print info about cores
%   ncores = feature ('numcores') ; % get # of logical cores MATLAB uses
%   ncores = maxNumCompThreads ;    % same as feature ('numcores')
%   gbthreads (2*ncores) ;          % GraphBLAS will use <= 2*ncores threads
%
% TODO add chunk
%
% See also feature, maxNumCompThreads.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('gbthreads mexFunction not found; use gbmake to compile GraphBLAS') ;

