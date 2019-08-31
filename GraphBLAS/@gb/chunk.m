function c = chunk (varargin)
%GB.CHUNK get/set the chunk size to use in GraphBLAS.
%
% Usage:
%   c = gb.chunk ;      % get the current chunk c
%   gb.chunk (c) ;      % set the chunk c
%
% gb.chunk gets and/or sets the chunk size to use in GraphBLAS, which
% controls how many threads GraphBLAS uses for small problems.  The
% default is 4096.  If w is a measure of the work required (w = nvals(A)
% + nvals(B) for C=A+B, for example), then the number of threads
% GraphBLAS uses is min (max (1, floor (w/c)), gb.nthreads).
%
% Changing the chunk via gb.chunk(c) causes all subsequent GraphBLAS
% operations to use that chunk size c.  The setting persists for the
% current MATLAB session, or until 'clear all' or gb.clear is used, at
% which point the setting reverts to the default.
%
% See also gb.threads.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

c = gbchunk (varargin {:}) ;

