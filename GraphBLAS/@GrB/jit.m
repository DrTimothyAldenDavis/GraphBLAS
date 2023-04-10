function [s,path] = jit (s,path)
%GRB.JIT controls the GraphBLAS JIT
%
%   s = GrB.jit ;      % get the current status of the JIT
%   s = GrB.jit (s) ;  % control the JIT and get its status
%
% The GraphBLAS JIT allows GraphBLAS to compile new kernels at run-time
% that are specifically tuned for the particular operators, types, and
% matrix formats.  Without the JIT, only a selected combination of these
% options are computed with high-performance kernels.
%
% GrB.jit controls the GraphBLAS JIT.  Its input/ouput s is a string:
%
%   'off'       do not use the JIT, and free any loaded JIT kernels.
%   'pause'     do not run JIT kernels but keep any already loaded.
%   'run'       run JIT kernels if already loaded; no load/compile.
%   'load'      able to load and run JIT kernels; may not compile.
%   'on'        full JIT: able to compile, load, and run.
%
% A second input/output parameter gives the path to a cache folder where
% GraphBLAS keeps the kernels it compiles for the user.  By default, this
% is ~/.SuiteSparse/GraphBLAS/8.0.0_matlab for Linux and the Mac for
% GraphBLAS v8.0.0, with a new cache path used for each future @GrB
% version.
%
% If GraphBLAS was compiled with its JIT disabled, only the 'off',
% 'pause', and'run' options are avaiable.  These options do not allow for
% any JIT kernels to be loaded and compiled from the cache path.
% Instead, they control just the 'PreJIT' kernels.  Those kernels are JIT
% kernels from a prior session that were then copied into
% GraphBLAS/GraphBLAS/PreJIT, after which the libgraphblas_matlab.so
% library was compiled.  Refer to the GraphBLAS User Guide for details.
%
% Example:
%
%   [s,path] = GrB.jit
%   [s,path] = GrB.jit ('on', '/home/me/myothercache')
%
% See also GrB.threads, GrB.clear.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    [s, path] = gbjit ;
elseif (nargin == 1)
    [s, path] = gbjit (s) ;
elseif (nargin == 2)
    [s, path] = gbjit (s, path) ;
end

