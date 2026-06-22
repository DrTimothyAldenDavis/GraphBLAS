function wait (A)
%WAIT finish work on a @GrB matrix
%
% Example:
%
%   GrB.wait (A)
%
% See also GrB.clear, GrB.finalize.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

gbmex_wait (A) ;

