function S = saveobj (G)
%SAVEOBJ prepares a @GrB matrix for MATLAB/Octave to save to a file.
%
% See also GrB/loadobj, GrB.save.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB
S.blob = gbmex_builtin (gzb_serialize (ghb, G)) ;

