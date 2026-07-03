function S = saveobj (G)
%SAVEOBJ prepares a @GhB matrix for MATLAB to save to a file.
% Octave does not use this method since it cannot save objects to a file.
%
% See also GhB/loadobj, GhB.save.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

S.blob = gbmex_builtin (gzb_serialize (ghb, G)) ;

