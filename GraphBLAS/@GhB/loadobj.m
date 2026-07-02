function G = loadobj (S)
%LOADOBJ loads a @GhB matrix from a file.
% MATLAB/Octave first reads in the struct S that saveobj created, and
% then passes it to this method.
%
% See also GhB/saveobj, GhB.load.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

% S is a struct created by saveobj with a single S.blob field containing the
% serialized matrix.  It cannot be a historical @GhB object, since the @GhB
% object was introduced in GraphBLAS v10.4.0, the same time loadobj and saveobj
% were added.
G = gzb_deserialize (ghb, S.blob) ;

