function G = loadobj (S)
%LOADOBJ loads a @GrB matrix from a file.
% MATLAB first reads in the struct S that saveobj created, and then passes it
% to this method.  Octave does not use this method since it cannot save/load
% objects to/from a file.
%
% See also GrB/saveobj, GrB.load.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB
if (isobject (S))
    % S is a @GrB matrix from GraphBLAS 10.3.1 or earlier, which
    % did not have saveobj and loadobj methods.
    G = gzb_loadhistorical (ghb, S.opaque) ;
else
    % S is a struct created by saveobj with a single
    % S.blob field containing the serialized matrix.
    G = gzb_deserialize (ghb, S.blob) ;
end

