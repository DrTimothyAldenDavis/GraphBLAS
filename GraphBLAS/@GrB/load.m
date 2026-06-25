function C = load (filename)
%GRB.LOAD Load a single GraphBLAS matrix from a file.
% C = GrB.load (filename) loads a single @GrB matrix from a file.  If the
% filename is not present, it defaults to 'GrB_Matrix.mat'.
%
% GrB.load can load in *.mat files created by GrB.save from this or earlier
% versions of GraphBLAS.
%
% NOTE: As of GraphBLAS v10.4.0, this method is no longer needed; just use
% the MATLAB/Octave load/save methods instead.
%
% Examples:
%
%   A = GrB.random (4, 4, 0.5)
%   GrB.save (A) ;              % A can be a @GrB or built-in matrix
%   clear all
%   A = GrB.load ('A.mat') ;    % A is now a @GrB matrix
%
%   % saving a matrix expression
%   GrB.save (2*A-1)            % save a matrix computation to GrB_Matrix.mat
%   GrB.load                    % load it back in
%
% See also load, save, GrB.save, GrB.serialize, GrB.deserialize.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (nargin < 1)
    filename = 'GrB_Matrix.mat' ;
end

S = load (filename) ;

if (isfield (S, 'GraphBLAS_struct_from_GrB_save'))
    % S was created by GrB.save from GraphBLAS v10.3.1 or earlier
    C = gzb_loadhistorical (ghb, S.GraphBLAS_struct_from_GrB_save) ;
elseif (isfield (S, 'GrB_Matrix_from_GrB_save'))
    % S was created by GrB.save from GraphBLAS v10.4.0 or later,
    % and it already contains a properly loaded @GrB matrix.
    % FIXME: convert to @GhB or @GrB, depending on ghb.
    C = S.GrB_Matrix_from_GrB_save ;
else
    % S has already been properly loaded by GrB/loadobj
    C = S ;
end

