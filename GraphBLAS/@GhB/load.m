function C = load (filename)
%GHB.LOAD Load a single GraphBLAS matrix from a file.
% C = GrB.load (filename) loads a single @GrB or @GhB matrix from a file.
% If the filename is not present, it defaults to 'GrB_Matrix.mat'.
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

ghb = 1 ;     % 0 for GrB, 1 for GhB

if (nargin < 1)
    filename = 'GrB_Matrix.mat' ;
end

C = gb_load (ghb, filename) ;

