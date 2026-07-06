function C = build (I,J,X,varargin)
%GHB.BUILD construct a sparse matrix from a list of entries.
%
% GhB.build is identical to GrB.build, except that it creates a @GhB matrix C,
% whereas GrB.build creates a @GrB matrix C.  See 'help GrB.build' for details.
%
% See also sparse, GhB/sparse, GhB/find, GhB.extracttuples.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_build (1, I, J, X, varargin {:}) ;

