function S = struct (G)
%GRB.STRUCT return the opaque (private) contents of a @GrB object.
% This method is meant for testing and development only.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

S = G.opaque ;
