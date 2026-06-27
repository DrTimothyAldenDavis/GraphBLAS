function C = speye (varargin)
%GRB.SPEYE sparse identity matrix.
% C = GrB.speye (...) is identical to GrB.eye; see 'help GrB.eye' for
% details.
%
% See also GrB.eye.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

C = gb_speye (ghb, 'speye', varargin {:}) ;

