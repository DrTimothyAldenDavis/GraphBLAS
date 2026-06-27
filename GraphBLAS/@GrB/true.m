function C = true (varargin)
%GRB.TRUE a logical matrix with all true values.
%
%   C = GrB.true (n) ;      n-by-n GrB logical matrix of all true entries.
%   C = GrB.true (m,n) ;    m-by-n GrB logical matrix of all true entries.
%   C = GrB.true ([m,n]) ;  m-by-n GrB logical matrix of all true entries.
%   C = GrB.true (..., type) ;      matrix of all true entries of given type.
%   C = GrB.true (..., 'like', G) ; matrix of all true entries, same type as G.
%
% The memory required to store C is O(1) not O(m*n), so both m and
% n can be as large as 2^60.
%
% See also GrB.zeros, GrB.ones, GrB.false, GrB.eye, GrB.speye.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

C = gb_true (ghb, varargin {:}) ;

