function C = ones (varargin)
%ONES an all-zero matrix, the same type as G
% C = ones (m, n, 'like', G)
% C = ones ([m n], 'like', G)

% TODO: test

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = gb.subassign (zeros (varargin {:}), 1, { }, { }) ;

