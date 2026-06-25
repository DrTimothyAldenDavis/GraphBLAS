function C = flip (A, dim)
%FLIP flip the order of elements.
% C = flip (A) flips the order of elements in each column of A.  That is,
% C = A (end:-1:1,:).  C = flip (A, dim) specifies the dimension to flip,
% so that flip (A,1) and flip (A) are the same thing, and flip (A,2) flips
% the columns so that C = A (:,end:-1,1).
%
% To use this function on a built-in matrix, use C = flip (A, GrB (dim)).
%
% See also GrB/transpose.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

[m, n] = gbmex_size (A) ;

if (nargin == 1)
    if (m == 1)
        dim = 2 ;
    else
        dim = 1 ;
    end
else
    dim = gb_get_scalar (dim) ;
end

dim = floor (double (dim)) ;
if (dim <= 0)
    error ('GrB:error', 'dim must be positive') ;
end

if (dim == 1 && m ~= 1)
    % C = A (m:-1:1, :)
    C = gzb_extract (ghb, A, {m,-1,1}, { }) ;
elseif (dim == 2 && n ~= 1)
    % C = A (:, n:-1:1)
    C = gzb_extract (ghb, A, { }, {n,-1,1}) ;
else
    % nothing to do
    % C = A
    C = gzb (ghb, A) ;
end

