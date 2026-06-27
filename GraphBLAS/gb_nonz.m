function result = gb_nonz (ghb, A, varargin)
%GB_NONZ implements GrB.nonz and GrB.nonz.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

builtin_sparse = builtin ('issparse', A) ;

% get the identity value
id = 0 ;
nargs = nargin ;
if (nargin > 2)
    lastarg = varargin {nargs-2} ;
    if (~ischar (lastarg))
        % the last argument is id, if it is not a string
        id = gb_get_scalar (ghb, lastarg) ;
        nargs = nargs - 1 ;
    end
end

if (id ~= 0)
    % id is nonzero, so prune A first (for any matrix A)
    T = gzb_select (ghb, A, '~=', id) ;
    result = gb_entries (ghb, T, varargin {1:nargs-2}) ;
elseif (~builtin_sparse)
    % id is zero, so prune A only if it is a GraphBLAS matrix,
    % or a built-in full matrix.  A built-in sparse matrix can remain
    % unchanged.
    T = gzb_select (ghb, A, 'nonzero') ;
    result = gb_entries (ghb, T, varargin {1:nargs-2}) ;
else
    % get the count/list of the entries of A
    result = gb_entries (ghb, A, varargin {1:nargs-2}) ;
end


