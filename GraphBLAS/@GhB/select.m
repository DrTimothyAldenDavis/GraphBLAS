function C = select (arg1, arg2, arg3, arg4, arg5, arg6, arg7)
%GHB.SELECT select entries from a GraphBLAS sparse matrix.
%
% syntax for a new matrix C (for ops with no b):    computation:
% C = GhB.select (op, A, desc)                      % C = op(A)
% C = GhB.select (Cin, accum, op, A, desc)          % C = Cin ; C += op(A)
% C = GhB.select (Cin, M, op, A, desc)              % C = Cin ; C<M> = op(A)
% C = GhB.select (Cin, M, accum, op, A, desc)       % C = Cin ; C<M> += op(A)
%
% syntax for a new matrix C (for ops with b):
% C = GhB.select (op, A, b, desc)                   % C = op(A,b)
% C = GhB.select (Cin, accum, op, A, b, desc)       % C = Cin ; C += op(A,b)
% C = GhB.select (Cin, M, op, A, b, desc)           % C = Cin ; C<M> = op(A,b)
% C = GhB.select (Cin, M, accum, op, A, b, desc)    % C = Cin ; C<M> += op(A,b)
%
% in-place syntax (for ops with no b):
% GhB.select (C, op, A, desc)                       % C = op(A)
% GhB.select (C, accum, op, A, desc)                % C += op(A)
% GhB.select (C, M, op, A, desc)                    % C<M> = op(A)
% GhB.select (C, M, accum, op, A, desc)             % C<M> += op(A)
%
% in-place syntax (for ops with b):
% GhB.select (C, op, A, b, desc)                    % C = op(A,b)
% GhB.select (C, accum, op, A, b, desc)             % C += op(A,b)
% GhB.select (C, M, op, A, b, desc)                 % C<M> = op(A,b)
% GhB.select (C, M, accum, op, A, b, desc)          % C<M> += op(A,b)
%
% GhB.select selects a subset of entries from the matrix A, based on their
% value or position (shown as op(A) or op(A,b) above).  For example, L =
% GhB.select ('tril', A, 0) returns the lower triangular part of the GraphBLAS
% or built-in matrix A, just like L = tril (A) for a built-in matrix A.  The
% select operators can also depend on the values of the entries.  The b
% parameter is an input scalar, used in many of the select operators.  For
% example, L = GhB.select ('tril', A, -1) is the same as L = tril (A, -1),
% which returns the strictly lower triangular part of A.  The b scalar is
% required for 'tril', 'triu', 'diag', 'offdiag' and the 2-input operators.  It
% must not appear when using the '*0' operators.
%
% The selectop is a string defining the operator:
%
%   operator        built-in equivalent         equivalent strings
%   --------        -----------------           ------------------
%   'tril'          C = tril (A,b)
%   'triu'          C = triu (A,b)
%   'diag'          C = diag (A,b), see note
%   'offdiag'       C = entries not in diag(A,b)
%   'nonzero'       C = A (A ~= 0)              '~=0'
%   'zero'          C = A (A == 0)              '==0'
%   'positive'      C = A (A >  0)              '>0'
%   'nonnegative'   C = A (A >= 0)              '>=0'
%   'negative'      C = A (A <  0)              '<0'
%   'nonpositive'   C = A (A <= 0)              '<=0'
%   '~='            C = A (A ~= b)
%   '=='            C = A (A == b)
%   '>'             C = A (A >  b)
%   '>='            C = A (A >= b)
%   '<'             C = A (A <  b)
%   '<='            C = A (A <= b)
%
% Many of the operations have equivalent synonyms, as listed above.
% Note that C = GhB.select ('diag',A,b) does not return a vector,
% but a diagonal matrix, instead.
%
% accum: a binary operator to accumulate the results; in the computations
% listed above it is shown as "+=" but any binary operator may be used.
% For the in-place syntax, the @GhB matrix C is modified in-place.
%
% Cin, the mask matrix M, the accum operator, and desc are optional.  If either
% accum or M is present, then C or Cin is a required input.  If desc.in0 is
% 'transpose' then A is transposed before applying the operator.
%
% The selectop is a required string defining the select operator to use.
% All operators operate on all types (the select operators do not do any
% typecasting of its inputs).
%
% See also GhB/tril, GhB/triu, GhB/diag, GrB.selectopinfo, GrB.binopinfo.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

if (nargout == 0)
    switch (nargin)
        case 3
            gbmex_select (ghb, arg1, arg2, arg3) ;
        case 4
            gbmex_select (ghb, arg1, arg2, arg3, arg4) ;
        case 5
            gbmex_select (ghb, arg1, arg2, arg3, arg4, arg5) ;
        case 6
            gbmex_select (ghb, arg1, arg2, arg3, arg4, arg5, arg6) ;
        case 7
            gbmex_select (ghb, arg1, arg2, arg3, arg4, arg5, arg6, arg7) ;
        otherwise
            error ('GrB:error', ...
                'usage: GhB.select (C, M, accum, selectop, A, b, desc)') ;
    end
else
    switch (nargin)
        case 2
            [C_opaque, kind] = gbmex_select (ghb, arg1, arg2) ;
        case 3
            [C_opaque, kind] = gbmex_select (ghb, arg1, arg2, arg3) ;
        case 4
            [C_opaque, kind] = gbmex_select (ghb, arg1, arg2, arg3, arg4) ;
        case 5
            [C_opaque, kind] = gbmex_select (ghb, arg1, arg2, arg3, arg4, ...
                arg5) ;
        case 6
            [C_opaque, kind] = gbmex_select (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6) ;
        case 7
            [C_opaque, kind] = gbmex_select (ghb, arg1, arg2, arg3, arg4, ...
                arg5, arg6, arg7) ;
        otherwise
            error ('GrB:error', ...
                'usage: C = GhB.select (Cin, M, accum, selectop, A, b, desc)') ;
    end
    C = gb_mexfunction_result (ghb, C_opaque, kind) ;
end

