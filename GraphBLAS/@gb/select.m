function Cout = select (varargin)
%GB.SELECT: select entries from a GraphBLAS sparse matrix.
%
% Usage:
%
%   Cout = gb.select (selectop, A)
%   Cout = gb.select (selectop, A, thunk)
%   Cout = gb.select (selectop, A, thunk, desc)
%
%   Cout = gb.select (Cin, accum, selectop, A)
%   Cout = gb.select (Cin, accum, selectop, A, thunk)
%   Cout = gb.select (Cin, accum, selectop, A, thunk, desc)
%
%   Cout = gb.select (Cin, M, selectop, A)
%   Cout = gb.select (Cin, M, selectop, A, thunk)
%   Cout = gb.select (Cin, M, selectop, A, thunk, desc)
%
%   Cout = gb.select (Cin, M, accum, selectop, A)
%   Cout = gb.select (Cin, M, accum, selectop, A, thunk)
%   Cout = gb.select (Cin, M, accum, selectop, A, thunk, desc)
%
% gb.select selects a subset of entries from the matrix A, based on their
% value or position.  For example, L = gb.select ('tril', A) returns the
% lower triangular part of the GraphBLAS or MATLAB matrix A, just like L
% = tril (A) for a MATLAB matrix A.  The select operators can also depend
% on the values of the entries.  The thunk parameter is an optional input
% scalar, used in many of the select operators.  For example, L =
% gb.select ('tril', A, -1) is the same as L = tril (A, -1), which
% returns the strictly lower triangular part of A.
%
% The selectop is a string defining the operator:
%
%   operator    MATLAB equivalent                   alternative strings
%   --------    -----------------                   -------------------
%   'tril'      C = tril (A,thunk)                  none
%   'triu'      C = triu (A,thunk)                  none
%   'diag'      C = diag (A,thunk), see note below  none
%   'offdiag'   C = entries not in diag(A,k)        none
%   'nonzero'   C = A (A ~= 0)                      '~=0'
%   'eqzero'    C = A (A == 0)                      '==0'
%   'gtzero'    C = A (A >  0)                      '>0'
%   'gezero'    C = A (A >= 0)                      '>=0'
%   'ltzero'    C = A (A <  0)                      '<0'
%   'lezero'    C = A (A <= 0)                      '<=0'
%   'nethunk'   C = A (A ~= thunk)                  '~=thunk'
%   'eqthunk'   C = A (A == thunk)                  '==thunk'
%   'gtthunk'   C = A (A >  thunk)                  '>thunk'
%   'gethunk'   C = A (A >= thunk)                  '>=thunk'
%   'ltthunk'   C = A (A <  thunk)                  '<thunk'
%   'lethunk'   C = A (A <= thunk)                  '<=thunk'
%
% Note that C = gb.select ('diag',A) does not returns a vector, but a
% diagonal matrix.
%
% Many of the operations have equivalent synonyms, as listed above.
%
% Cin is an optional input matrix.  If Cin is not present or is an empty
% matrix (Cin = [ ]) then it is implicitly a matrix with no entries, of
% the right size (which depends on A, and the descriptor).  Its type is
% the output type of the accum operator, if it is present; otherwise, its
% type is the type of the matrix A.
%
% M is the optional mask matrix.  If not present, or if empty, then no
% mask is used.  If present, M must have the same size as C.
%
% If accum is not present, then the operation becomes C<...> =
% select(...).  Otherwise, accum (C, select(...)) is computed.  The accum
% operator acts like a sparse matrix addition (see gb.eadd).
%
% The selectop is a required string defining the select operator to use.
% All operators operate on all types (the select operators do not do any
% typecasting of its inputs).
%
% A is the input matrix.  It is transposed on input if desc.in0 =
% 'transpose'.
%
% The descriptor desc is optional.  If not present, all default settings
% are used.  Fields not present are treated as their default values.  See
% 'help gb.descriptorinfo' for more details.
%
% All input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  Cout is returned as a GraphBLAS matrix, by default;
% see 'help gb/descriptorinfo' for more options.
%
% See also tril, triu, diag.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[args, is_gb] = gb_get_args (varargin {:}) ;
if (is_gb)
    Cout = gb (gbselect (args {:})) ;
else
    Cout = gbselect (args {:}) ;
end

