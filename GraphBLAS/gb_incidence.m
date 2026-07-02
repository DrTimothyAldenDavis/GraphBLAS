function C = gb_incidence (ghb, A_arg, varargin)
%GB_INCIDENCE implements GrB.incidence and GhB.incidence.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n] = gbmex_size (A_arg) ;
if (m ~= n)
    error ('GrB:error', 'A must be square') ;
end

% get the string options
kind = 'directed' ;
type = 'double' ;
for k = 1:nargin-2
    arg = lower (varargin {k}) ;
    switch arg
        case { 'directed', 'undirected', 'symmetric', 'unsymmetric', ...
            'lower', 'upper' }
            kind = arg ;
        case { 'double', 'single', 'int8', 'int16', 'int32', 'int64' }
            type = arg ;
        case { 'uint8', 'uint16', 'uint32', 'uint64', 'logical' }
            error ('GrB:error', 'type must be signed') ;
        otherwise
            error ('GrB:error', 'unknown option') ;
    end
end

switch (kind)

    case { 'directed', 'unsymmetric' }

        % create the incidence matrix of a directed graph, using all of A;
        % except that diagonal entries are ignored.
        A = gzb_select (ghb, 'offdiag', A_arg, 0) ;

    case { 'upper' }

        % create the incidence matrix of an undirected graph, using only
        % entries in the strictly upper triangular part of A.
        A = gzb_select (ghb, 'triu', A_arg, 1) ;

    otherwise   % 'undirected', 'symmetric', or 'lower'

        % create the incidence matrix of an undirected graph, using only
        % entries in the strictly lower triangular part of A.
        A = gzb_select (ghb, 'tril', A_arg, -1) ;

end

% build the incidence matrix
desc.base = 'zero-based' ;
gbmex_wait (A) ;
[I, J] = gbmex_extracttuples (ghb, A, desc) ;
e = length (I) ;
I = [I ; J] ;
if (e > intmax ('uint32'))
    % this case cannot be tested by gbtest; it requires a huge test problem:
    J = (uint64 (0) : uint64 (e-1))' ;
else
    J = (uint32 (0) : uint32 (e-1))' ;
end
J = [J ; J] ;
X = ones (e, 1, type) ;
X = [-X ; X] ;
C = gzb_build (ghb, I, J, X, n, e, desc) ;

