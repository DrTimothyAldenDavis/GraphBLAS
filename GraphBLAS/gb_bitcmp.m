function C = gb_bitcmp (ghb, A, assumedtype)
%GB_BITCMP implements GrB/bitcmp and GhB/bitcmp.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 3)
    assumedtype = 'uint64' ;
end

atype = gbmex_type (A) ;

if (gb_contains (atype, 'complex'))
    error ('GrB:error', 'inputs must be real') ;
end

if (isequal (atype, 'logical'))
    error ('GrB:error', 'inputs must not be logical') ;
end

if (~gb_contains (assumedtype, 'int'))
    error ('GrB:error', 'assumedtype must be an integer type') ;
end

% C will have the same type as A on input
ctype = atype ;

if (isequal (atype, 'double') || isequal (atype, 'single'))
    % cast A to the assumedtype
    C = gzb_full (ghb, gzb (ghb, A, assumedtype)) ;
else
    C = gzb_full (ghb, A) ;
end

C = gzb_apply (ghb, 'bitcmp', C) ;

if (~isequal (gbmex_type (C), ctype))
    C = gzb (ghb, C, ctype) ;
end


