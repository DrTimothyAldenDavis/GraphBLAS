function C = sign (G)
%SIGN signum function.
% C = sign (G) is the signum function for each entry of G.  For real
% values, sign(x) is 1 if x > 0, zero if x is zero, and -1 if x < 0.
% For the complex case, sign(x) = x ./ abs (x).
%
% See also GrB/abs.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

type = gbmex_type (G) ;

if (isequal (type, 'logical'))
    C = gzb (ghb, G) ;
elseif (~gb_isfloat (type))
    T = gzb_apply (ghb, 'signum.single', G) ;
    C = gzb (ghb, T, type) ;
else
    C = gzb_apply (ghb, 'signum', G) ;
end

