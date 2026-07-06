function C = gb_full (ghb, A, type, identity)
%GB_FULL implements GrB/full and GhB/full.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (nargin < 3)
    type = gbmex_type (A) ;
    right_type = true ;
else
    right_type = isequal (type, gbmex_type (A)) ;
end

if (gb_isfull (A) && right_type)

    % nothing to do, A is already full and has the right type
    C = gzb (ghb, A) ;

else

    % convert A to a full GraphBLAS matrix
    if (nargin < 4)
        identity = 0 ;
    end
    C = gzb_full (ghb, A, type, identity) ;

end

