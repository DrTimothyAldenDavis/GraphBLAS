function C = ceil (G)
%CEIL round entries of a matrix to nearest integers towards infinity.
%
% See also GrB/floor, GrB/round, GrB/fix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_isfloat (gbtype (G)) && gbnvals (G) > 0)
    C = GrB (gbapply ('ceil', G)) ;
else
    C = G ;
end

