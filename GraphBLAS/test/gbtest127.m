function gbtest127 (ghb)
%GBTEST127 test [GrB,GhB].semirings

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 0)
    ghb = 0 ;
end
gtb_name = gtb_prep (ghb) ;

list = gtb_semirings (ghb)
gtb_semirings (ghb) ;
if (ghb)
    help GhB.semirings ;
else
    help GhB.semirings ;
end

fprintf ('\ngbtest127 (%d): all tests passed\n', ghb) ;

