function Mask = GB_spec_getmask (Mask)
%GB_SPEC_GETMASK return the mask, typecasted to logical

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isstruct (Mask))
    Mask = Mask.matrix ;
end

Mask = GB_mex_cast (full (Mask), 'logical') ;

