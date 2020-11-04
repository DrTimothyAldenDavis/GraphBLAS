function Mask = GB_spec_getmask (Mask, Mask_struct)
%GB_SPEC_GETMASK return the mask, typecasted to logical

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isstruct (Mask))
    if (Mask_struct && isfield (Mask, 'pattern'))
        Mask = Mask.pattern ;
    else
        Mask = Mask.matrix ;
    end
end

if (Mask_struct)
    if (issparse (Mask))
        Mask = GB_spones_mex (Mask) ;
    else
        Mask = true (size (Mask)) ;
    end
end

Mask = GB_mex_cast (full (Mask), 'logical') ;
