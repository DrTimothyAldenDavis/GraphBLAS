function x = gb_get_scalar (A)
%GB_GET_SCALAR get the first scalar from a matrix

[~, ~, x] = gb.extracttuples (A) ;
if (length (x) == 0)
    x = 0 ;
else
    x = x (1) ;
end

