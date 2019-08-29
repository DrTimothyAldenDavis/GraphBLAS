function x = get_scalar (A)
% get_scalar: get the first scalar from a matrix

[~, ~, x] = gb.extracttuples (A) ;
if (length (x) == 0)
    x = 0 ;
else
    x = x (1) ;
end

