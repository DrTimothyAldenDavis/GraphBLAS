function [i j] = convert_index_1d_to_2d (k, m) ;
% convert_index_1d_to_2d: convert 1D indices to 2D
% the indices must be zero-based

i = rem (k, m) ;
j = (k - i) / m ;


