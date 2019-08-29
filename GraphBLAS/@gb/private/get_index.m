function [I, whole] = get_index (I_input)
% get_index: helper function for subsref and subsasgn

whole = isequal (I_input, {':'}) ;
if (whole)
    % C (:)
    I = { } ;
elseif (iscell (I_input {1}))
    % C ({ }), C ({ list }), C ({start,fini}), or C ({start,inc,fini}).
    I = I_input {1} ;
else
    % C (I) for an explicit list I, or MATLAB colon notation
    I = I_input ;
end

