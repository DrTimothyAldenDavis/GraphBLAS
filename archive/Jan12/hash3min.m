function Cout = hash3min (varargin)

% TODO delete this when done

[args, is_gb] = gb_get_args (varargin {:}) ;
if (is_gb)
    Cout = GrB (gbhash3min (args {:})) ;
else
    Cout = gbhash3min (args {:}) ;
end

