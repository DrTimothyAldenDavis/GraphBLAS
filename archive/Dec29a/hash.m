function Cout = hash (varargin)

[args, is_gb] = gb_get_args (varargin {:}) ;
if (is_gb)
    Cout = GrB (gbhash (args {:})) ;
else
    Cout = gbhash (args {:}) ;
end

