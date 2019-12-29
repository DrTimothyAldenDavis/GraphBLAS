function Cout = hash2 (varargin)

[args, is_gb] = gb_get_args (varargin {:}) ;
if (is_gb)
    Cout = GrB (gbhash2 (args {:})) ;
else
    Cout = gbhash2 (args {:}) ;
end

