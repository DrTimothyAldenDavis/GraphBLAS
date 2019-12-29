function Cout = hash3 (varargin)

[args, is_gb] = gb_get_args (varargin {:}) ;
if (is_gb)
    Cout = GrB (gbhash3 (args {:})) ;
else
    Cout = gbhash3 (args {:}) ;
end

