function Cout = hash4 (varargin)

[args, is_gb] = gb_get_args (varargin {:}) ;
if (is_gb)
    Cout = GrB (gbhash4 (args {:})) ;
else
    Cout = gbhash4 (args {:}) ;
end

