function result = gb_printf_helper (printf_function, varargin)
%GB_PRINTF_HELPER wrapper for fprintf and sprintf

% convert all GraphBLAS matrices to full MATLAB matrices
for k = 2:nargin-1
    arg = varargin {k} ;
    if (isa (arg, 'gb'))
        varargin {k} = full (cast (full (arg), gb.type (arg))) ;
    end
end

% call the built-in fprintf or sprintf
result = builtin (printf_function, varargin {:}) ;

