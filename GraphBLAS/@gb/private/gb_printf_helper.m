function result = gb_printf_helper (printf_function, varargin)
%GB_PRINTF_HELPER wrapper for fprintf and sprintf

% convert all GraphBLAS matrices to MATLAB matrices
for k = 2:nargin-1
    arg = varargin {k} ;
    if (isa (arg, 'gb'))
        type = gb.type (arg) ;
        if (gb.isfull (arg))
            % if the GraphBLAS matrix is full, it can be safely cast to a
            % MATLAB full matrix, of any type supported by GraphBLAS.
            arg = full (cast (arg, type)) ;
        elseif (isequal (gb.type (arg), 'logical'))
            % GraphBLAS logical matrices are converted to MATLAB sparse logical
            arg = logical (arg) ;
        else
            % all others are converted to MATLAB double, to keep them sparse
            varargin {k} = double (arg) ;
        end
        varargin {k} = arg ;
    end
end

% call the built-in fprintf or sprintf
result = builtin (printf_function, varargin {:}) ;

