function result = gb_printf_helper (printf_function, varargin)
%GB_PRINTF_HELPER wrapper for fprintf and sprintf

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% convert all GraphBLAS matrices to full MATLAB matrices
len = length (varargin) ;
for k = 2:len
    arg = varargin {k} ;
    if (isobject (arg))
        arg = arg.opaque ;
        varargin {k} = gbfull (arg) ;
    end
end

% call the built-in fprintf or sprintf
result = builtin (printf_function, varargin {:}) ;

