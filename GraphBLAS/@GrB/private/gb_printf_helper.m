function result = gb_printf_helper (printf_function, varargin)
%GB_PRINTF_HELPER wrapper for fprintf and sprintf

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% TODO

% convert all GraphBLAS matrices to full MATLAB matrices
for k = 2:nargin-1
    arg = varargin {k} ;
    if (isobject (arg))
        % TODO FIXME for complex case
        varargin {k} = full (cast (full (arg), GrB.type (arg))) ;
    end
end

% call the built-in fprintf or sprintf
result = builtin (printf_function, varargin {:}) ;

