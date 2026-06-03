function result = gb_printf_helper (printf_function, varargin)
%GB_PRINTF_HELPER wrapper for fprintf and sprintf

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% convert all GraphBLAS matrices to full built-in matrices
len = length (varargin) ;
args = cell (1, len) ;
for k = 1:len
    arg = varargin {k} ;
    if (isobject (arg))
        desc.kind = 'full' ;
        args {k} = gb2builtin (GrB (gbfull (arg, gbtype (arg), 0, desc))) ;
    else
        args {k} = arg ;
    end
end

% call the built-in fprintf or sprintf
result = builtin (printf_function, args {:}) ;

