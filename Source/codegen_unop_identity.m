function codegen_unop_identity
%CODEGEN_UNOP_IDENTITY create identity functions
%
% The 'identity' operator is unique: it is used for typecasting, and all 13*13
% pairs of functions are generated.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

unop = 'identity' ;
fprintf ('\n%-9s', unop) ;

types = { 
'bool',
'int8_t',
'int16_t',
'int32_t',
'int64_t',
'uint8_t',
'uint16_t',
'uint32_t',
'uint64_t',
'float',
'double',
'GxB_FC32_t',
'GxB_FC64_t' } ;

ntypes = length (types) ;

for code1 = 1:ntypes
    ctype = types {code1} ;
    for code2 = 1:ntypes
        atype = types {code2} ;

        % determine the casting function

        if (isequal (atype, ctype))

            % no typecasting
            func = 'xarg' ;

        elseif (isequal (atype, 'GxB_FC32_t'))

            % typecasting from GxB_FC32_t
            if (isequal (ctype, 'bool')) 
                % to bool from GxB_FC32_t 
                func = '(crealf (xarg) != 0) || (cimagf (xarg) != 0)' ;
            elseif (codegen_contains (ctype, 'int'))
                % to integer from GxB_FC32_t 
                func = sprintf ('GB_cast_to_%s ((double) crealf (xarg))', ctype) ;
            elseif (isequal (ctype, 'float') || isequal (ctype, 'double')) 
                % to float or double from GxB_FC32_t 
                func = sprintf ('(%s) crealf (xarg)', ctype) ;
            elseif (isequal (ctype, 'GxB_FC64_t'))
                % to GxB_FC64_t from GxB_FC32_t 
                func = 'GB_CMPLX64 ((double) crealf (xarg), (double) cimagf (xarg))' ;
            end

        elseif (isequal (atype, 'GxB_FC64_t'))

            % typecasting from GxB_FC64_t
            if (isequal (ctype, 'bool')) 
                % to bool from GxB_FC64_t 
                func = '(creal (xarg) != 0) || (cimag (xarg) != 0)' ;
            elseif (codegen_contains (ctype, 'int'))
                % to integer from GxB_FC64_t 
                func = sprintf ('GB_cast_to_%s (creal (xarg))', ctype) ;
            elseif (isequal (ctype, 'float') || isequal (ctype, 'double')) 
                % to float or double from GxB_FC64_t 
                func = sprintf ('(%s) creal (xarg)', ctype) ;
            elseif (isequal (ctype, 'GxB_FC32_t'))
                % to GxB_FC32_t from GxB_FC64_t 
                func = 'GB_CMPLX32 ((float) creal (xarg), (float) cimag (xarg))' ;
            end

        elseif (isequal (atype, 'float') || isequal (atype, 'double'))

            % typecasting from float or double
            if (isequal (ctype, 'bool')) 
                % to bool from float or double 
                func = '(xarg != 0)' ;
            elseif (codegen_contains (ctype, 'int'))
                % to integer from float or double 
                func = sprintf ('GB_cast_to_%s ((double) (xarg))', ctype) ;
            elseif (isequal (ctype, 'GxB_FC32_t'))
                % to GxB_FC32_t from float or double
                func = 'GB_CMPLX32 ((float) (xarg), 0)' ;
            elseif (isequal (ctype, 'GxB_FC64_t'))
                % to GxB_FC64_t from float or double
                func = 'GB_CMPLX64 ((double) (xarg), 0)' ;
            end

        elseif (isequal (ctype, 'GxB_FC32_t'))

            % typecasting to GxB_FC32_t from any real type
            func = 'GB_CMPLX32 ((float) (xarg), 0)' ;

        elseif (isequal (ctype, 'GxB_FC64_t'))

            % typecasting to GxB_FC64_t from any real type
            func = 'GB_CMPLX64 ((double) (xarg), 0)' ;

        else
            % use ANSI typecasting rules
            func = sprintf ('(%s) xarg', ctype) ;
        end

        codegen_unop_method (unop, func, ctype, atype) ;
    end
end

