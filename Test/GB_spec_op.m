function C = GB_spec_op (op, A, B)
%GB_SPEC_OP apply a unary or binary operator
%
% Apply a binary operator z = f (x,y) element-wise to x and y, or a unary
% operator z = f(x) just x.  The operator op is any built-in GraphBLAS
% operator.
%
% op or op.opname is a string with just the operator name.  Valid names of
% binary operators are 'first', 'second', 'min', 'max', 'plus', 'minus',
% 'rminus', 'times', 'div', 'rdiv', 'eq', 'ne', 'gt', 'lt', 'ge', 'le', 'or',
% 'and', 'xor'.  'iseq', 'isne', 'isgt', 'islt', 'isge', 'le', 'pair', 'any',
% 'pow', ('bitget' or 'bget'), ('bitset' or 'bset'), ('bitclr' or 'bclr'),
% ('bitand' or 'band'), ('bitor' or 'bor'), ('bitxor' or 'bxor'), ('bitshift'
% or 'bshift'), ('bitnot' or 'bitcmp'), 'atan2', 'hypot', ('ldexp' or 'pow2'),
% ('complex', 'cmplx').  

% Unary operators are 'one', 'identity', 'ainv', 'abs', 'minv', 'not', 'bnot',
% 'sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh',
% 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'ceil', 'floor', 'round', ('trunc'
% or 'fix'), ('exp2' or 'fix'), 'expm1', 'log10', 'log2', ('lgamma' or
% 'gammaln'), ('tgamma' or 'gamma'), 'erf', 'erfc', 'frexpx', 'frexpe', 'conj',
% ('creal' or 'real'), ('cimag' or 'imag'), ('carg' or 'angle'), 'isinf',
% 'isnan', 'isfinite'.
%
% op.optype: 'logical', 'int8', 'uint8', 'int16', 'uint16', 'int32',
%   'uint32', 'int64', 'uint64', 'single', 'double', 'single complex'
%   or 'double complex'.
%
% The class of z is the same as the class of the output of the operator,
% which is op.optype except for: (1) 'eq', 'ne', 'gt', 'lt', 'ge', 'le',
% in which case z is logical, (2) 'complex', where x and y are real and
% z is complex, (3) ...
%
% Intrinsic MATLAB operators are used as much as possible, so as to test
% GraphBLAS operators.  Some must be done in GraphBLAS because the
% divide-by-zero and overflow rules for integers differs between MATLAB and C.
% Also, typecasting in MATLAB and GraphBLAS differs with underflow and overflow
% conditions.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

% get the operator name and class
[opname optype] = GB_spec_operator (op, GB_spec_type (A)) ;

% cast the inputs A and B to the inputs of the operator
if (~isequal (GB_spec_type (A), optype))
    x = GB_mex_cast (A, optype) ;
else
    x = A ;
end

% use GB_mex_op for integer and logical plus, minus, times, and div
use_matlab = (isa (x, 'float') && ...
    (contains (optype, 'single') || contains (optype, 'double'))) ;

if (nargin > 2)
    if (~isequal (GB_spec_type (B), optype))
        y = GB_mex_cast (B, optype) ;
    else
        y = B ;
    end
    use_matlab = use_matlab && isa (y, 'float') ;
end

switch opname

    % binary operators, result is optype
    case 'first'
        z = x ;
    case 'second'
        z = y ;
    case 'any'
        z = y ;
    case 'pair'
        z = GB_spec_ones (size (x), optype) ;
    case 'min'
        % min(x,y) in SuiteSparse:GraphBLAS is min(x,y,'omitnan') in MATLAB.
        % see discussion in SuiteSparse/GraphBLAS/Source/GB.h
        % z = min (x,y,'omitnan') ;
        z = GB_mex_op (op, x, y) ;
    case 'max'
        % z = max (x,y,'omitnan') ;
        z = GB_mex_op (op, x, y) ;
    case 'plus'
        if (use_matlab)
            z = x + y ;
        else
            z = GB_mex_op (op, x, y) ;
        end
    case 'minus'
        if (use_matlab)
            z = x - y ;
        else
            z = GB_mex_op (op, x, y) ;
        end
    case 'rminus'
        if (use_matlab)
            z = y - x ;
        else
            z = GB_mex_op (op, x, y) ;
        end
    case 'times'
        if (use_matlab)
            z = x .* y ;
        else
            z = GB_mex_op (op, x, y) ;
        end
    case 'div'
        if (use_matlab)
            z = x ./ y ;
        else
            z = GB_mex_op (op, x, y) ;
        end
    case 'rdiv'
        if (use_matlab)
            z = y ./ x ;
        else
            z = GB_mex_op (op, x, y) ;
        end
    case 'pow'
        if (use_matlab)
            z = y .^ x ;
        else
            z = GB_mex_op (op, x, y) ;
        end

    % 6 binary comparison operators (result is same as optype)
    case 'iseq'
        z = cast (x == y, optype) ;
    case 'isne'
        z = cast (x ~= y, optype) ;
    case 'isgt'
        z = cast (x >  y, optype) ;
    case 'islt'
        z = cast (x <  y, optype) ;
    case 'isge'
        z = cast (x >= y, optype) ;
    case 'isle'
        z = cast (x <= y, optype) ;

    % 6 binary comparison operators (result is boolean)
    case 'eq'
        z = (x == y) ;
    case 'ne'
        z = (x ~= y) ;
    case 'gt'
        z = (x >  y) ;
    case 'lt'
        z = (x <  y) ;
    case 'ge'
        z = (x >= y) ;
    case 'le'
        z = (x <= y) ;

    % 3 binary logical operators (result is optype)
    case 'or'
        z = cast ((x ~= 0) | (y ~= 0), optype) ;
    case 'and'
        z = cast ((x ~= 0) & (y ~= 0), optype) ;
    case 'xor'
        z = cast ((x ~= 0) ~= (y ~= 0), optype) ;

    % bitwise operators
    case { 'bitget', 'bget' }
        z = bitget (x, y, optype) ;
    case { 'bitset', 'bset' }
        z = bitset (x, y, optype) ;
    case { 'bitclr', 'bclr' }
        z = bitset (x, y, 0, optype) ;
    case { 'bitand', 'band' }
        z = bitand (x, y, optype) ;
    case { 'bitor', 'bor' }
        z = bitor (x, y, optype) ;
    case { 'bitxor', 'bxor' }
        z = bitxor (x, y, optype) ;
    case { 'bitshift', 'bshift' }
        z = bitshift (x, y, optype) ;
    case { 'bitnot', 'bitcmp' }
        z = bitcmp (x, optype) ;

    case 'atan2'
        z = atan2 (x,y) ;

    case 'hypot'
        z = hypot (x,y) ;

    case { 'ldexp', 'pow2' }
        z = pow2 (x,y) ;

        % z = fmod (x,y)
        % z = remainder (x,y)
        % z = copysign (x,y)

    case { 'complex', 'cmplx' }
        z = complex (x,y) ;

    % unary operators (result is optype)
    case 'one'
        z = cast (1, optype) ;
    case 'identity'
        z = x ;
    case 'ainv'
        if (use_matlab)
            z = -x ;
        else
            z = GB_mex_op (op, x) ;
        end
    case 'abs'
        if (use_matlab)
            z = abs (x) ;
        else
            z = GB_mex_op (op, x) ;
        end
    case 'minv'
        if (use_matlab)
            z = 1 ./ x ;
        else
            z = GB_mex_op (op, x) ;
        end
    case 'not'
        z = cast (~(x ~= 0), optype) ;

    case 'bnot'
        z = bitcmp (x) ;

    case 'sqrt'
        z = sqrt (x) ;

    case 'log'
        z = log (x) ;

    case 'exp'
        z = exp (x) ;

    case 'sin'
        z = sin (x) ;

    case 'cos'
        z = cos (x) ;

    case 'tan'
        z = tan (x) ;

    case 'asin'
        z = asin (x) ;

    case 'acos'
        z = acos (x) ;

    case 'atan'
        z = atan (x) ;

    case 'sinh'
        z = sinh (x) ;

    case 'cosh'
        z = cosh (x) ;

    case 'tanh'
        z = tanh (x) ;

    case 'asinh'
        z = asinh (x) ;

    case 'acosh'
        z = acosh (x) ;

    case 'atanh'
        z = atanh (x) ;

    case 'ceil'
        z = ceil (x) ;

    case 'floor'
        z = floor (x) ;

    case 'round'
        z = round (x) ;

    case { 'trunc', 'fix' }
        z = fix (x) ;

    case { 'exp2', 'fix' }
        z = 2.^x ;

    case 'expm1'
        z = expm1 (x) ;

    case 'log10'
        z = log10 (x) ;

    case 'log2'
        z = log2 (x) ;

    case { 'lgamma', 'gammaln' }
        z = gammaln (x) ;

    case { 'tgamma', 'gamma' }
        z = gamma (x) ;

    case 'erf'
        z = erf (x) ;

    case 'erfc'
        z = erfc (x) ;

    case 'frexpx'
        [z,~] = log2 (x) ;

    case 'frexpe'
        [~,z] = log2 (x) ;

    case 'conj'
        z = conj (x) ;

    case { 'creal', 'real' }
        z = real (x) ;

    case { 'cimag', 'imag' }
        z = imag (x) ;

    case { 'carg', 'angle' }
        z = angle (x) ;

    case 'isinf'
        z = isinf (x) ;

    case 'isnan'
        z = isnan (x) ;

    case 'isfinite'
        z = isfinite (x) ;

    otherwise
        opname
        error ('unknown op') ;
end

C = z ;


