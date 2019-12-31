function codegen_axb_template (multop, bmult, imult, fmult, dmult)
%CODEGEN_AXB_TEMPLATE create a function for a semiring with a TxT->T multiplier

fprintf ('\n%-7s', multop) ;

if (nargin < 4)
    fmult = [ ] ;
end

if (nargin < 5)
    dmult = [ ] ;
end

if (isempty (fmult))
    fmult = imult ;
end

if (isempty (dmult))
    dmult = fmult ;
end

% min monoid: all are terminal
% TODO: find atomic alternative
add = 'w = GB_IMIN (w, t)' ;
codegen_axb_method ('min', multop, add, imult, 'int8_t'  , 'int8_t'  , 'INT8_MAX'  , 'INT8_MIN'  , 0) ;
codegen_axb_method ('min', multop, add, imult, 'int16_t' , 'int16_t' , 'INT16_MAX' , 'INT16_MIN' , 0) ;
codegen_axb_method ('min', multop, add, imult, 'int32_t' , 'int32_t' , 'INT32_MAX' , 'INT32_MIN' , 0) ;
codegen_axb_method ('min', multop, add, imult, 'int64_t' , 'int64_t' , 'INT64_MAX' , 'INT64_MIN' , 0) ;
codegen_axb_method ('min', multop, add, imult, 'uint8_t' , 'uint8_t' , 'UINT8_MAX' , '0'         , 0) ;
codegen_axb_method ('min', multop, add, imult, 'uint16_t', 'uint16_t', 'UINT16_MAX', '0'         , 0) ;
codegen_axb_method ('min', multop, add, imult, 'uint32_t', 'uint32_t', 'UINT32_MAX', '0'         , 0) ;
codegen_axb_method ('min', multop, add, imult, 'uint64_t', 'uint64_t', 'UINT64_MAX', '0'         , 0) ;
add = 'w = fminf (w, t)' ;
codegen_axb_method ('min', multop, add, fmult, 'float'   , 'float'   , 'INFINITY'  , '(-INFINITY)' , 0) ;
add = 'w = fmin (w, t)' ;
codegen_axb_method ('min', multop, add, dmult, 'double'  , 'double'  , ....
        '((double) INFINITY)'  , '((double) -INFINITY)' , 0) ;

% max monoid: all are terminal
% TODO: find atomic alternative
add = 'w = GB_IMAX (w, t)' ;
codegen_axb_method ('max', multop, add, imult, 'int8_t'  , 'int8_t'  , 'INT8_MIN'  , 'INT8_MAX'  , 0) ;
codegen_axb_method ('max', multop, add, imult, 'int16_t' , 'int16_t' , 'INT16_MIN' , 'INT16_MAX' , 0) ;
codegen_axb_method ('max', multop, add, imult, 'int32_t' , 'int32_t' , 'INT32_MIN' , 'INT32_MAX' , 0) ;
codegen_axb_method ('max', multop, add, imult, 'int64_t' , 'int64_t' , 'INT64_MIN' , 'INT64_MAX' , 0) ;
codegen_axb_method ('max', multop, add, imult, 'uint8_t' , 'uint8_t' , '0'         , 'UINT8_MAX' , 0) ;
codegen_axb_method ('max', multop, add, imult, 'uint16_t', 'uint16_t', '0'         , 'UINT16_MAX', 0) ;
codegen_axb_method ('max', multop, add, imult, 'uint32_t', 'uint32_t', '0'         , 'UINT32_MAX', 0) ;
codegen_axb_method ('max', multop, add, imult, 'uint64_t', 'uint64_t', '0'         , 'UINT64_MAX', 0) ;
add = 'w = fmaxf (w, t)' ;
codegen_axb_method ('max', multop, add, fmult, 'float'   , 'float'   , '(-INFINITY)' , 'INFINITY', 0) ;
add = 'w = fmax (w, t)' ;
codegen_axb_method ('max', multop, add, dmult, 'double'  , 'double'  , ...
        '((double) -INFINITY)'  , '((double) INFINITY)' , 0) ;

% plus monoid: none are terminal, all are atomic.
add = 'w += t' ;
codegen_axb_method ('plus', multop, add, imult, 'int8_t'  , 'int8_t'  , '0', [ ], 1) ;
codegen_axb_method ('plus', multop, add, imult, 'uint8_t' , 'uint8_t' , '0', [ ], 1) ;
codegen_axb_method ('plus', multop, add, imult, 'int16_t' , 'int16_t' , '0', [ ], 1) ;
codegen_axb_method ('plus', multop, add, imult, 'uint16_t', 'uint16_t', '0', [ ], 1) ;
codegen_axb_method ('plus', multop, add, imult, 'int32_t' , 'int32_t' , '0', [ ], 1) ;
codegen_axb_method ('plus', multop, add, imult, 'uint32_t', 'uint32_t', '0', [ ], 1) ;
codegen_axb_method ('plus', multop, add, imult, 'int64_t' , 'int64_t' , '0', [ ], 1) ;
codegen_axb_method ('plus', multop, add, imult, 'uint64_t', 'uint64_t', '0', [ ], 1) ;
codegen_axb_method ('plus', multop, add, fmult, 'float'   , 'float'   , '0', [ ], 1) ;
codegen_axb_method ('plus', multop, add, dmult, 'double'  , 'double'  , '0', [ ], 1) ;

% times monoid: integers are terminal, float and double are not.  All are atomic.
add = 'w *= t' ;
codegen_axb_method ('times', multop, add, imult, 'int8_t'  , 'int8_t'  , '1', '0', 1) ;
codegen_axb_method ('times', multop, add, imult, 'uint8_t' , 'uint8_t' , '1', '0', 1) ;
codegen_axb_method ('times', multop, add, imult, 'int16_t' , 'int16_t' , '1', '0', 1) ;
codegen_axb_method ('times', multop, add, imult, 'uint16_t', 'uint16_t', '1', '0', 1) ;
codegen_axb_method ('times', multop, add, imult, 'int32_t' , 'int32_t' , '1', '0', 1) ;
codegen_axb_method ('times', multop, add, imult, 'uint32_t', 'uint32_t', '1', '0', 1) ;
codegen_axb_method ('times', multop, add, imult, 'int64_t' , 'int64_t' , '1', '0', 1) ;
codegen_axb_method ('times', multop, add, imult, 'uint64_t', 'uint64_t', '1', '0', 1) ;
codegen_axb_method ('times', multop, add, fmult, 'float'   , 'float'   , '1', [ ], 1) ;
codegen_axb_method ('times', multop, add, dmult, 'double'  , 'double'  , '1', [ ], 1) ;

% boolean monoids: lor, land are terminal; lxor, eq are not
% TODO not yet atomic
if (~isempty (bmult))
    codegen_axb_method ('lor',  multop, 'w = (w || t)', bmult, 'bool', 'bool', 'false', 'true' , 0) ;
    codegen_axb_method ('land', multop, 'w = (w && t)', bmult, 'bool', 'bool', 'true' , 'false', 0) ;
    codegen_axb_method ('lxor', multop, 'w = (w != t)', bmult, 'bool', 'bool', 'false', [ ]    , 0) ;
    % TODO also called lxnor; can it be atomic?
    codegen_axb_method ('eq',   multop, 'w = (w == t)', bmult, 'bool', 'bool', 'true' , [ ]    , 0) ;
end

