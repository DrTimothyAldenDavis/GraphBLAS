function axb_compare_template (multop, do_boolean, imult, fmult)
%AXB_COMPARE_TEMPLATE create a function for a semiring with a TxT->bool multiplier

if (nargin < 4)
    fmult = imult ;
end

% lor monoid
add = 'w = (w || t)' ;
if (do_boolean)
axb_method ('lor', multop, add, imult, 'bool', 'bool'    , 'false', 'true', 0) ;
end
axb_method ('lor', multop, add, imult, 'bool', 'int8_t'  , 'false', 'true', 0) ;
axb_method ('lor', multop, add, imult, 'bool', 'uint8_t' , 'false', 'true', 0) ;
axb_method ('lor', multop, add, imult, 'bool', 'int16_t' , 'false', 'true', 0) ;
axb_method ('lor', multop, add, imult, 'bool', 'uint16_t', 'false', 'true', 0) ;
axb_method ('lor', multop, add, imult, 'bool', 'int32_t' , 'false', 'true', 0) ;
axb_method ('lor', multop, add, imult, 'bool', 'uint32_t', 'false', 'true', 0) ;
axb_method ('lor', multop, add, imult, 'bool', 'int64_t' , 'false', 'true', 0) ;
axb_method ('lor', multop, add, imult, 'bool', 'uint64_t', 'false', 'true', 0) ;
axb_method ('lor', multop, add, fmult, 'bool', 'float'   , 'false', 'true', 0) ;
axb_method ('lor', multop, add, fmult, 'bool', 'double'  , 'false', 'true', 0) ;

% land monoid
add = 'w = (w && t)' ;
if (do_boolean)
axb_method ('land', multop, add, imult, 'bool', 'bool'    , 'true', 'false', 0) ;
end
axb_method ('land', multop, add, imult, 'bool', 'int8_t'  , 'true', 'false', 0) ;
axb_method ('land', multop, add, imult, 'bool', 'uint8_t' , 'true', 'false', 0) ;
axb_method ('land', multop, add, imult, 'bool', 'int16_t' , 'true', 'false', 0) ;
axb_method ('land', multop, add, imult, 'bool', 'uint16_t', 'true', 'false', 0) ;
axb_method ('land', multop, add, imult, 'bool', 'int32_t' , 'true', 'false', 0) ;
axb_method ('land', multop, add, imult, 'bool', 'uint32_t', 'true', 'false', 0) ;
axb_method ('land', multop, add, imult, 'bool', 'int64_t' , 'true', 'false', 0) ;
axb_method ('land', multop, add, imult, 'bool', 'uint64_t', 'true', 'false', 0) ;
axb_method ('land', multop, add, fmult, 'bool', 'float'   , 'true', 'false', 0) ;
axb_method ('land', multop, add, fmult, 'bool', 'double'  , 'true', 'false', 0) ;

% lxor monoid
add = 'w = (w != t)' ;
if (do_boolean)
axb_method ('lxor', multop, add, imult, 'bool', 'bool'    , 'false', [ ], 0) ;
end
axb_method ('lxor', multop, add, imult, 'bool', 'int8_t'  , 'false', [ ], 0) ;
axb_method ('lxor', multop, add, imult, 'bool', 'uint8_t' , 'false', [ ], 0) ;
axb_method ('lxor', multop, add, imult, 'bool', 'int16_t' , 'false', [ ], 0) ;
axb_method ('lxor', multop, add, imult, 'bool', 'uint16_t', 'false', [ ], 0) ;
axb_method ('lxor', multop, add, imult, 'bool', 'int32_t' , 'false', [ ], 0) ;
axb_method ('lxor', multop, add, imult, 'bool', 'uint32_t', 'false', [ ], 0) ;
axb_method ('lxor', multop, add, imult, 'bool', 'int64_t' , 'false', [ ], 0) ;
axb_method ('lxor', multop, add, imult, 'bool', 'uint64_t', 'false', [ ], 0) ;
axb_method ('lxor', multop, add, fmult, 'bool', 'float'   , 'false', [ ], 0) ;
axb_method ('lxor', multop, add, fmult, 'bool', 'double'  , 'false', [ ], 0) ;

% eq monoid
add = 'w = (w == t)' ;
if (do_boolean)
axb_method ('eq', multop, add, imult, 'bool', 'bool'    , 'true', [ ], 0) ;
end
axb_method ('eq', multop, add, imult, 'bool', 'int8_t'  , 'true', [ ], 0) ;
axb_method ('eq', multop, add, imult, 'bool', 'uint8_t' , 'true', [ ], 0) ;
axb_method ('eq', multop, add, imult, 'bool', 'int16_t' , 'true', [ ], 0) ;
axb_method ('eq', multop, add, imult, 'bool', 'uint16_t', 'true', [ ], 0) ;
axb_method ('eq', multop, add, imult, 'bool', 'int32_t' , 'true', [ ], 0) ;
axb_method ('eq', multop, add, imult, 'bool', 'uint32_t', 'true', [ ], 0) ;
axb_method ('eq', multop, add, imult, 'bool', 'int64_t' , 'true', [ ], 0) ;
axb_method ('eq', multop, add, imult, 'bool', 'uint64_t', 'true', [ ], 0) ;
axb_method ('eq', multop, add, fmult, 'bool', 'float'   , 'true', [ ], 0) ;
axb_method ('eq', multop, add, fmult, 'bool', 'double'  , 'true', [ ], 0) ;

