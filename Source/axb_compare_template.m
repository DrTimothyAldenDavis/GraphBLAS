function axb_compare_template (multop, do_boolean, mult)
%AXB_COMPARE_TEMPLATE create a function for a semiring with a TxT->bool multiplier

fprintf ('\n%-7s', multop) ;

% lor monoid
add = 'w = (w || t)' ;
if (do_boolean)
axb_method ('lor', multop, add, mult, 'bool', 'bool'    , 'false', 'true') ;
end
axb_method ('lor', multop, add, mult, 'bool', 'int8_t'  , 'false', 'true') ;
axb_method ('lor', multop, add, mult, 'bool', 'uint8_t' , 'false', 'true') ;
axb_method ('lor', multop, add, mult, 'bool', 'int16_t' , 'false', 'true') ;
axb_method ('lor', multop, add, mult, 'bool', 'uint16_t', 'false', 'true') ;
axb_method ('lor', multop, add, mult, 'bool', 'int32_t' , 'false', 'true') ;
axb_method ('lor', multop, add, mult, 'bool', 'uint32_t', 'false', 'true') ;
axb_method ('lor', multop, add, mult, 'bool', 'int64_t' , 'false', 'true') ;
axb_method ('lor', multop, add, mult, 'bool', 'uint64_t', 'false', 'true') ;
axb_method ('lor', multop, add, mult, 'bool', 'float'   , 'false', 'true') ;
axb_method ('lor', multop, add, mult, 'bool', 'double'  , 'false', 'true') ;

% land monoid
add = 'w = (w && t)' ;
if (do_boolean)
axb_method ('land', multop, add, mult, 'bool', 'bool'    , 'true', 'false') ;
end
axb_method ('land', multop, add, mult, 'bool', 'int8_t'  , 'true', 'false') ;
axb_method ('land', multop, add, mult, 'bool', 'uint8_t' , 'true', 'false') ;
axb_method ('land', multop, add, mult, 'bool', 'int16_t' , 'true', 'false') ;
axb_method ('land', multop, add, mult, 'bool', 'uint16_t', 'true', 'false') ;
axb_method ('land', multop, add, mult, 'bool', 'int32_t' , 'true', 'false') ;
axb_method ('land', multop, add, mult, 'bool', 'uint32_t', 'true', 'false') ;
axb_method ('land', multop, add, mult, 'bool', 'int64_t' , 'true', 'false') ;
axb_method ('land', multop, add, mult, 'bool', 'uint64_t', 'true', 'false') ;
axb_method ('land', multop, add, mult, 'bool', 'float'   , 'true', 'false') ;
axb_method ('land', multop, add, mult, 'bool', 'double'  , 'true', 'false') ;

% lxor monoid
add = 'w = (w != t)' ;
if (do_boolean)
axb_method ('lxor', multop, add, mult, 'bool', 'bool'    , 'false', [ ]) ;
end
axb_method ('lxor', multop, add, mult, 'bool', 'int8_t'  , 'false', [ ]) ;
axb_method ('lxor', multop, add, mult, 'bool', 'uint8_t' , 'false', [ ]) ;
axb_method ('lxor', multop, add, mult, 'bool', 'int16_t' , 'false', [ ]) ;
axb_method ('lxor', multop, add, mult, 'bool', 'uint16_t', 'false', [ ]) ;
axb_method ('lxor', multop, add, mult, 'bool', 'int32_t' , 'false', [ ]) ;
axb_method ('lxor', multop, add, mult, 'bool', 'uint32_t', 'false', [ ]) ;
axb_method ('lxor', multop, add, mult, 'bool', 'int64_t' , 'false', [ ]) ;
axb_method ('lxor', multop, add, mult, 'bool', 'uint64_t', 'false', [ ]) ;
axb_method ('lxor', multop, add, mult, 'bool', 'float'   , 'false', [ ]) ;
axb_method ('lxor', multop, add, mult, 'bool', 'double'  , 'false', [ ]) ;

% eq monoid
add = 'w = (w == t)' ;
if (do_boolean)
axb_method ('eq', multop, add, mult, 'bool', 'bool'    , 'true', [ ]) ;
end
axb_method ('eq', multop, add, mult, 'bool', 'int8_t'  , 'true', [ ]) ;
axb_method ('eq', multop, add, mult, 'bool', 'uint8_t' , 'true', [ ]) ;
axb_method ('eq', multop, add, mult, 'bool', 'int16_t' , 'true', [ ]) ;
axb_method ('eq', multop, add, mult, 'bool', 'uint16_t', 'true', [ ]) ;
axb_method ('eq', multop, add, mult, 'bool', 'int32_t' , 'true', [ ]) ;
axb_method ('eq', multop, add, mult, 'bool', 'uint32_t', 'true', [ ]) ;
axb_method ('eq', multop, add, mult, 'bool', 'int64_t' , 'true', [ ]) ;
axb_method ('eq', multop, add, mult, 'bool', 'uint64_t', 'true', [ ]) ;
axb_method ('eq', multop, add, mult, 'bool', 'float'   , 'true', [ ]) ;
axb_method ('eq', multop, add, mult, 'bool', 'double'  , 'true', [ ]) ;

