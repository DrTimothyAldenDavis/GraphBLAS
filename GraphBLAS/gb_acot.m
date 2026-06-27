function C = gb_acot (ghb, G)
%GB_ACOT implements GrB/acot and GhB/acot.  Not user-callable.

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gzb_apply (ghb, 'atan', gzb_apply (ghb, 'minv', gzb_full (ghb, G, type))) ;

