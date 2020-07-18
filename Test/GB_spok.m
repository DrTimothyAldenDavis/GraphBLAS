function ok = GB_spok (A)
%GB_SPOK check if a matrix is valid
if (issparse (A))
    ok = spok (A) ;
else
    ok = true ;
end
