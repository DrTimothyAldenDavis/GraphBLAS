function ncores = feature_numcores
%FEATURE_NUMCORES determine # of cores the system has
have_octave = (exist ('OCTAVE_VERSION', 'builtin') == 5) ;
if (have_octave)
    ncores = nproc ;
else
    ncores = feature ('numcores') ;
end

