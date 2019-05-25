function nth = gb
%GB print info about the GraphBLAS version
%
% nthreads_max = gb

[nthreads_max threading thread_safety format hyperratio ... 
name version date about license compiledate compiletime api api_about] ... 
 = GB_mex_init ;

d = stat ;

fprintf ('\n%s version: %d.%d.%d\n', name, version) ;

if (d)
    fprintf ('    malloc debug: on\n') ;
else
    fprintf ('    malloc debug: off\n') ;
end

fprintf ('    max nthreads: %d\n', nthreads_max) ;

nthreads = nthreads_get ;
fprintf ('    # of threads to use (nthreads_get/set): %d\n', nthreads) ;

ncores = feature ('numcores') ;
fprintf ('    # of cores for MATLAB: %d\n', ncores) ;


switch (threading)
    case {0}
        fprintf ('    no internal threading\n') ;
    case {1}
        fprintf ('    OpenMP for internal threads\n') ;
    otherwise
        error ('?') ;
end

switch (thread_safety)
    case {0}
        fprintf ('    no thread safety\n') ;
    case {1}
        fprintf ('    OpenMP for user thread safety\n') ;
    case {2}
        fprintf ('    POSIX for user thread safety\n') ;
    otherwise
        error ('?') ;
end

switch (format)
    case {0}
        fprintf ('    default format: CSR\n') ;
    case {1}
        fprintf ('    default format: CSC\n') ;
    otherwise
        error ('?') ;
end

fprintf ('    hyperratio: %g\n', hyperratio) ;
fprintf ('    date: %s\n', date) ;
fprintf ('    compile date: %s\n', compiledate) ;
fprintf ('    compile time: %s\n\n', compiletime) ;

if (nargout > 0)
    fprintf ('\n---------------------------------\n%s', about) ;
    fprintf ('\n---------------------------------\n%s', license) ;
    fprintf ('\n---------------------------------\n%s\n', api_about) ;
    fprintf ('API version: %d.%d.%d\n', api) ;
    nth = nthreads_max ;
end

