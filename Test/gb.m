% function nth = gb
%GB print info about the GraphBLAS version
%
% nthreads_max = gb

[nthreads_max threading thread_safety format hyperratio ... 
name version date about license compiledate compiletime api api_about] ... 
 = GB_mex_init ;

fprintf ('\nGraphBLAS:\n    max threads: %d\n', nthreads_max) ;

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

fprintf ('    hyperratio: %g\n\n', hyperratio) ;
fprintf ('    name: %s\n', name) ;
fprintf ('    version: %d.%d.%d\n', version) ;
fprintf ('    date: %s\n', date) ;

fprintf ('    compile date: %s\n', compiledate) ;
fprintf ('    compile time: %s\n', compiletime) ;
fprintf ('    API version: %d.%d.%d\n', api) ;


fprintf ('\n---------------------------------\n%s', about) ;
fprintf ('\n---------------------------------\n%s', license) ;
fprintf ('\n---------------------------------\n%s\n', api_about) ;

% if (nargout > 0)
    nth = nthreads_max ;
% end

