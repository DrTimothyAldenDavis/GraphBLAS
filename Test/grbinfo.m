function [nth chnk] = grbinfo
%GRBINFO print info about the GraphBLAS version
%
% nthreads = grbinfo

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[nthreads, ~, ~, format, hyper_switch,
name, version, date, about, license, compiledate, compiletime, api, ...
api_about, chunk ] = GB_mex_init ;

d = stat ;

fprintf ('\n%s version: %d.%d.%d\n', name, version) ;

if (d)
    fprintf ('    malloc debug: on\n') ;
else
    fprintf ('    malloc debug: off\n') ;
end

ncores = feature ('numcores') ;
[nthreads chunk] = nthreads_get ;

fprintf ('    # of threads to use:   %d\n', nthreads) ;
fprintf ('    chunk:                 %g\n', chunk) ;
fprintf ('    OpenMP max threads:    %d\n', GB_mex_omp_max_threads) ;
fprintf ('    # of cores for MATLAB: %d\n', ncores) ;

switch (format)
    case {0}
        fprintf ('    default format: CSR\n') ;
    case {1}
        fprintf ('    default format: CSC\n') ;
    otherwise
        error ('?') ;
end

fprintf ('    hyper_switch: %g\n', hyper_switch) ;
fprintf ('    date: %s\n', date) ;
fprintf ('    compile date: %s\n', compiledate) ;
fprintf ('    compile time: %s\n\n', compiletime) ;

if (nargout > 0)
    fprintf ('\n---------------------------------\n%s', about) ;
    fprintf ('\n---------------------------------\n%s', license) ;
    fprintf ('\n---------------------------------\n%s\n', api_about) ;
    fprintf ('API version: %d.%d.%d\n', api) ;
    nth = nthreads ;
    chnk = chunk ;
end

