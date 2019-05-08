% test GrB_reduce to vector and scalar
clear all

for k = [1 2 4]
    nthreads_set (k) ;
    fprintf ('\n=============== GrB_reduce to scalar tests: nthreads %d\n', k) ;
    test29
    fprintf ('\n=============== GrB_reduce to vector tests: nthreads %d\n', k) ;
    test66
    test95
    test14
    test24(0)
end
