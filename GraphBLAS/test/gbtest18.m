function gbtest18
%GBTEST18 test comparators (and, or, >, ...)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
for trial = 1:100

    if (mod (trial, 5) == 0)
        fprintf ('.') ;
    end

    MA = sprand (4,5, 0.5) ;    A (2,2) = 2 ;
    MB = sprand (4,5, 0.5) ;    B (2,2) = 2 ;
    GA = gb (MA) ;
    GB = gb (MB) ;

    C1 = (MA < MB) ;
    C2 = (GA < GB) ;
    C3 = (MA < GB) ;
    C4 = (GA < MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA <= MB) ;
    C2 = (GA <= GB) ;
    C3 = (MA <= GB) ;
    C4 = (GA <= MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA > MB) ;
    C2 = (GA > GB) ;
    C3 = (MA > GB) ;
    C4 = (GA > MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA >= MB) ;
    C2 = (GA >= GB) ;
    C3 = (MA >= GB) ;
    C4 = (GA >= MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA == MB) ;
    C2 = (GA == GB) ;
    C3 = (MA == GB) ;
    C4 = (GA == MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA ~= MB) ;
    C2 = (GA ~= GB) ;
    C3 = (MA ~= GB) ;
    C4 = (GA ~= MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA .^ MB) ;
    C2 = (GA .^ GB) ;
    C3 = (MA .^ GB) ;
    C4 = (GA .^ MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA & MB) ;
    C2 = (GA & GB) ;
    C3 = (MA & GB) ;
    C4 = (GA & MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA | MB) ;
    C2 = (GA | GB) ;
    C3 = (MA | GB) ;
    C4 = (GA | MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = ~MA ;
    C2 = ~GA ;
    assert (gbtest_eq (C1, C2)) ;

    thunk = (trial - 5) / 10 ;
    gbthunk = gb (thunk) ;

    assert (gbtest_eq (MA <  thunk, GA <  thunk)) ;
    assert (gbtest_eq (MA <= thunk, GA <= thunk)) ;
    assert (gbtest_eq (MA >  thunk, GA >  thunk)) ;
    assert (gbtest_eq (MA >= thunk, GA >= thunk)) ;
    assert (gbtest_eq (MA == thunk, GA == thunk)) ;
    assert (gbtest_eq (MA ~= thunk, GA ~= thunk)) ;
    assert (gbtest_eq (MA .^ thunk, GA .^ thunk)) ;

    assert (gbtest_eq (MA <  thunk, GA <  gbthunk)) ;
    assert (gbtest_eq (MA <= thunk, GA <= gbthunk)) ;
    assert (gbtest_eq (MA >  thunk, GA >  gbthunk)) ;
    assert (gbtest_eq (MA >= thunk, GA >= gbthunk)) ;
    assert (gbtest_eq (MA == thunk, GA == gbthunk)) ;
    assert (gbtest_eq (MA ~= thunk, GA ~= gbthunk)) ;
    assert (gbtest_eq (MA .^ thunk, GA .^ gbthunk)) ;

    assert (gbtest_eq (MA <  thunk, MA <  gbthunk)) ;
    assert (gbtest_eq (MA <= thunk, MA <= gbthunk)) ;
    assert (gbtest_eq (MA >  thunk, MA >  gbthunk)) ;
    assert (gbtest_eq (MA >= thunk, MA >= gbthunk)) ;
    assert (gbtest_eq (MA == thunk, MA == gbthunk)) ;
    assert (gbtest_eq (MA ~= thunk, MA ~= gbthunk)) ;
    assert (gbtest_eq (MA .^ thunk, MA .^ gbthunk)) ;

    assert (gbtest_eq (thunk <  MA, thunk <  GA)) ;
    assert (gbtest_eq (thunk <= MA, thunk <= GA)) ;
    assert (gbtest_eq (thunk >  MA, thunk >  GA)) ;
    assert (gbtest_eq (thunk >= MA, thunk >= GA)) ;
    assert (gbtest_eq (thunk == MA, thunk == GA)) ;
    assert (gbtest_eq (thunk ~= MA, thunk ~= GA)) ;
    if (thunk >= 0)
        assert (gbtest_eq (thunk .^ MA, thunk .^ GA)) ;
    end

    assert (gbtest_eq (thunk <  MA, gbthunk <  GA)) ;
    assert (gbtest_eq (thunk <= MA, gbthunk <= GA)) ;
    assert (gbtest_eq (thunk >  MA, gbthunk >  GA)) ;
    assert (gbtest_eq (thunk >= MA, gbthunk >= GA)) ;
    assert (gbtest_eq (thunk == MA, gbthunk == GA)) ;
    assert (gbtest_eq (thunk ~= MA, gbthunk ~= GA)) ;
    if (thunk >= 0)
        assert (gbtest_eq (thunk .^ MA, gbthunk .^ GA)) ;
    end

    assert (gbtest_eq (thunk <  MA, gbthunk <  MA)) ;
    assert (gbtest_eq (thunk <= MA, gbthunk <= MA)) ;
    assert (gbtest_eq (thunk >  MA, gbthunk >  MA)) ;
    assert (gbtest_eq (thunk >= MA, gbthunk >= MA)) ;
    assert (gbtest_eq (thunk == MA, gbthunk == MA)) ;
    assert (gbtest_eq (thunk ~= MA, gbthunk ~= MA)) ;
    if (thunk >= 0)
        assert (gbtest_eq (thunk .^ MA, gbthunk .^ MA)) ;
    end

    k = (mod (trial,2) == 0) ;
    gbk = gb (k) ;

    C1 = (MA & k) ;
    C2 = (GA & gbk) ;
    C3 = (MA & gbk) ;
    C4 = (GA & k) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (k   & MA) ;
    C2 = (gbk & GA) ;
    C3 = (gbk & MA) ;
    C4 = (k   & GA) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (MA | k) ;
    C2 = (GA | gbk) ;
    C3 = (MA | gbk) ;
    C4 = (GA | k) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = xor (MA, k) ;
    C2 = xor (GA, gbk) ;
    C3 = xor (MA, gbk) ;
    C4 = xor (GA, k) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (k   | MA) ;
    C2 = (gbk | GA) ;
    C3 = (gbk | MA) ;
    C4 = (k   | GA) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = (k   .^ MA) ;
    C2 = (gbk .^ GA) ;
    C3 = (gbk .^ MA) ;
    C4 = (k   .^ GA) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

end

fprintf ('\ngbtest18: all tests passed\n') ;

