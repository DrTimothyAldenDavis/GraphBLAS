function gbtest18
%GBTEST18 test comparators

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
    assert (isequal (C1, logical (C2))) ;
    assert (isequal (C1, logical (C3))) ;
    assert (isequal (C1, logical (C4))) ;

    C1 = (MA <= MB) ;
    C2 = (GA <= GB) ;
    C3 = (MA <= GB) ;
    C4 = (GA <= MB) ;
    assert (isequal (C1, logical (C2))) ;
    assert (isequal (C1, logical (C3))) ;
    assert (isequal (C1, logical (C4))) ;

    C1 = (MA > MB) ;
    C2 = (GA > GB) ;
    C3 = (MA > GB) ;
    C4 = (GA > MB) ;
    assert (isequal (C1, logical (C2))) ;
    assert (isequal (C1, logical (C3))) ;
    assert (isequal (C1, logical (C4))) ;

    C1 = (MA >= MB) ;
    C2 = (GA >= GB) ;
    C3 = (MA >= GB) ;
    C4 = (GA >= MB) ;
    assert (isequal (C1, logical (C2))) ;
    assert (isequal (C1, logical (C3))) ;
    assert (isequal (C1, logical (C4))) ;

    C1 = (MA == MB) ;
    C2 = (GA == GB) ;
    C3 = (MA == GB) ;
    C4 = (GA == MB) ;
    assert (isequal (C1, logical (C2))) ;
    assert (isequal (C1, logical (C3))) ;
    assert (isequal (C1, logical (C4))) ;

    C1 = (MA ~= MB) ;
    C2 = (GA ~= GB) ;
    C3 = (MA ~= GB) ;
    C4 = (GA ~= MB) ;
    assert (isequal (C1, logical (C2))) ;
    assert (isequal (C1, logical (C3))) ;
    assert (isequal (C1, logical (C4))) ;

    C1 = (MA .^ MB) ;
    C2 = (GA .^ GB) ;
    C3 = (MA .^ GB) ;
    C4 = (GA .^ MB) ;
    assert (isequal (C1, double (C2))) ;
    assert (isequal (C1, double (C3))) ;
    assert (isequal (C1, double (C4))) ;

    C1 = (MA & MB) ;
    C2 = (GA & GB) ;
    C3 = (MA & GB) ;
    C4 = (GA & MB) ;
    assert (isequal (C1, logical (C2))) ;
    assert (isequal (C1, logical (C3))) ;
    assert (isequal (C1, logical (C4))) ;

    C1 = (MA | MB) ;
    C2 = (GA | GB) ;
    C3 = (MA | GB) ;
    C4 = (GA | MB) ;
    assert (isequal (C1, logical (C2))) ;
    assert (isequal (C1, logical (C3))) ;
    assert (isequal (C1, logical (C4))) ;

    C1 = ~MA ;
    C2 = ~GA ;
    assert (isequal (C1, logical (C2))) ;

    thunk = (trial - 5) / 10 ;
    gbthunk = gb (thunk) ;

    assert (isequal (MA <  thunk, logical (GA <  thunk))) ;
    assert (isequal (MA <= thunk, logical (GA <= thunk))) ;
    assert (isequal (MA >  thunk, logical (GA >  thunk))) ;
    assert (isequal (MA >= thunk, logical (GA >= thunk))) ;
    assert (isequal (MA == thunk, logical (GA == thunk))) ;
    assert (isequal (MA ~= thunk, logical (GA ~= thunk))) ;
    assert (isequal (MA .^ thunk, double  (GA .^ thunk))) ;

    assert (isequal (MA <  thunk, logical (GA <  gbthunk))) ;
    assert (isequal (MA <= thunk, logical (GA <= gbthunk))) ;
    assert (isequal (MA >  thunk, logical (GA >  gbthunk))) ;
    assert (isequal (MA >= thunk, logical (GA >= gbthunk))) ;
    assert (isequal (MA == thunk, logical (GA == gbthunk))) ;
    assert (isequal (MA ~= thunk, logical (GA ~= gbthunk))) ;
    assert (isequal (MA .^ thunk, double  (GA .^ gbthunk))) ;

    assert (isequal (MA <  thunk, logical (MA <  gbthunk))) ;
    assert (isequal (MA <= thunk, logical (MA <= gbthunk))) ;
    assert (isequal (MA >  thunk, logical (MA >  gbthunk))) ;
    assert (isequal (MA >= thunk, logical (MA >= gbthunk))) ;
    assert (isequal (MA == thunk, logical (MA == gbthunk))) ;
    assert (isequal (MA ~= thunk, logical (MA ~= gbthunk))) ;
    assert (isequal (MA .^ thunk, double  (MA .^ gbthunk))) ;

    assert (isequal (thunk <  MA, logical (thunk <  GA))) ;
    assert (isequal (thunk <= MA, logical (thunk <= GA))) ;
    assert (isequal (thunk >  MA, logical (thunk >  GA))) ;
    assert (isequal (thunk >= MA, logical (thunk >= GA))) ;
    assert (isequal (thunk == MA, logical (thunk == GA))) ;
    assert (isequal (thunk ~= MA, logical (thunk ~= GA))) ;
    if (thunk >= 0)
        assert (isequal (thunk .^ MA, double (thunk .^ GA))) ;
    end

    assert (isequal (thunk <  MA, logical (gbthunk <  GA))) ;
    assert (isequal (thunk <= MA, logical (gbthunk <= GA))) ;
    assert (isequal (thunk >  MA, logical (gbthunk >  GA))) ;
    assert (isequal (thunk >= MA, logical (gbthunk >= GA))) ;
    assert (isequal (thunk == MA, logical (gbthunk == GA))) ;
    assert (isequal (thunk ~= MA, logical (gbthunk ~= GA))) ;
    if (thunk >= 0)
        assert (isequal (thunk .^ MA, double (gbthunk .^ GA))) ;
    end

    assert (isequal (thunk <  MA, logical (gbthunk <  MA))) ;
    assert (isequal (thunk <= MA, logical (gbthunk <= MA))) ;
    assert (isequal (thunk >  MA, logical (gbthunk >  MA))) ;
    assert (isequal (thunk >= MA, logical (gbthunk >= MA))) ;
    assert (isequal (thunk == MA, logical (gbthunk == MA))) ;
    assert (isequal (thunk ~= MA, logical (gbthunk ~= MA))) ;
    if (thunk >= 0)
        assert (isequal (thunk .^ MA, double (gbthunk .^ MA))) ;
    end

    k = (mod (trial,2) == 0) ;
    gbk = gb (k) ;

    C1 = (MA & k) ;
    C2 = (GA & gbk) ;
    C3 = (MA & gbk) ;
    C4 = (GA & k) ;
    assert (isequal (C1, logical (C2))) ;
    assert (isequal (C1, logical (C3))) ;
    assert (isequal (C1, logical (C4))) ;

    C1 = (k   & MA) ;
    C2 = (gbk & GA) ;
    C3 = (gbk & MA) ;
    C4 = (k   & GA) ;
    assert (isequal (C1, logical (C2))) ;
    assert (isequal (C1, logical (C3))) ;
    assert (isequal (C1, logical (C4))) ;

    C1 = (MA | k) ;
    C2 = (GA | gbk) ;
    C3 = (MA | gbk) ;
    C4 = (GA | k) ;
    assert (isequal (C1, logical (C2))) ;
    assert (isequal (C1, logical (C3))) ;
    assert (isequal (C1, logical (C4))) ;

    C1 = (k   | MA) ;
    C2 = (gbk | GA) ;
    C3 = (gbk | MA) ;
    C4 = (k   | GA) ;
    assert (isequal (C1, logical (C2))) ;
    assert (isequal (C1, logical (C3))) ;
    assert (isequal (C1, logical (C4))) ;

    C1 = (k   .^ MA) ;
    C2 = (gbk .^ GA) ;
    C3 = (gbk .^ MA) ;
    C4 = (k   .^ GA) ;
    assert (isequal (C1, double (C2))) ;
    assert (isequal (C1, double (C3))) ;
    assert (isequal (C1, double (C4))) ;

end

fprintf ('\ngbtest18: all tests passed\n') ;

