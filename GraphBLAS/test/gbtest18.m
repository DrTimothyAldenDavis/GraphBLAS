function gbtest18
%GBTEST18 test comparators (and, or, >, ...)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
for trial = 1:100

    if (mod (trial, 5) == 0)
        fprintf ('.') ;
    end

    if (mod (trial, 10) == 1)
        m = 1 ;
        n = 1 ;
    else
        m = 4 ;
        n = 5 ;
    end

    MA = sprand (m,n, 0.5) ;    A (2,2) = 2 ;
    MB = sprand (m,n, 0.5) ;    B (2,2) = 2 ;

    if (rand < 0.1)
        MA = logical (MA) ;
        MB = logical (MB) ;
    end

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

    if (~islogical (MA))
        C1 = (MA .^ MB) ;
        C2 = (GA .^ GB) ;
        C3 = (MA .^ GB) ;
        C4 = (GA .^ MB) ;
        assert (gbtest_eq (C1, C2)) ;
        assert (gbtest_eq (C1, C3)) ;
        assert (gbtest_eq (C1, C4)) ;
    end

    C1 = (MA & MB) ;
    C2 = (GA & GB) ;
    C3 = (MA & GB) ;
    C4 = (GA & MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;
    if (islogical (MA))
        % C1 = min (MA , MB) ;
        C2 = min (GA , GB) ;
        C3 = min (MA , GB) ;
        C4 = min (GA , MB) ;
        assert (gbtest_eq (C1, C2)) ;
        assert (gbtest_eq (C1, C3)) ;
        assert (gbtest_eq (C1, C4)) ;
    end

    C1 = (MA | MB) ;
    C2 = (GA | GB) ;
    C3 = (MA | GB) ;
    C4 = (GA | MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;
    if (islogical (MA))
        % C1 = max (MA , MB) ;
        C2 = max (GA , GB) ;
        C3 = max (MA , GB) ;
        C4 = max (GA , MB) ;
        assert (gbtest_eq (C1, C2)) ;
        assert (gbtest_eq (C1, C3)) ;
        assert (gbtest_eq (C1, C4)) ;
    end

    C1 = xor (MA , MB) ;
    C2 = xor (GA , GB) ;
    C3 = xor (MA , GB) ;
    C4 = xor (GA , MB) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    C1 = ~MA ;
    C2 = ~GA ;
    assert (gbtest_eq (C1, C2)) ;

    thunk = (trial - 5) / 10 ;
    if (islogical (MA))
        thunk = logical (thunk) ;
    end
    gbthunk = gb (thunk) ;

    assert (gbtest_eq (MA <  thunk, GA <  thunk)) ;
    assert (gbtest_eq (MA <= thunk, GA <= thunk)) ;
    assert (gbtest_eq (MA >  thunk, GA >  thunk)) ;
    assert (gbtest_eq (MA >= thunk, GA >= thunk)) ;
    assert (gbtest_eq (MA == thunk, GA == thunk)) ;
    assert (gbtest_eq (MA ~= thunk, GA ~= thunk)) ;
    if (~islogical (MA))
        assert (gbtest_eq (MA .^ thunk, GA .^ thunk)) ;
    end

    assert (gbtest_eq (MA <  thunk, GA <  gbthunk)) ;
    assert (gbtest_eq (MA <= thunk, GA <= gbthunk)) ;
    assert (gbtest_eq (MA >  thunk, GA >  gbthunk)) ;
    assert (gbtest_eq (MA >= thunk, GA >= gbthunk)) ;
    assert (gbtest_eq (MA == thunk, GA == gbthunk)) ;
    assert (gbtest_eq (MA ~= thunk, GA ~= gbthunk)) ;
    if (~islogical (MA))
        assert (gbtest_eq (MA .^ thunk, GA .^ gbthunk)) ;
    end

    assert (gbtest_eq (MA <  thunk, MA <  gbthunk)) ;
    assert (gbtest_eq (MA <= thunk, MA <= gbthunk)) ;
    assert (gbtest_eq (MA >  thunk, MA >  gbthunk)) ;
    assert (gbtest_eq (MA >= thunk, MA >= gbthunk)) ;
    assert (gbtest_eq (MA == thunk, MA == gbthunk)) ;
    assert (gbtest_eq (MA ~= thunk, MA ~= gbthunk)) ;
    if (~islogical (MA))
        assert (gbtest_eq (MA .^ thunk, MA .^ gbthunk)) ;
    end

    assert (gbtest_eq (thunk <  MA, thunk <  GA)) ;
    assert (gbtest_eq (thunk <= MA, thunk <= GA)) ;
    assert (gbtest_eq (thunk >  MA, thunk >  GA)) ;
    assert (gbtest_eq (thunk >= MA, thunk >= GA)) ;
    assert (gbtest_eq (thunk == MA, thunk == GA)) ;
    assert (gbtest_eq (thunk ~= MA, thunk ~= GA)) ;
    if (thunk >= 0 && ~islogical (MA))
        assert (gbtest_eq (thunk .^ MA, thunk .^ GA)) ;
    end

    assert (gbtest_eq (thunk <  MA, gbthunk <  GA)) ;
    assert (gbtest_eq (thunk <= MA, gbthunk <= GA)) ;
    assert (gbtest_eq (thunk >  MA, gbthunk >  GA)) ;
    assert (gbtest_eq (thunk >= MA, gbthunk >= GA)) ;
    assert (gbtest_eq (thunk == MA, gbthunk == GA)) ;
    assert (gbtest_eq (thunk ~= MA, gbthunk ~= GA)) ;
    if (thunk >= 0 && ~islogical (MA))
        assert (gbtest_eq (thunk .^ MA, gbthunk .^ GA)) ;
    end

    assert (gbtest_eq (thunk <  MA, gbthunk <  MA)) ;
    assert (gbtest_eq (thunk <= MA, gbthunk <= MA)) ;
    assert (gbtest_eq (thunk >  MA, gbthunk >  MA)) ;
    assert (gbtest_eq (thunk >= MA, gbthunk >= MA)) ;
    assert (gbtest_eq (thunk == MA, gbthunk == MA)) ;
    assert (gbtest_eq (thunk ~= MA, gbthunk ~= MA)) ;
    if (thunk >= 0 && ~islogical (MA))
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
    if (islogical (MA))
        % C1 = min (MA , k) ;
        C2 = min (GA , gbk) ;
        C3 = min (MA , gbk) ;
        C4 = min (GA , k) ;
        assert (gbtest_eq (C1, C2)) ;
        assert (gbtest_eq (C1, C3)) ;
        assert (gbtest_eq (C1, C4)) ;
    end

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

    C1 = xor (k   , MA) ;
    C2 = xor (gbk , GA) ;
    C3 = xor (gbk , MA) ;
    C4 = xor (k   , GA) ;
    assert (gbtest_eq (C1, C2)) ;
    assert (gbtest_eq (C1, C3)) ;
    assert (gbtest_eq (C1, C4)) ;

    if (~islogical (MA))
        C1 = (k   .^ MA) ;
        C2 = (gbk .^ GA) ;
        C3 = (gbk .^ MA) ;
        C4 = (k   .^ GA) ;
        assert (gbtest_eq (C1, C2)) ;
        assert (gbtest_eq (C1, C3)) ;
        assert (gbtest_eq (C1, C4)) ;
    end

end

fprintf ('\ngbtest18: all tests passed\n') ;

