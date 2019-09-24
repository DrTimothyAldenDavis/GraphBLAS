function gbtest8
%GBTEST8 test gb.select

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

%   tril
%   triu
%   diag
%   offdiag

%   nonzero     ~=0
%   eqzero      ==0
%   gtzero      >0
%   gezero      >=0
%   ltzero      <0
%   lezero      <=0

%   nethunk     ~=thunk
%   eqthunk     ==thunk
%   gtthunk     >thunk
%   gethunk     >=thunk
%   ltthunk     <thunk
%   lethunk     <=thunk

rng ('default') ;
n = 5 ;
m = 8 ;
A = sparse (10 * rand (m,n) - 5) .* sprand (m, n, 0.8) ;

thunk = 0.5 ;
desc.kind = 'sparse' ;

A (1,1) = thunk ;
A (2,2) = -thunk ;
A (3,4) = thunk ;

%-------------------------------------------------------------------------------
% tril
%-------------------------------------------------------------------------------

    C1 = tril (A) ;
    C2 = gb.select ('tril', A) ;
    assert (gbtest_eq (C1, C2))
    for k = -m:n
        C1 = tril (A, k) ;
        C2 = gb.select ('tril', A, k) ;
        C3 = gb.select ('tril', A, k, desc) ;
        assert (gbtest_eq (C1, C2))
        assert (gbtest_eq (C1, C3))
        assert (isequal (class (C3), 'double')) ;
    end
    C1 = tril (A, 0) ;
    C2 = gb.select ('tril', A, 0) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% triu
%-------------------------------------------------------------------------------

    C1 = triu (A) ;
    C2 = gb.select ('triu', A) ;
    assert (gbtest_eq (C1, C2))
    for k = -m:n
        C1 = triu (A, k) ;
        C2 = gb.select ('triu', A, k) ;
        assert (gbtest_eq (C1, C2))
    end
    C1 = triu (A, 0) ;
    C2 = gb.select ('triu', A, 0) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% diag
%-------------------------------------------------------------------------------

    d = min (m,n) ;
    C1 = A .* spdiags (ones (d,1), 0, m, n) ;
    C2 = gb.select ('diag', A) ;
    assert (gbtest_eq (C1, C2))
    for k = -m:n
        C1 = A .* spdiags (ones (d,1), k, m, n) ;
        C2 = gb.select ('diag', A, k) ;
        assert (gbtest_eq (C1, C2))
    end
    C1 = A .* spdiags (ones (d,1), 0, m, n) ;
    C2 = gb.select ('diag', A, 0) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% offdiag
%-------------------------------------------------------------------------------

    d = min (m,n) ;
    C1 = A .* (1 - spdiags (ones (d,1), 0, m, n)) ;
    C2 = gb.select ('offdiag', A) ;
    assert (gbtest_eq (C1, C2))
    for k = -m:n
        C1 = A .* (1 - spdiags (ones (d,1), k, m, n)) ;
        C2 = gb.select ('offdiag', A, k) ;
        assert (gbtest_eq (C1, C2))
    end
    C1 = A .* (1 - spdiags (ones (d,1), 0, m, n)) ;
    C2 = gb.select ('offdiag', A, 0) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% nonzero
%-------------------------------------------------------------------------------

    % all explicit entries in the MATLAB sparse matrix are nonzero,
    % so this does nothing.  A better test would be to compute a GraphBLAS
    % matrix with explicit zeros first.

    %   nonzero     ~=0

    M = (A ~= 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('nonzero', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = gb.select ('~=0', A) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% eqzero
%-------------------------------------------------------------------------------

    % all explicit entries in the MATLAB sparse matrix are nonzero,
    % so this does nothing.

    %   eqzero      ==0

    C1 = sparse (m,n) ;

    C2 = gb.select ('eqzero', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = gb.select ('==0', A) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% gtzero
%-------------------------------------------------------------------------------

    %   gtzero      >0

    M = (A > 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('gtzero', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = gb.select ('>0', A) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% gezero
%-------------------------------------------------------------------------------

    %   gezero      >=0

    M = (A >= 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('gezero', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = gb.select ('>=0', A) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% ltzero
%-------------------------------------------------------------------------------

    %   ltzero      <0

    M = (A < 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('ltzero', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = gb.select ('<0', A) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% lezero
%-------------------------------------------------------------------------------

    %   lezero      <=0

    M = (A <= 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('lezero', A) ;
    assert (gbtest_eq (C1, C2))

    C2 = gb.select ('<=0', A) ;
    assert (gbtest_eq (C1, C2))


%-------------------------------------------------------------------------------
% nonthunk
%-------------------------------------------------------------------------------

    %   nonthunk     ~=thunk

    M = (A ~= thunk) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('nethunk', A, thunk) ;
    assert (gbtest_eq (C1, C2))

    C2 = gb.select ('~=thunk', A, thunk) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% eqthunk
%-------------------------------------------------------------------------------

    %   eqthunk      ==thunk

    M = (A == thunk) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('eqthunk', A, thunk) ;
    assert (gbtest_eq (C1, C2))

    C2 = gb.select ('==thunk', A, thunk) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% gtthunk
%-------------------------------------------------------------------------------

    %   gtthunk      >thunk

    M = (A > thunk) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('gtthunk', A, thunk) ;
    assert (gbtest_eq (C1, C2))

    C2 = gb.select ('>thunk', A, thunk) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% gethunk
%-------------------------------------------------------------------------------

    %   gethunk      >=thunk

    M = (A >= thunk) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('gethunk', A, thunk) ;
    assert (gbtest_eq (C1, C2))

    C2 = gb.select ('>=thunk', A, thunk) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% ltthunk
%-------------------------------------------------------------------------------

    %   ltthunk      <thunk

    M = (A < thunk) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('ltthunk', A, thunk) ;
    assert (gbtest_eq (C1, C2))

    C2 = gb.select ('<thunk', A, thunk) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% lethunk
%-------------------------------------------------------------------------------

    %   lethunk      <=thunk

    M = (A <= thunk) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('lethunk', A, thunk) ;
    assert (gbtest_eq (C1, C2))

    C2 = gb.select ('<=thunk', A, thunk) ;
    assert (gbtest_eq (C1, C2))

%-------------------------------------------------------------------------------
% gtzero, with mask and accum
%-------------------------------------------------------------------------------

    Cin = sprand (m, n, 0.5) ;
    C2 = gb.select (Cin, '+', '>0', A) ;
    C1 = Cin ;
    C1 (A > 0) = C1 (A > 0) + A (A > 0) ; 
    assert (gbtest_eq (C1, C2))

    M = logical (sprand (m, n, 0.5)) ;
    Cin = sprand (m, n, 0.5) ;
    C2 = gb.select (Cin, M, '>0', A) ;
    C1 = Cin ;
    T = sparse (m, n) ;
    T (A > 0) = A (A > 0) ;
    C1 (M) = T (M) ;
    assert (gbtest_eq (C1, C2))

    C2 = gb.select (Cin, M, '+', '>0', A) ;
    C1 = Cin ;
    T = sparse (m, n) ;
    T (A > 0) = A (A > 0) ;
    C1 (M) = C1 (M) + T (M) ;
    assert (gbtest_eq (C1, C2))

fprintf ('gbtest8: all tests passed\n') ;

