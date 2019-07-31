clear all

fprintf ('test8: select operators (tril, triu, diag, offdiag, ...)\n') ;

%   tril
%   triu
%   diag
%   offdiag

%   nonzero     ne0         !=0   ~=0
%   eqzero      eq0         ==0
%   gtzero      gt0         >0
%   gezero      ge0         >=0
%   ltzero      lt0         <0
%   lezero      le0         <=0

%   nethunk     !=thunk     ~=thunk
%   eqthunk     ==thunk
%   gtthunk     >thunk
%   gethunk     >=thunk
%   ltthunk     <thunk
%   lethunk     <=thunk

n = 5 ;
m = 8 ;
A = sparse (10 * rand (m,n) - 5) .* sprand (m, n, 0.8) ;

thunk = sparse (0.5) ;

A (1,1) = thunk ;
A (2,2) = -thunk ;
A (3,4) = thunk ;

%-------------------------------------------------------------------------------
% tril
%-------------------------------------------------------------------------------

    C1 = tril (A) ;
    C2 = gb.select ('tril', A) ;
    assert (isequal (C1, sparse (C2)))
    for k = -m:n
        C1 = tril (A, k) ;
        C2 = gb.select ('tril', A, sparse (k)) ;
        assert (isequal (C1, sparse (C2)))
    end
    C1 = tril (A, 0) ;
    C2 = gb.select ('tril', A, sparse (0)) ;
    assert (isequal (C1, sparse (C2)))

%-------------------------------------------------------------------------------
% triu
%-------------------------------------------------------------------------------

    C1 = triu (A) ;
    C2 = gb.select ('triu', A) ;
    assert (isequal (C1, sparse (C2)))
    for k = -m:n
        C1 = triu (A, k) ;
        C2 = gb.select ('triu', A, sparse (k)) ;
        assert (isequal (C1, sparse (C2)))
    end
    C1 = triu (A, 0) ;
    C2 = gb.select ('triu', A, sparse (0)) ;
    assert (isequal (C1, sparse (C2)))

%-------------------------------------------------------------------------------
% diag
%-------------------------------------------------------------------------------

    d = min (m,n) ;
    C1 = A .* spdiags (ones (d,1), 0, m, n) ;
    C2 = gb.select ('diag', A) ;
    assert (isequal (C1, sparse (C2)))
    for k = -m:n
        C1 = A .* spdiags (ones (d,1), k, m, n) ;
        C2 = gb.select ('diag', A, sparse (k)) ;
        assert (isequal (C1, sparse (C2)))
    end
    C1 = A .* spdiags (ones (d,1), 0, m, n) ;
    C2 = gb.select ('diag', A, sparse (0)) ;
    assert (isequal (C1, sparse (C2)))

%-------------------------------------------------------------------------------
% offdiag
%-------------------------------------------------------------------------------

    d = min (m,n) ;
    C1 = A .* (1 - spdiags (ones (d,1), 0, m, n)) ;
    C2 = gb.select ('offdiag', A) ;
    assert (isequal (C1, sparse (C2)))
    for k = -m:n
        C1 = A .* (1 - spdiags (ones (d,1), k, m, n)) ;
        C2 = gb.select ('offdiag', A, sparse (k)) ;
        assert (isequal (C1, sparse (C2)))
    end
    C1 = A .* (1 - spdiags (ones (d,1), 0, m, n)) ;
    C2 = gb.select ('offdiag', A, sparse (0)) ;
    assert (isequal (C1, sparse (C2)))

%-------------------------------------------------------------------------------
% nonzero
%-------------------------------------------------------------------------------

    % all explicit entries in the MATLAB sparse matrix are nonzero,
    % so this does nothing.  A better test would be to compute a GraphBLAS
    % matrix with explicit zeros first.

    %   nonzero     ne0         !=0   ~=0

    M = (A ~= 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('nonzero', A) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('ne0', A) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('!=0', A) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('~=0', A) ;
    assert (isequal (C1, sparse (C2)))

%-------------------------------------------------------------------------------
% eqzero
%-------------------------------------------------------------------------------

    % all explicit entries in the MATLAB sparse matrix are nonzero,
    % so this does nothing.

    %   eqzero      eq0         ==0

    C1 = sparse (m,n) ;

    C2 = gb.select ('eqzero', A) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('eq0', A) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('==0', A) ;
    assert (isequal (C1, sparse (C2)))

%-------------------------------------------------------------------------------
% gtzero
%-------------------------------------------------------------------------------

    %   gtzero      gt0         >0

    M = (A > 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('gtzero', A) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('gt0', A) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('>0', A) ;
    assert (isequal (C1, sparse (C2)))

%-------------------------------------------------------------------------------
% gezero
%-------------------------------------------------------------------------------

    %   gezero      ge0         >=0

    M = (A >= 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('gezero', A) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('ge0', A) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('>=0', A) ;
    assert (isequal (C1, sparse (C2)))

%-------------------------------------------------------------------------------
% ltzero
%-------------------------------------------------------------------------------

    %   ltzero      lt0         <0

    M = (A < 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('ltzero', A) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('lt0', A) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('<0', A) ;
    assert (isequal (C1, sparse (C2)))

%-------------------------------------------------------------------------------
% lezero
%-------------------------------------------------------------------------------

    %   lezero      le0         <=0

    M = (A <= 0) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('lezero', A) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('le0', A) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('<=0', A) ;
    assert (isequal (C1, sparse (C2)))


%-------------------------------------------------------------------------------
% nonthunk
%-------------------------------------------------------------------------------

    %   nonthunk     !=thunk   ~=thunk

    M = (A ~= thunk) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('nethunk', A, thunk) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('!=thunk', A, thunk) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('~=thunk', A, thunk) ;
    assert (isequal (C1, sparse (C2)))

%-------------------------------------------------------------------------------
% eqthunk
%-------------------------------------------------------------------------------

    %   eqthunk      ==thunk

    M = (A == thunk) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('eqthunk', A, thunk) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('eqthunk', A, thunk) ;
    assert (isequal (C1, sparse (C2)))

%-------------------------------------------------------------------------------
% gtthunk
%-------------------------------------------------------------------------------

    %   gtthunk      >thunk

    M = (A > thunk) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('gtthunk', A, thunk) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('>thunk', A, thunk) ;
    assert (isequal (C1, sparse (C2)))

%-------------------------------------------------------------------------------
% gethunk
%-------------------------------------------------------------------------------

    %   gethunk      >=thunk

    M = (A >= thunk) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('gethunk', A, thunk) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('>=thunk', A, thunk) ;
    assert (isequal (C1, sparse (C2)))

%-------------------------------------------------------------------------------
% ltthunk
%-------------------------------------------------------------------------------

    %   ltthunk      <thunk

    M = (A < thunk) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('ltthunk', A, thunk) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('<thunk', A, thunk) ;
    assert (isequal (C1, sparse (C2)))

%-------------------------------------------------------------------------------
% lethunk
%-------------------------------------------------------------------------------

    %   lethunk      <=thunk

    M = (A <= thunk) ;
    C1 = sparse (m,n) ;
    C1 (M) = A (M) ;

    C2 = gb.select ('lethunk', A, thunk) ;
    assert (isequal (C1, sparse (C2)))

    C2 = gb.select ('<=thunk', A, thunk) ;
    assert (isequal (C1, sparse (C2)))

fprintf ('test8: all tests passed\n') ;
