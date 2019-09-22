function gbtest51
%GBTEST51 test gb.tricount

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

% the files in ../../Demo/Matrix with no .mtx extension are zero-based
files =  {
'../../Demo/Matrix/2blocks'
'../../Demo/Matrix/ash219'
'../../Demo/Matrix/bcsstk01'
'../../Demo/Matrix/bcsstk16'
'../../Demo/Matrix/eye3'
'../../Demo/Matrix/fs_183_1'
'../../Demo/Matrix/ibm32a'
'../../Demo/Matrix/ibm32b'
'../../Demo/Matrix/lp_afiro'
'../../Demo/Matrix/mbeacxc'
'../../Demo/Matrix/t1'
'../../Demo/Matrix/t2'
'../../Demo/Matrix/west0067' } ;
nfiles = length (files) ;

valid_count = [
           0
           0
         160
     1512964
           0
         863
           0
           0
           0
           0
           2
           0
         120 ] ;

[filepath, name, ext] = fileparts (mfilename ('fullpath')) ;

for k = 1:nfiles
    filename = files {k} ;
    T = load (fullfile (filepath, files {k})) ;
    G = gb.build (int64 (T (:,1)), int64 (T (:,2)), T (:,3)) ;
    [m, n] = size (G) ;
    if (m ~= n)
        G = [gb(m,m) G ; G' gb(n,n)] ;
    elseif (~issymmetric (G))
        G = G + G' ;
    end
    c = gb.tricount (G) ;
    fprintf ('triangle count: %-30s : # triangles %d\n', filename, c) ;
    assert (c == valid_count (k)) ;

    G = gb (G, 'by row') ;
    c = gb.tricount (G) ;
    assert (c == valid_count (k)) ;
end

c = gb.tricount (G, 'check') ;
assert (c == valid_count (end)) ;

fprintf ('\ngbtest51: all tests passed\n') ;

