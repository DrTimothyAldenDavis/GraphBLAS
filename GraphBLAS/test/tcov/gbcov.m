function gbcov
%GBCOV run all GraphBLAS tests, with statement coverage

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

gbcovmake
addpath ('..') ;            % add the test folder to the path
rmpath ('../..') ;          % remove the regular @gb class
addpath ('tmp') ;           % add the modified @gb class
path
which gb
pause

try
    gbtest ;
catch me
    me
    disp (me.message) ;
end

% addpath ('../..') ;         % add back the regular @gb class
% rmpath ('tmp') ;            % remove the modified @gb class
% which gb

gbcovstat
fprintf ('\ngbcov: all tests passed\n') ;

