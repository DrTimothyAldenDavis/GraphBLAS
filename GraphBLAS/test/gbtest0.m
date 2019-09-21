function gbtest0
%GBTEST0 test gb.clear

gb.clear

assert (isequal (gb.format, 'by col')) ;
assert (isequal (gb.chunk, 4096)) ;

fprintf ('default # of threads: %d\n', gb.threads) ;

fprintf ('gbtest0: all tests passed\n') ;

