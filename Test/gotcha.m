
clear all
make
debug_on


threads {1} = [4 1] ;

t = threads ;

% single thread
t1 {1} = [1 1] ;

s {1} = [4 1] ;
s {2} = [1 1] ;

addpath ('../Test/spok') ;
addpath ('../Demo/MATLAB') ;

try
    spok (sparse (1)) ;
catch
    cd spok ; spok_install ; cd ..
end

logstat ;             % start the log.txt

%-------------------------------------------------------------------------------

%439
logstat ('test20',s) ;  % quick test of GB_mex_mxm on a few semirings

%495
logstat ('test74',t) ;  % test GrB_mxm on all semirings

%512
logstat ('test90',s) ;  % test pre-compiled user-defined semirings

%532
logstat ('test06',s) ;  % test GrB_mxm on all semirings

%----------------------------------------
% too long (more than 600 seconds)
%----------------------------------------

% TODO reduce the length of these tests

%113 no malloc debug, 940 with (TODO: try turning off debug)
logstat ('test21',t) ;  % quick test of GB_mex_subassign
logstat ('test21',t1) ;  % quick test of GB_mex_subassign

%1292
logstat ('test25',s) ;  % quick test of GxB_select

%115 no debug, 2085 with (TODO: try turning off debug)
% only covers 10 statements
logstat ('test21b',t) ; % quick test of GB_mex_assign

%2897
logstat ('test18',s) ;  % quick tests of GrB_eWiseAdd and eWiseMult

%-------------------------------------------------------------------------------
