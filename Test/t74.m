
clear all
gbclear
make
threads {1} = [4 1] ;
t = threads ;
logstat ('test20',t) ;  % quick test of GB_mex_mxm on a few semirings
logstat ('test74',t) ;  % test GrB_mxm on all semirings

