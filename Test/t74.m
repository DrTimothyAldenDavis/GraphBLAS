
clear all
gbclear
make
threads {1} = [4 1] ;
t = threads ;
logstat ('test74',t) ;  % test GrB_mxm on all semirings

