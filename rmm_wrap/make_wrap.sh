#!/bin/bash

g++ -std=c++17 -c rmm_wrap.cpp -o rmm_wrap.o -I$CONDA_PREFIX/include -I/usr/local/cuda/include -I/share/workspace/GraphBLAS/rmm/include -I.
ar -rcsv librmm_wrap.a rmm_wrap.o 
ranlib  librmm_wrap.a

gcc -std=c11 rmm_wrap_test.c -o rmm_wrap_test  rmm_wrap.a rmm_wrap.h  -I$CONDA_PREFIX/include -I/usr/local/cuda/include -I. -L/usr/local/cuda/lib64 -I/share/workspace/GraphBLAS/rmm/include -L.  -lcuda -lcudart -lstdc++

