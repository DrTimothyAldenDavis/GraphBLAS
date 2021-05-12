#!/bin/bash

g++ -c rmm_wrap.cpp -o rmm_wrap.o -I/home/jeaton/lib/include -I/usr/local/cuda/include -I.

ar rvs rmm_wrap.a rmm_wrap.o

g++  rmm_wrap_test.c -o rmm_wrap_test  rmm_wrap.a  -I/home/jeaton/lib/include -I/usr/local/cuda/include -I. -L/usr/local/cuda/lib64 -L. -lcuda -lcudart

