function codegen_red
%CODEGEN_RED create functions for all reduction operators
%
% This function creates all files of the form GB_reduce__*.[ch],
% and the include file GB_reduce__include.h.

fprintf ('\nreduction operators:\n') ;

f = fopen ('Generated/GB_reduce__include.h', 'w') ;
fprintf (f, '//------------------------------------------------------------------------------\n') ;
fprintf (f, '// GB_reduce__include.h: definitions for GB_reduce__*.c\n') ;
fprintf (f, '//------------------------------------------------------------------------------\n') ;
fprintf (f, '\n') ;
fprintf (f, '// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.\n') ;
fprintf (f, '// http://suitesparse.com   See GraphBLAS/Doc/License.txargt for license.\n') ;
fprintf (f, '\n') ;
fprintf (f, '// This file has been automatically generated from Generator/GB_reduce.h') ;
fprintf (f, '\n\n') ;
fclose (f) ;

% MIN: 10 monoids:  name     function                       type        identity      terminal
codegen_red_method ('min',   'zarg = GB_IMIN (zarg, yarg)', 'int8_t'  , 'INT8_MAX'  , 'INT8_MIN'  ) ;
codegen_red_method ('min',   'zarg = GB_IMIN (zarg, yarg)', 'int16_t' , 'INT16_MAX' , 'INT16_MIN' ) ;
codegen_red_method ('min',   'zarg = GB_IMIN (zarg, yarg)', 'int32_t' , 'INT32_MAX' , 'INT32_MIN' ) ;
codegen_red_method ('min',   'zarg = GB_IMIN (zarg, yarg)', 'int64_t' , 'INT64_MAX' , 'INT64_MIN' ) ;
codegen_red_method ('min',   'zarg = GB_IMIN (zarg, yarg)', 'uint8_t' , 'UINT8_MAX' , '0'         ) ;
codegen_red_method ('min',   'zarg = GB_IMIN (zarg, yarg)', 'uint16_t', 'UINT16_MAX', '0'         ) ;
codegen_red_method ('min',   'zarg = GB_IMIN (zarg, yarg)', 'uint32_t', 'UINT32_MAX', '0'         ) ;
codegen_red_method ('min',   'zarg = GB_IMIN (zarg, yarg)', 'uint64_t', 'UINT64_MAX', '0'         ) ;
codegen_red_method ('min',   'zarg = fminf (zarg, yarg)'  , 'float'   , 'INFINITY'  , '-INFINITY' ) ;
codegen_red_method ('min',   'zarg = fmin (zarg, yarg)'   , 'double'  , 'INFINITY'  , '-INFINITY' ) ;

% MAX: 10 monoids
codegen_red_method ('max',   'zarg = GB_IMAX (zarg, yarg)', 'int8_t'  , 'INT8_MIN'  , 'INT8_MAX'  ) ;
codegen_red_method ('max',   'zarg = GB_IMAX (zarg, yarg)', 'int16_t' , 'INT16_MIN' , 'INT16_MAX' ) ;
codegen_red_method ('max',   'zarg = GB_IMAX (zarg, yarg)', 'int32_t' , 'INT32_MIN' , 'INT32_MAX' ) ;
codegen_red_method ('max',   'zarg = GB_IMAX (zarg, yarg)', 'int64_t' , 'INT64_MIN' , 'INT64_MAX' ) ;
codegen_red_method ('max',   'zarg = GB_IMAX (zarg, yarg)', 'uint8_t' , '0'         , 'UINT8_MAX' ) ;
codegen_red_method ('max',   'zarg = GB_IMAX (zarg, yarg)', 'uint16_t', '0'         , 'UINT16_MAX') ;
codegen_red_method ('max',   'zarg = GB_IMAX (zarg, yarg)', 'uint32_t', '0'         , 'UINT32_MAX') ;
codegen_red_method ('max',   'zarg = GB_IMAX (zarg, yarg)', 'uint64_t', '0'         , 'UINT64_MAX') ;
codegen_red_method ('max',   'zarg = fmaxf (zarg, yarg)'  , 'float'   , '-INFINITY' , 'INFINITY'  ) ;
codegen_red_method ('max',   'zarg = fmax (zarg, yarg)'   , 'double'  , '-INFINITY' , 'INFINITY'  ) ;

% PLUS: 10 monoids
codegen_red_method ('plus',  'zarg += yarg'               , 'int8_t'  , '0'         , [ ]         ) ;
codegen_red_method ('plus',  'zarg += yarg'               , 'int16_t' , '0'         , [ ]         ) ;
codegen_red_method ('plus',  'zarg += yarg'               , 'int32_t' , '0'         , [ ]         ) ;
codegen_red_method ('plus',  'zarg += yarg'               , 'int64_t' , '0'         , [ ]         ) ;
codegen_red_method ('plus',  'zarg += yarg'               , 'uint8_t' , '0'         , [ ]         ) ;
codegen_red_method ('plus',  'zarg += yarg'               , 'uint16_t', '0'         , [ ]         ) ;
codegen_red_method ('plus',  'zarg += yarg'               , 'uint32_t', '0'         , [ ]         ) ;
codegen_red_method ('plus',  'zarg += yarg'               , 'uint64_t', '0'         , [ ]         ) ;
codegen_red_method ('plus',  'zarg += yarg'               , 'float'   , '0'         , [ ]         ) ;
codegen_red_method ('plus',  'zarg += yarg'               , 'double'  , '0'         , [ ]         ) ;

% TIMES: 10 monoids
codegen_red_method ('times', 'zarg *= yarg'               , 'int8_t'  , '1'         , '0'         ) ;
codegen_red_method ('times', 'zarg *= yarg'               , 'int16_t' , '1'         , '0'         ) ;
codegen_red_method ('times', 'zarg *= yarg'               , 'int32_t' , '1'         , '0'         ) ;
codegen_red_method ('times', 'zarg *= yarg'               , 'int64_t' , '1'         , '0'         ) ;
codegen_red_method ('times', 'zarg *= yarg'               , 'uint8_t' , '1'         , '0'         ) ;
codegen_red_method ('times', 'zarg *= yarg'               , 'uint16_t', '1'         , '0'         ) ;
codegen_red_method ('times', 'zarg *= yarg'               , 'uint32_t', '1'         , '0'         ) ;
codegen_red_method ('times', 'zarg *= yarg'               , 'uint64_t', '1'         , '0'         ) ;
codegen_red_method ('times', 'zarg *= yarg'               , 'float'   , '1'         , [ ]         ) ;
codegen_red_method ('times', 'zarg *= yarg'               , 'double'  , '1'         , [ ]         ) ;

% 4 boolean monoids
codegen_red_method ('lor' ,  'zarg = (zarg || yarg)'      , 'bool'    , 'false'     , 'true'      ) ;
codegen_red_method ('land',  'zarg = (zarg && yarg)'      , 'bool'    , 'true'      , 'true'      ) ;
codegen_red_method ('lxor',  'zarg = (zarg != yarg)'      , 'bool'    , 'false'     , [ ]         ) ;
codegen_red_method ('eq'  ,  'zarg = (zarg == yarg)'      , 'bool'    , 'true'      , [ ]         ) ;

fprintf ('\n') ;

