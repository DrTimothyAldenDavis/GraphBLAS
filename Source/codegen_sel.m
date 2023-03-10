function codegen_sel
%CODEGEN_SEL create functions for all selection operators
%
% This function creates all files of the form GB_sel__*.c,
% and the include file GB_sel__include.h.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

fprintf ('\nselection operators:\n') ;
addpath ('../Test') ;

fh = fopen ('Generated1/GB_sel__include.h', 'w') ;
fprintf (fh, '//------------------------------------------------------------------------------\n') ;
fprintf (fh, '// GB_sel__include.h: definitions for GB_sel__*.c\n') ;
fprintf (fh, '//------------------------------------------------------------------------------\n') ;
fprintf (fh, '\n') ;
fprintf (fh, '// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.\n') ;
fprintf (fh, '// SPDX-License-Identifier: Apache-2.0\n\n') ;
fprintf (fh, '// This file has been automatically generated from Generator/GB_sel.h') ;
fprintf (fh, '\n\n') ;
fclose (fh) ;

% NONZOMBIE:         name         selector                     type
fprintf ('\nnonzombie  ') ;
test_value = 'bool keep = (Ai [p] >= 0)' ;
codegen_sel_method ('nonzombie', test_value, 'bool'      ) ;
codegen_sel_method ('nonzombie', test_value, 'int8_t'    ) ;
codegen_sel_method ('nonzombie', test_value, 'int16_t'   ) ;
codegen_sel_method ('nonzombie', test_value, 'int32_t'   ) ;
codegen_sel_method ('nonzombie', test_value, 'int64_t'   ) ;
codegen_sel_method ('nonzombie', test_value, 'uint8_t'   ) ;
codegen_sel_method ('nonzombie', test_value, 'uint16_t'  ) ;
codegen_sel_method ('nonzombie', test_value, 'uint32_t'  ) ;
codegen_sel_method ('nonzombie', test_value, 'uint64_t'  ) ;
codegen_sel_method ('nonzombie', test_value, 'float'     ) ;
codegen_sel_method ('nonzombie', test_value, 'double'    ) ;
codegen_sel_method ('nonzombie', test_value, 'GxB_FC32_t') ;
codegen_sel_method ('nonzombie', test_value, 'GxB_FC64_t') ;

% NE_THUNK           name         selector            type
fprintf ('\nne_thunk   ') ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != thunk)', 'int8_t'  ) ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != thunk)', 'int16_t' ) ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != thunk)', 'int32_t' ) ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != thunk)', 'int64_t' ) ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != thunk)', 'uint8_t' ) ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != thunk)', 'uint16_t') ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != thunk)', 'uint32_t') ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != thunk)', 'uint64_t') ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != thunk)', 'float'   ) ;
codegen_sel_method ('ne_thunk'  , 'bool keep = (Ax [p] != thunk)', 'double'  ) ;
codegen_sel_method ('ne_thunk'  , 'bool keep = GB_FC32_ne (Ax [p], thunk)', 'GxB_FC32_t') ;
codegen_sel_method ('ne_thunk'  , 'bool keep = GB_FC64_ne (Ax [p], thunk)', 'GxB_FC64_t') ;

% EQ_THUNK           name         selector            type
fprintf ('\neq_thunk   ') ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == thunk)', 'int8_t'  ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == thunk)', 'int16_t' ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == thunk)', 'int32_t' ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == thunk)', 'int64_t' ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == thunk)', 'uint8_t' ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == thunk)', 'uint16_t') ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == thunk)', 'uint32_t') ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == thunk)', 'uint64_t') ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == thunk)', 'float'   ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = (Ax [p] == thunk)', 'double'  ) ;
codegen_sel_method ('eq_thunk'  , 'bool keep = GB_FC32_eq (Ax [p], thunk)', 'GxB_FC32_t') ;
codegen_sel_method ('eq_thunk'  , 'bool keep = GB_FC64_eq (Ax [p], thunk)', 'GxB_FC64_t') ;

% GT_THUNK           name         selector            type
fprintf ('\ngt_thunk   ') ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > thunk)', 'int8_t'  ) ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > thunk)', 'int16_t' ) ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > thunk)', 'int32_t' ) ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > thunk)', 'int64_t' ) ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > thunk)', 'uint8_t' ) ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > thunk)', 'uint16_t') ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > thunk)', 'uint32_t') ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > thunk)', 'uint64_t') ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > thunk)', 'float'   ) ;
codegen_sel_method ('gt_thunk'  , 'bool keep = (Ax [p] > thunk)', 'double'  ) ;

% GE_THUNK           name         selector            type
fprintf ('\nge_thunk   ') ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= thunk)', 'int8_t'  ) ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= thunk)', 'int16_t' ) ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= thunk)', 'int32_t' ) ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= thunk)', 'int64_t' ) ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= thunk)', 'uint8_t' ) ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= thunk)', 'uint16_t') ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= thunk)', 'uint32_t') ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= thunk)', 'uint64_t') ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= thunk)', 'float'   ) ;
codegen_sel_method ('ge_thunk'  , 'bool keep = (Ax [p] >= thunk)', 'double'  ) ;

% LT_THUNK           name         selector            type
fprintf ('\nlt_thunk   ') ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < thunk)', 'int8_t'  ) ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < thunk)', 'int16_t' ) ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < thunk)', 'int32_t' ) ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < thunk)', 'int64_t' ) ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < thunk)', 'uint8_t' ) ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < thunk)', 'uint16_t') ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < thunk)', 'uint32_t') ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < thunk)', 'uint64_t') ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < thunk)', 'float'   ) ;
codegen_sel_method ('lt_thunk'  , 'bool keep = (Ax [p] < thunk)', 'double'  ) ;

% LE_THUNK           name         selector            type
fprintf ('\nle_thunk   ') ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= thunk)', 'int8_t'  ) ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= thunk)', 'int16_t' ) ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= thunk)', 'int32_t' ) ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= thunk)', 'int64_t' ) ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= thunk)', 'uint8_t' ) ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= thunk)', 'uint16_t') ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= thunk)', 'uint32_t') ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= thunk)', 'uint64_t') ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= thunk)', 'float'   ) ;
codegen_sel_method ('le_thunk'  , 'bool keep = (Ax [p] <= thunk)', 'double'  ) ;

fprintf ('\n') ;

