#-------------------------------------------------------------------------------
# GraphBLAS/GraphBLAS/private/Makefile
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2018, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

# This is just here for 'make clean', to remove compiled codes.

default: mexfuncs

clean:
	- $(RM) -rf *.o *.mex*

purge: clean

distclean: clean

#-------------------------------------------------------------------------------
# MATLAB R2021a and following:
#-------------------------------------------------------------------------------

# https://www.mathworks.com/matlabcentral/answers/377799-compiling-mex-files-without-the-mex-command

MWROOT := /usr/local/MATLAB/R2021a
CC := gcc

DEFINES := -D_GNU_SOURCE
MATLABMEX := -DMATLAB_MEX_FILE -DMATLAB_DEFAULT_RELEASE=R2018a

CFLAGS := -fexceptions -fPIC -fno-omit-frame-pointer -pthread -std=c11 \
    -Wno-pragmas -fexcess-precision=fast -fcx-limited-range -fno-math-errno \
    -fwrapv -fopenmp
COPTIMFLAGS := -O3 -DNDEBUG
CDEBUGFLAGS := -g

INCLUDE := -I"$(MWROOT)/extern/include" -I"$(MWROOT)/simulink/include"
INCLUDE += -Iutil -I../../../Include -I../../../Source
INCLUDE += -I../../../Source/Template

LD := gcc
LDFLAGS := -pthread -Wl,--no-undefined -fopenmp
LDTYPE := -shared
LINKEXPORTVER := -Wl,--version-script,"$(MWROOT)/extern/lib/glnxa64/c_exportsmexfileversion.map"

LINKLIBS := -Wl,-rpath-link,/usr/local/lib -L/usr/local/lib
LINKLIBS += -Wl,-rpath-link,../../../build -L../../../build
LINKLIBS += -Wl,--as-needed -lgraphblas
LINKLIBS += -Wl,-rpath-link,$(MWROOT)/bin/glnxa64 -L"$(MWROOT)/bin/glnxa64"
LINKLIBS += -Wl,-rpath-link,$(MWROOT)/extern/bin/glnxa64
LINKLIBS += -L"$(MWROOT)/extern/bin/glnxa64"
LINKLIBS += -lmx -lmex -lmat -lm -lstdc++

CF := $(MATLABMEX) $(DEFINES) $(CFLAGS) $(COPTIMFLAGS) $(INCLUDE)
LF := $(LD) $(LDFLAGS) $(LDTYPE) $(LINKEXPORTVER) 
I := util/gb_matlab.h Makefile

OBJ := gb_abort.o gb_assign.o gb_binop_to_monoid.o gb_by_col.o \
    gb_default_format.o gb_default_type.o gb_expand_to_full.o gb_export.o \
    gb_export_to_mxfull.o gb_export_to_mxsparse.o gb_export_to_mxstruct.o \
    gb_find_dot.o gb_first_binop.o gb_flush.o gb_get_deep.o gb_get_format.o \
    gb_get_mxargs.o gb_get_shallow.o gb_get_sparsity.o gb_is_all.o \
    gb_is_equal.o gb_is_float.o gb_is_integer.o gb_is_vector.o \
    gb_matrix_assign_scalar.o gb_mxarray_is_empty.o gb_mxarray_is_scalar.o \
    gb_mxarray_to_descriptor.o gb_mxarray_to_list.o gb_mxarray_type.o \
    gb_mxcell_to_index.o gb_mxclass_to_mxstring.o gb_mxfree.o \
    gb_mxstring_to_binop.o gb_mxstring_to_format.o gb_mxstring_to_monoid.o \
    gb_mxstring_to_selectop.o gb_mxstring_to_semiring.o \
    gb_mxstring_to_string.o gb_mxstring_to_type.o gb_mxstring_to_unop.o \
    gb_new.o gb_norm.o gb_norm_kind.o gb_round_binop.o gb_semiring.o \
    gb_string_and_type_to_binop.o gb_string_and_type_to_unop.o \
    gb_string_to_binop.o gb_string_to_monoid.o gb_string_to_selectop.o \
    gb_string_to_semiring.o gb_string_to_type.o gb_string_to_unop.o \
    gb_typecast.o gb_type_to_mxstring.o gb_usage.o \
    c_mexapi_version.o

MEXFUNC := gbapply2.mexa64 gbapply.mexa64 gbassign.mexa64 gbbinopinfo.mexa64 \
    gbbuild.mexa64 gbburble.mexa64 gbcat.mexa64 gbchunk.mexa64 \
    gbdegree.mexa64 gbdescriptorinfo.mexa64 gbdisp.mexa64 gbeadd.mexa64 \
    gbemult.mexa64 gbextract.mexa64 gbextracttuples.mexa64 \
    gbextractvalues.mexa64 gbformat.mexa64 gbfull.mexa64 gbisequal.mexa64 \
    gbkronecker.mexa64 gblogassign.mexa64 gblogextract.mexa64 gbmatlab.mexa64 \
    gbmdiag.mexa64 gbmonoidinfo.mexa64 gbmxm.mexa64 gbnew.mexa64 \
    gbnorm.mexa64 gbnormdiff.mexa64 gbnvals.mexa64 gboptype.mexa64 \
    gbreduce.mexa64 gbselect.mexa64 gbselectopinfo.mexa64 \
    gbsemiringinfo.mexa64 gbsetup.mexa64 gbsize.mexa64 gbsplit.mexa64 \
    gbsubassign.mexa64 gbthreads.mexa64 gbtrans.mexa64 gbtype.mexa64 \
    gbunopinfo.mexa64 gbvdiag.mexa64 gbver.mexa64 gbversion.mexa64 \
    gbvreduce.mexa64

%.o : util/%.c $(I)
	$(CC) -c $< -o $@ $(CF)

%.mexa64: mexfunctions/%.c $(OBJ) $(I)
	$(CC) -c $< -o $*.o $(CF)
	$(LF) $*.o $(OBJ) $(LINKLIBS) -o $@

mexfuncs: $(MEXFUNC)

c_mexapi_version.o : $(MWROOT)/extern/version/c_mexapi_version.c Makefile
	$(CC) -c $< -o $@ $(CF)

config:
	@echo 'CFLAGS   = ' '$(CFLAGS)'
	@echo 'LDFLAGS  = ' '$(LDFLAGS)'
	@echo 'LINKLIBS = ' '$(LINKLIBS)'

.PRECIOUS: $(OBJ)
