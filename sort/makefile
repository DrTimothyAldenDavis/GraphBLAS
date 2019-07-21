#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#

include make.def

EXES= testSort$(EXE) 

SRT_OBJS = testSort.$(OBJ) sort.$(OBJ) parsort1.$(OBJ) parsort2.$(OBJ)

all: $(EXES)

testSort$(EXE): $(SRT_OBJS)
	$(CLINKER) $(CFLAGS) -o testSort$(EXE) $(SRT_OBJS) $(LIBS)

test: $(EXES)
	for i in $(EXES); do \
            $(PRE)$$i; \
        done

clean:
	$(RM) $(EXES) *.$(OBJ) 

.SUFFIXES:
.SUFFIXES: .c .cpp .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	$(CC) $(CFLAGS) -c $<
