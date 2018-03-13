CC=    gcc
CXX=   g++
LD=    gcc


PROJ = Ia

PROJ_DIR = ./$(PROJ)


CFLAGS= -O3 -std=c11 -Wall -Wextra -mavx2
INCPATH= -I/usr/local/include -I/opt/local/include -I/usr/include -I$(PROJ_DIR)
LDFLAGS=
LIBPATH= -L/usr/local/lib -L/opt/local/lib -L/usr/lib
LIBS=    -lcrypto


SRCS = $(wildcard $(PROJ_DIR)/*.c)
SRCS_O = $(SRCS:.c=.o)
SRCS_O_ND = $(subst $(PROJ_DIR)/,,$(SRCS_O))

OBJ = $(SRCS_O_ND) utils.o

EXE= PQCgenKAT_sign rainbow-genkey rainbow-sign rainbow-verify












.PHONY: all tests tables clean

all: $(OBJ) $(EXE)


#%-test: $(OBJ) %-test.o
#	$(LD) $(LDFLAGS) $(LIBPATH) -o $@ $^ $(LIBS)

#%-benchmark: $(OBJ) %-benchmark.o
#	$(LD) $(LDFLAGS) $(LIBPATH) -o $@ $^ $(LIBS)

rainbow-genkey: $(OBJ) rainbow-genkey.o
	$(LD) $(LDFLAGS) $(LIBPATH) -o $@ $^ $(LIBS)

rainbow-sign: $(OBJ) rainbow-sign.o
	$(LD) $(LDFLAGS) $(LIBPATH) -o $@ $^ $(LIBS)

rainbow-verify: $(OBJ) rainbow-verify.o
	$(LD) $(LDFLAGS) $(LIBPATH) -o $@ $^ $(LIBS)

PQCgenKAT_sign: $(OBJ) PQCgenKAT_sign.o
	$(LD) $(LDFLAGS) $(LIBPATH) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCPATH) -c $<

%.o: $(PROJ_DIR)/%.c
	$(CC) $(CFLAGS) $(INCPATH) -c $<

%.o: common/%.c
	$(CC) $(CFLAGS) $(INCPATH) -c $<


%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCPATH) -c $<

tests:
	cd tests; make

tables:
	cd supplement; make

clean:
#	rm *.o *-test *-benchmark rainbow-genkey rainbow-sign rainbow-verify PQCgenKAT_sign;
	rm *.o  rainbow-genkey rainbow-sign rainbow-verify  PQCgenKAT_sign

