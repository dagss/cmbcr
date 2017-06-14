LAPACK = /home/dagss/astro/OpenBLAS/libopenblas_haswell-r0.2.19.a
SHARP = -L/home/dagss/astro/libsharp/auto/lib -lsharp -lfftpack -lc_utils

FORTRAN_CYTHON_MODULES = cmbcr/harmonic_preconditioner.so
CYTHON_MODULES = cmbcr/mmajor.so
FORTRAN_MODULES = build/src/types_as_c.mod build/src/constants.mod

CFLAGS = -fPIC -O2 -march=native
FCFLAGS = -fPIC -fopenmp -march=native -g -O3 -Wall -fcheck=bounds -fcheck=do -fcheck=mem -fcheck=recursion
F90_LDFLAGS = -fopenmp
LDFLAGS =
LIBS = $(LAPACK) $(SHARP)

#$(shell python2.7-config --ldflags)

F90 = gfortran
F90_LD = gfortran
LD = gcc
CC = gcc
CYTHON = cython

PYTHON_INCLUDES = $(shell python2.7-config --includes)
INCLUDES = $(PYTHON_INCLUDES)
FC_INCLUDES = -Ibuild/src

PYPKG = cmbcr



all: $(FORTRAN_PYTHON_MODULES) build



build/$(PYPKG)/%.pyx.c: $(PYPKG)/%.pyx build
	$(CYTHON) --fast-fail -o $@ $<

build/$(PYPKG)/%.pyx.o: build/$(PYPKG)/%.pyx.c build
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

$(FORTRAN_CYTHON_MODULES): %.so: build/%.pyx.o build/%.f90.o build
	$(F90_LD) $(F90_LDFLAGS) -shared -o $@ $< $(word 2,$^) $(LIBS)

$(CYTHON_MODULES): %.so: build/%.pyx.o build
	$(LD) $(LDFLAGS) -shared -o $@ $<

build/src/%.mod: src/%.f90 build
	$(F90) $(FCFLAGS) $(FC_INCLUDES) -o $@ -c $<

build/$(PYPKG)/%.f90.o: $(PYPKG)/%.f90 $(FORTRAN_MODULES) build
	$(F90) $(FCFLAGS) $(FC_INCLUDES) -o $@ -c $<


testimport: $(FORTRAN_CYTHON_MODULES)
	PYTHONPATH=. python -c 'import cmbcr.harmonic_preconditioner'

build:
	mkdir build || true
	mkdir build/cmbcr || true
	mkdir build/src || true

clean:
	rm -rf build
	rm -f $(PYPKG)/*.so

