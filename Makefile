CONFIG?=default
ifndef CONFIG
  CONFIG:=$(error CONFIG undefined. Please see README.md for help)UNDEFINED
endif

include config/config.$(CONFIG)


F90 ?= gfortran
F90_LD ?= gfortran
LD ?= gcc
CC ?= gcc
CYTHON ?= cython


FORTRAN_CYTHON_MODULES = cmbcr/harmonic_preconditioner.so cmbcr/rotate_alm.so
CYTHON_MODULES = cmbcr/mmajor.so cmbcr/mblocks.so cmbcr/sharp.so cmbcr/healpix.so
FORTRAN_MODULES = build/src/types_as_c.mod build/src/constants.mod

F90_LDFLAGS = -fopenmp
LDFLAGS =
LIBS = $(LAPACK) $(SHARP)

#$(shell python2.7-config --ldflags)

PYTHON_INCLUDES = $(shell python2.7-config --includes)
INCLUDES = $(PYTHON_INCLUDES) $(SHARP_INCLUDES)
FC_INCLUDES = -Ibuild/src

PYPKG = cmbcr


all: $(FORTRAN_CYTHON_MODULES) $(CYTHON_MODULES)


build/$(PYPKG)/%.pyx.c: $(PYPKG)/%.pyx build
	$(CYTHON) --fast-fail -o $@ $<

build/$(PYPKG)/%.pyx.o: build/$(PYPKG)/%.pyx.c build
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

$(FORTRAN_CYTHON_MODULES): %.so: build/%.pyx.o build/%.f90.o build
	$(F90_LD) $(F90_LDFLAGS) -shared -o $@ $< $(word 2,$^) $(LIBS)

$(CYTHON_MODULES): %.so: build/%.pyx.o build
	$(LD) $(LDFLAGS) -shared -o $@ $< $(LIBS)

build/src/%.mod: src/%.f90 build
	$(F90) $(FCFLAGS) $(FC_INCLUDES) -o $@ -c $<

build/$(PYPKG)/%.f90.o: $(PYPKG)/%.f90 $(FORTRAN_MODULES) build
	$(F90) $(FCFLAGS) $(FC_INCLUDES) -o $@ -c $<


testimport: $(FORTRAN_CYTHON_MODULES)
	PYTHONPATH=. python -c 'import cmbcr.ha_preconditioner'

build:
	mkdir build || true
	mkdir build/cmbcr || true
	mkdir build/src || true

clean:
	rm -rf build
	rm -f $(PYPKG)/*.so

.PRECIOUS: build/$(PYPKG)/%.pyx.c build/src/*.mod

