# CMB component seperation / constrained realizations / Wiener-filtering

This is an experimental testbed for developing algorithms for CMB component
seperation. We do not expect this code to be runnable / useful to anyone
besides the developers. The Commander code (https://github.com/hke/commander)
contains another implementation of some of these algorithms, for use in a
production analysis.

That said the code is made available so that it is possible to verify
the claims made in the paper arxiv:TODO. It is also most useful for reading,
not so much for execution.

After building the code as described below, one may run

PYTHONPATH=. python scripts/precond_benchmark.py input/mask.yaml 64

...to run the solver on the model using a mask. The input specification file
`input/mask.yaml` must be copied and modified so as to point to your own
datafiles (not included here). The second argument is the Nside at which to
run the preconditioners.


## Setup

cmbcr depends on Libsharp and OpenBLAS. See the config subdirectory and create
a configuration that matches your system. Then:

```
make CONFIG=yourconfig
```

will build using the configuration in `config/config.yourconfig`.
To avoid having to specify this every time you may also create
the `config/config.default` file, e.g.,

```
(cd config; ln -s config.yourconfig config.default)
make
```

## Copyright & license

The cmbcr code is Copyright 2017 Dag Sverre Seljebotn

Since this code depends on code from Libsharp and HEALPix, which are
both under the GPLv2, this code is also licensed under GPLv2 (or any
later version). Details (see also LICENSE.txt):

cmbcr is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

libsharp is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
