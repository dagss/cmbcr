# CMB component seperation / constrained realizations / Wiener-filtering

## Setup

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
