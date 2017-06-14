import healpy
import numpy as np
import pyfits

def load_map(map_type, filename, field=0, units_in_file=None, extno=1,
             dtype=np.double):
    """
    Loads a HEALPix spherical map from FITS file.

    The result will be in ring-ordering regardless of what ordering is
    in the file (i.e., conversion from nested is done automatically
    based on metadata).

    The intent for temperature maps is that the output of this routine
    should always be in uK (or uK^2); the `units_in_file` and
    `map_type` are present to assist in unit conversion tasks. The
    options for `map_type` are:

    `raw`:
        Do not process data
    `temp`, `rms`:
        Loading a temperature map, output should be in uK.
    `var`:
        Loading a variance map, output should be in uK^2
    `mask`:
        Loading a mask, all values should be in range [0, 1].

    The unit conversion is based on reading the temperature metadata
    in the file. However, `units_in_file` overrides whatever metadata
    is present. If temperature metadata is not present, `units_in_file`
    must be provided or an exception is raised.

    Parameters
    ----------

    map_type : str
        One of 'temp', 'rms', 'var', 'raw', 'mask'.

    filename : str

    field : int or str (optional)

        Field in the FITS record, this can either be an integer (0-based)
        or a string with the field name.

    extno : int (optional)

        FITS extension number (usually 1 for HEALPix maps)

    units_in_file : str or None (optional)

        The units the contents of the file is in. Temperature metadata
        will be ignored. This should only be used when `map_type` is
        `temperature`, `rms` or `variance`.

    dtype : dtype
        dtype to load the data in

    Returns
    -------

    A ring-ordered map; in uK for temperature maps or uK^2 for
    variance maps.
    """
    hdulist = pyfits.open(filename)
    try:
        ext = hdulist[extno]

        # Probe for presence of field and resolve it to an integer, in order to be
        # able to look up TUNIT{fieldno} below.
        nfields = ext.header['TFIELDS']
        if isinstance(field, int):
            if field > nfields:
                raise ValueError('Cannot find FITS field: %s' % repr(desc))
            if field < 0:
                raise ValueError('Field number must be positive: %s' % repr(desc))
        else:
            found = False
            for i in range(1, nfields+1):
                if ext.header['TTYPE%d' % i].strip() == field:
                    field = i - 1
                    found = True
                    break
            if not found:
                raise ValueError('Cannot find FITS field: %s' % repr(desc))
        
        if map_type in ('temp', 'rms', 'var'):
            # Figure out units
            u = units_in_file
            if u is None:
                try:
                    metadata_units = ext.header['TUNIT%d' % (field + 1)].split(',')[0]
                except KeyError:
                    metadata_units = None
                else:
                    if metadata_units.lower() in ('n/a', 'na', ''):
                        metadata_units = None
                if metadata_units is None:
                    raise ValueError('units not given in code and not found in file')
                u = metadata_units

            # Misc unit-juggling
            u = u.strip()
            u = u.replace('K_CMB', 'K')
                
            if map_type == 'var':
                if not u.endswith('^2'):
                    raise ValueError('map_type is "var", but units did not end with ^2: %s' %
                                     u)
                u = u[:-2]
                if u.startswith('(') and u.endswith(')'):
                    u = u[1:-1]
                conversion_factor = units.get_temperature_conversion_factor(u)
                conversion_factor = conversion_factor**2
            else:
                conversion_factor = units.get_temperature_conversion_factor(u)

        elif map_type in ('mask', 'raw'):
            conversion_factor = 1
        else:
            raise ValueError('Unexpected `interpretation`: "%s"' % data_type)

        try:
            is_nested = (ext.header['ORDERING'].lower() == 'nested')
        except:
            # Could implement "ordering_in_file" argument
            raise NotImplementedError('No ORDERING metadata given in file')

        # Load the data
        mapdata = ext.data.field(field).ravel().astype(dtype)
    finally:
        hdulist.close()

    if conversion_factor != 1:
        mapdata[...] *= conversion_factor
    if map_type == 'mask':
        if not np.all((mapdata >= 0) & (mapdata <= 1)):
            raise ValueError('Attempted to load mask, but got values outside of [0, 1]')
    if is_nested:
        mapdata = healpy.pixelfunc.reorder(mapdata, inp='NESTED', out='RING')

    # badpix -> nan
    mapdata[mapdata == -1.63750e30] = np.nan
    
    return mapdata
