"""
Load resources from the Healpix library (pixel window and quadrature
ring weights).

The resources are cached in memory.
"""
import pyfits
import os
import numpy as np

class HealpixData(object):
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = get_healpix_data_path()
        self.data_path = data_path
        self.weight_ring_cache = {}
        self.pixel_window_cache = {}
        self.w_T_cache = {}

    def get_raw_ring_weights(self, nside):
        """
        Returns the ring weights to use. NOTE that 1 is automatically
        added; these weights can be passed directly to HEALPix.

        Returns array of shape (3, 2 * nside); the first axis specifies
        T, Q, U.
        """
        result = self.weight_ring_cache.get(nside)
        if result is None:
            hdulist = pyfits.open(os.path.join(self.data_path,
                                               'weight_ring_n%05d.fits' % nside))
            try:
                # Get data to plain 2D float array
                data = hdulist[1].data.view(np.ndarray)
                temp = data['TEMPERATURE WEIGHTS'].ravel()
                qpol = data['Q-POLARISATION WEIGHTS'].ravel()
                upol = data['U-POLARISATION WEIGHTS'].ravel()
                # Convert to native endian...
                data = np.asarray([temp, qpol, upol], np.double, order='C')
                # Add 1
                data += 1
                result = data
                self.weight_ring_cache[nside] = result
            finally:
                hdulist.close()
        return result

    def get_ring_weights_T(self, nside):
        """HEALPix temperature weights properly normalized and copied to contiguous buffer
        """
        w = self.w_T_cache.get(nside, None)
        if w is None:
            w = self.get_raw_ring_weights(nside)[0, :].copy()
            w *= 4 * np.pi / (12 * nside * nside)
            self.w_T_cache[nside] = w
        return w

    def get_pixel_window(self, nside, dtype=np.double):
        """

        """
        result = self.pixel_window_cache.get(nside)
        if result is None:
            hdulist = pyfits.open(os.path.join(self.data_path,
                                               'pixel_window_n%04d.fits' % nside))
            try:
                # Important to use astype, in order to convert to native endia
                data = hdulist[1].data
                result = (data.field('TEMPERATURE').astype(dtype),
                          data.field('POLARIZATION').astype(dtype))
                self.pixel_window_cache[nside] = result
            finally:
                hdulist.close()
        return result
        

_HEALPIX_DATA_PATH = None

def set_healpix_data_path(path):
    global _HEALPIX_DATA_PATH
    _HEALPIX_DATA_PATH = path

def get_healpix_data_path():
    global _HEALPIX_DATA_PATH
    if _HEALPIX_DATA_PATH is None:
        import os
        try:
            _HEALPIX_DATA_PATH = os.path.join(os.environ['HEALPIX'], 'data')
        except KeyError:
            raise Exception("Environment variable HEALPIX not set and did not "
                            "call commander.sphere.set_healpix_data_path")
    return _HEALPIX_DATA_PATH

_instance = None
def get_healpix_data():
    global _instance
    if _instance is None:
        _instance = HealpixData()
    return _instance

def get_ring_weights_T(nside):
    return get_healpix_data().get_ring_weights_T(nside)

def get_healpix_temperature_ring_weights(nside):
    return get_ring_weights_T(nside)
    
def get_healpix_pixel_window(nside):
    t, p = get_healpix_data().get_pixel_window(nside)
    return t.copy(), p.copy()

def nside_of(_map):
    return int(np.sqrt(_map.size/12))
