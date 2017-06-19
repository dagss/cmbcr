from __future__ import division
import sympy
from fractions import Fraction
import numpy as np
from libsharp import legendre_roots
from numpy import pi

class SymPixGrid(object):
    """
    A `band` is the fundamental configuration unit; number of bands
    the sphere is split into *in each hemisphere*.

    Within each band, pixels are organized in strips of increasing theta ('column-major'
    if north is 'up'). We also formally see a band as divided into ``tilesize * tilesize``
    tiles (this doesn't matter for the storage of the band, but we have a guarantee that
    the length of a band, in number of pixels, is divisible by `tilesize`).

    A map array will first have the data of the first band is the northern hemisphere
    as described above, then the first band of the southern, then the second of the northern,
    and so on.

    The the arrays `ring_lengths`, `theta`, `weights`, containing information about
    rings, first have `tilesize` entries for the first band-pair, then `tilesize`
    for the second band-pair, and so on; this basically corresponds to contiguous
    entries for the northern hemisphere while the info for the southern
    hemisphere is left implicit.

    Constructor arguments
    ---------------------

    tile_counts: array of int
        The number of tiles in each band.

    Properties
    ----------

    thetas, weights, phi0s: array of double
        Specified for every pixel ring (not tile)

    band_pair_count: int
        Number of band pairs

    tile_counts: array of int
        Number of tiles in each band; indexed by band-pair index

    band_offsets:
        Offsets of the starting pixel in each band-pair, indexed by

    ring_lengths:
        Lengths of all rings (one per ring-pair, so npix=2*sum(ring_lengths))
    """

    def __init__(self, tile_counts, tilesize=1, thetas=None, weights=None):
        self.band_pair_count = len(tile_counts)
        self.tile_counts = np.asarray(tile_counts, dtype=np.int32)
        self.tilesize = tilesize
        self.nrings = 2 * self.band_pair_count * tilesize

        if thetas is None or weights is None:
            th, we = _theta_and_weights(self.nrings)
            if thetas is None:
                thetas = th
            if weights is None:
                weights = we

        self.thetas = thetas
        # rescale weights w.r.t. ring size, looked up in libsharp sources
        self.ring_lengths = np.repeat(self.tile_counts * tilesize, tilesize)
        self.weights = weights * 2 * np.pi / self.ring_lengths
        self.phi0s = 0.5 * 2 * np.pi / self.ring_lengths
        self.band_offsets = 2 * tilesize**2 * np.concatenate([[0], np.cumsum(self.tile_counts)])
        self.npix = self.band_offsets[-1]
        self.ntiles = 2 * np.sum(self.tile_counts)

    def compute_weight_map(self):
        return scatter_to_rings(self, np.concatenate([self.weights, self.weights[::-1]]))

    def with_tilesize(self, tilesize):
        """
        Return a grid with matching tile setup, but another tilesize.
        """
        return SymPixGrid(self.tile_counts, tilesize)

    def get_strip(self, iband, phi_start, phi_stop):
        """
        Get a horizontal strip of tiles within a given single tile band
        (numbered `iband`) that covers the arc spanned by
        ``[phi_start, phi_stop]``, inclusive. `iband` refers to a band
        index on the northern hemisphere.

        Returns an array of tile indices in the band, sorted by how
        they would be picked by a scan from phi_start to phi_stop.

        Definition of tile for this purpose: First tile in band covers
        [0, 2 * pi / ntiles_in_band).

        Negative `phi_start`, `phi_stop` are allowed (so that you can
        potenitally cross the ring origin and have two 'sets' of pixels
        returned, one consecutive from the end of the ring before a consecutive
        from the beginning of the ring).

        If phi_start and phi_stop are sufficiently far apart you will get
        some pixels repeated twice.
        """
        def within_2pi(a): return (-2 * pi <= a < 2 * pi)

        ntiles_in_ring = self.tile_counts[iband]
        dphi = 2 * pi / ntiles_in_ring  # size of each pixel

        if not (within_2pi(phi_start) and within_2pi(phi_stop)):
            raise ValueError('phi_start and phi_stop must be within [-2pi, 2pi)')
        # if both are negative, reduce to both positive case
        if phi_start < 0 and phi_stop < 0:
            phi_start += 2 * pi
            phi_stop += 2 * pi

        # we work with pixel indices relative to start of ring, then simply increment
        # with index of zero-pixel afterwards

        if phi_start >= 0:
            # return a single section
            result = np.arange(int(np.floor(phi_start / dphi)),
                               int(np.ceil(phi_stop / dphi)))
        else:
            # we're in the split ring case, get two sections
            result = np.concatenate([
                np.arange(int(np.floor((phi_start + 2 * pi) / dphi)), ntiles_in_ring),
                np.arange(0, int(np.ceil(phi_stop / dphi)))])
        return result



POSSIBLE_INCREMENTS = (
#    Fraction(8, 7),
#    Fraction(7, 6),
                       Fraction(6, 5),
                       Fraction(5, 4),
                       Fraction(4, 3),
                       Fraction(1, 1),
                       Fraction(2, 1),
                       Fraction(3, 1),
#                       Fraction(4, 1),
#                       Fraction(5, 1),
#                       Fraction(6, 1),
#                       Fraction(7, 1),
#                       Fraction(8, 1),
                       )

def _theta_and_weights(nrings):
    x, weights = legendre_roots(nrings)
    thetas = np.arccos(x[::-1][:x.shape[0]//2])
    weights = weights[:weights.shape[0]//2]
    return thetas, weights

def roundup(x, k):
    return x if x % k == 0 else x + k - x % k

def get_min_ring_length(theta, lmax, T=None):
    """
    Returns the minimal number of pixels on a ring on latitude theta
    to represent a signal of size lmax. theta can be an array.
    """
    if T is None:
        T = max(100, 0.01 * lmax)
    ct = np.abs(np.cos(theta))
    st = np.sin(theta)
    mmax = ct + np.sqrt(ct**2 + lmax**2 * st**2 + T)
    mmax = np.floor(mmax).astype(int) + 1
    return 2 * mmax + 1

def find_optimal_ring_lengths(thetas, n_start, optimal_lengths, possible_increments,
                              undersample=False, cost_function=None):
    """
    Use dynamic programming to figure out the optimal ring lengths
    where the possible increments are given by `possible_increments`
    as list of Fraction. The cost is computed as the distance between
    min_lengths and the chosen lengths.
    """
    if cost_function is None:
        def cost_function(thetas, iband, length, opt_length):
            if undersample and ringlen > optimal_lengths[i]:
                return np.inf
            elif not undersample and ringlen < optimal_lengths[i]:
                return np.inf
            else:
                return (length - opt_length)**2
    
    assert all(isinstance(f, Fraction) for f in possible_increments)
    nrings = len(optimal_lengths)
    G = [] # list of { ringlen: (cost, prev_ring_len) }, used for caching results
               # from previously computed rings
    G.append({n_start: (0, None)})
    for i in range(1, nrings):
        possibilities = {}
        for prev_len, (prev_cost, prev_prev_len) in G[i - 1].items():
            for inc in possible_increments:
                #if inc != 1 and prev_prev_len is not None and prev_len != prev_prev_len:
                #    # we always want to rings of same length in a row
                #    continue
                ringlen = prev_len * inc
                if ringlen > 3 * optimal_lengths[-1]:
                    # no point to continue along this path
                    continue
                if ringlen.denominator != 1:
                    # Don't want a fractional number of tiles..
                    continue
                ringlen = ringlen.numerator
                cost = prev_cost + cost_function(thetas, i, ringlen, optimal_lengths[i])
                if cost < possibilities.get(ringlen, (np.inf, None))[0]:
                    possibilities[ringlen] = (cost, prev_len)
        G.append(possibilities)

    # Find the end solution with the lowest cost
    result = np.zeros(nrings, dtype=np.int)
    solutions = [(cost, ring_len, prev_ring_len)
                 for ring_len, (cost, prev_ring_len) in G[-1].items()]
    cost, result[nrings - 1], prev_ring_len = sorted(solutions)[0]
    # Backtrack the solution to the starting ring.
    for i in range(nrings - 2, -1, -1):
        result[i] = prev_ring_len
        prev_ring_len = G[i][prev_ring_len][1]
    return cost, result

def make_sympix_grid(nrings_min, k, undersample=False, cost_function=None, n_start=None):
    """
    Makes our special multi-grid grid that is optimized for precomputing rotationally
    invariant operators in pixel domain.

    nrings_min: minimal number of rings; a grid with another nrings >= nrings_min will be
    returned.

    k: tilesize

    Returns a SymPixGrid object descripting the grid.
    """
    def is_acceptable_ring_length(n):
        """
        Determine whether `n` is on the form ``2^a * 3^b * 5^c * k``;
        any other prime factors would non-productively be a part of every ring
        length.
        """
        if n % k != 0:
            return False
        else:
            n //= k
            for factor in sympy.factorint(n).keys():
                if factor not in [2, 3, 5, k]:
                    return False
            else:
                return True
    # Compute for upper half only
    nrings = roundup(nrings_min, 2 * k)
    thetas, weights = _theta_and_weights(nrings)
    # ring_lengths_min is monotonically increasing, so we're bounded by every k-th theta.
    # ntiles_min_arr will contain a lower bound on number of tiles on a ring
    if undersample:
        ntiles_optimal_arr = [roundup(n, k) // k for n in get_min_ring_length(thetas[::k], nrings - 1)]
    else:
        ntiles_optimal_arr = [roundup(n, k) // k for n in get_min_ring_length(thetas[k - 1::k], nrings - 1)]

    if n_start is None:
        n_start = ntiles_optimal_arr[0]
        while not is_acceptable_ring_length(n_start):
            n_start += 1

    cost, tile_counts = find_optimal_ring_lengths(thetas, n_start, ntiles_optimal_arr,
                                                  POSSIBLE_INCREMENTS,
                                                  undersample=undersample,
                                                  cost_function=cost_function)
    return SymPixGrid(tile_counts, tilesize=k, thetas=thetas, weights=weights)

def plot_sympix_grid_efficiency(nrings_min, grid):
    """
    Debug/diagnositc plot to verify a given grid found.
    """
    from matplotlib.pyplot import clf, gcf, draw
    # Get pure reduced legendre grid
    x0, w0 = legendre_roots(nrings_min)
    thetas0 = np.arccos(x0[::-1])
    lens0 = get_min_ring_length(thetas0, grid.nrings - 1)

    lens = grid.ring_lengths
    lens = np.concatenate([lens, lens[::-1]])
    thetas = np.concatenate([grid.thetas, np.pi - grid.thetas[::-1]])

    clf()
    fig = gcf()
    ax0 = fig.add_subplot(1,2,1)
    ax1 = fig.add_subplot(1,2,2)

    pix_dists0 = 2 * np.pi * np.sin(thetas0) / lens0
    pix_dists = 2 * np.pi * np.sin(thetas) / lens

    ax0.plot(thetas0, pix_dists0)
    ax0.plot(thetas, pix_dists)

    ax1.plot(thetas0, lens0)
    ax1.plot(thetas, lens)
    draw()


class CscMaker(object):
    def __init__(self, n, nnz, dtype=np.double):
        self.indices = np.zeros(nnz, dtype=np.int32)
        self.indptr = np.zeros(n + 1, dtype=np.int32)
        self.data = np.zeros(nnz, dtype=dtype)
        self.n = n
        self.colidx = -1 # should call start_column initially
        self.ptr = 0
        self.examples = {}  #

    def start_column(self):
        self.colidx += 1
        self.indptr[self.colidx] = self.ptr

    def put(self, rowidx, value):
        assert 0 <= rowidx < self.n
        self.indices[self.ptr] = rowidx
        self.data[self.ptr] = value
        self.ptr += 1

    def finish(self):
        self.start_column()
        self.indptr[self.colidx:] = self.ptr
        self.indices = self.indices[:self.ptr]
        self.data = self.data[:self.ptr]

    def as_csc_matrix(self):
        from scipy.sparse import csc_matrix
        return csc_matrix((self.data, self.indices, self.indptr), shape=(self.n, self.n))


class LabelMatrixMaker(CscMaker):
    """
    Extends CscMaker to record a number of examples, (ipix, jpix) for each
    label recorded in the matrix.

    It also adds upper-triangular mirrored half with shifted
    labels if reflect=True. In that case label_allocated must also be set
    to a `Counter` instance.
    """
    def __init__(self, n, nnz, reflect=False, label_allocator=None):
        CscMaker.__init__(self, n, nnz, np.int32)
        self.examples = {}
        self.reflect = reflect
        if self.reflect:
            # Also store matrix entries in rows, and emit them when we start a new column.
            # Each row is a list of tuples (col, label)
            self.rows = [[] for i in range(n)]
            self.label_allocator = label_allocator
            self.upper_labels = {}  # lower_label : corresponding upper label

    def put_example(self, rowidx, label):
        if label not in self.examples:
            self.examples[label] = (rowidx, self.colidx)

    def put(self, rowidx, label):
        self.put_example(rowidx, label)
        if self.reflect:
            if rowidx < self.colidx:
                raise ValueError()
            if rowidx > self.colidx:
                try:
                    upper_label = self.upper_labels[label]
                except KeyError:
                    self.upper_labels[label] = upper_label = self.label_allocator.next()
                self.rows[rowidx].append((self.colidx, upper_label))
        return CscMaker.put(self, rowidx, label)

    def start_column(self):
        CscMaker.start_column(self)
        if self.reflect and self.colidx < self.n:
            for col, label in self.rows[self.colidx]:
                self.put_example(col, label)
                CscMaker.put(self, col, label)


class Counter(object):
    def __init__(self, value):
        self.value = value

    def next(self):
        r = self.value
        self.value += 1
        return r

    def multi_next(self, n):
        r = range(self.value, self.value + n)
        self.value += n
        return r


def sympix_csc_neighbours(grid, corner_factor=0.6, lower_only=True):
    """
    Given a sympix grid, compute the lower half of a CSC neighbour
    matrix.

    This treat the sympix grid as if it isn't tiled at all (i.e. it's
    the CSC neighbour graph of tiles, not individual pixels).

    Returns a scipy.sparse.csc_matrix of integer dtype, where the
    integer values refers to a unique label of the case one is in
    (symmetric with other elements with the same label).

    For the purposes of labelling we assume that every ring length
    is repeated at least twice, so that it's the same either below
    or above.
    """
    # m is a helper object that produces the CSC matrix
    offsets = 2 * np.concatenate([[0], np.cumsum(grid.tile_counts)])
    ntiles = offsets[-1]
    labels = Counter(0)
    nnz = 6 * ntiles + grid.tile_counts[0]**2
    if not lower_only:
        nnz *= 2
    m = LabelMatrixMaker(ntiles, nnz,
                         reflect=not lower_only, label_allocator=labels)

    # Comments are with respect to traversing the northern half;
    # 'right' and 'below' means both increasing array index and
    # increasing theta/phi

    # i - ring-pair index
    # j - pixel within ring
    # offset - offset of first pixel in ring
    # nj = number of pixels in ring

    # There's a special case for the last ring pair (the equatorial ring pair)
    # which is woven in throughout

    for i in range(grid.band_pair_count):
        nj = grid.tile_counts[i]
        dphi = 2 * pi / nj
        last_ring = (i == grid.band_pair_count - 1)
        dphi_next = dphi if last_ring else 2 * pi / grid.tile_counts[i + 1]
        offset = offsets[i]
        offset_next = offset + nj if last_ring else offsets[i + 1]
        tile_count = grid.tile_counts[i]
        # ^ in the last_ring case, the 'next' ring is the southern ring of the same pair

        if i > 0 and not last_ring and 0:
            if sum([tile_count == grid.tile_counts[i - 1],
                    tile_count == grid.tile_counts[i + 1]]) == 0:
                raise NotImplementedError('We do not support grids that does not repeat tile counts '
                                          'at least twice (except for first ring), due to loss of symmetry')

        # For each band-pair we pick new labels for 'left', 'same', 'right' neighbours, but
        # the relationship is the same for all tiles in the band-pair
        label_same, label_left, label_right = labels.multi_next(3)

        # Figure out how many sets of labels we need for the angles between
        # tiles on current ring and tiles on next ring
        if last_ring:
            tile_count_below = tile_count
            increase = cycle_length = cycle_length_below = 1
        else:
            tile_count_below = grid.tile_counts[i + 1]
            increase = Fraction(int(tile_count), int(tile_count_below))
            cycle_length = increase.numerator
            cycle_length_below = increase.denominator

        # we cache the lookup of tiles on the band below; the tiles_below structure
        # has the format
        #   tiles_below[j % cycle_length] == [
        #      (index_of_tile_below_as_in_first_case, label), ...]
        tiles_below_table = [None] * cycle_length

        for south in [False, True]:
            if south:
                # The southern half is a copy of the northern half, just with
                # different pixel indices. All comments below treat the northern
                # case.
                offset += nj
                if not last_ring:
                    offset_next += nj if last_ring else grid.tile_counts[i + 1]

            for j in range(nj):
                m.start_column()
                m.put(offset + j, label_same)

                # For our own ring, we select everything if on the pole ring,
                # otherwise the left and right neighbour
                if i == 0:  # pole ring
                    for k in range(offset + j + 1, offset + nj):
                        ## TODO: we're too lazy now to properly compute symmetry labels for polar region
                        m.put(k, labels.next())
                elif j == 0:  # first pixel on the ring; left neighbour is last on ring
                    m.put(offset + j + 1, label_right)  # right
                    m.put(offset + nj - 1, label_left)  # left
                elif j == nj - 1:
                    # last pixel on ring; both left and right neighbour comes before in ordering
                    pass
                else:
                    # interior pixel; left neighbour comes before in ordering and is ignored,
                    # right neighbour comes after
                    m.put(offset + j + 1, label_right)

                if last_ring and south:
                    # last column block of CSC matrix, only within-ring couplings needed
                    continue

                tiles_below = tiles_below_table[j % cycle_length]
                if tiles_below is None:
                    # Need to figure them out and store in table. (Primary purpose of tabulation
                    # is to reuse labels for the cases, not speed).

                    # For the ring below, we use SymPixGrid.get_strip to select pixels based on
                    # a phi range. First find edges on current ring
                    phi_left = j * dphi
                    phi_right = (j + 1) * dphi
                    # then go out a certain distance on next ring
                    phi_start = phi_left - corner_factor * dphi_next
                    phi_stop = phi_right + corner_factor * dphi_next
                    if phi_stop > 2 * pi:
                        # TOOD: move to get_strip
                        phi_start -= 2 * pi
                        phi_stop -= 2 * pi

                    if last_ring:
                        next_band = i
                    else:
                        next_band = i + 1
                    tiles_below_indices = grid.get_strip(next_band, phi_start, phi_stop - dphi/100)

                    tiles_below_labels = labels.multi_next(len(tiles_below_indices))
                    tiles_below = tiles_below_table[j % cycle_length] = zip(tiles_below_indices, tiles_below_labels)

                # Emit to matrix; we need to a) shift indices right to current position, b) sort by
                # that new index
                shift = (j // cycle_length) * cycle_length_below

                tiles_below_shifted = [((k + shift) % tile_count_below + offset_next, lab) for k, lab in tiles_below]
                tiles_below_shifted.sort()
                for k, lab in tiles_below_shifted:
                    m.put(k, lab)

    m.finish()

    example_i = np.zeros(labels.value, dtype=np.int32)
    example_j = np.zeros(labels.value, dtype=np.int32)
    for key, (i, j) in m.examples.items():
        example_i[key] = i
        example_j[key] = j

    return m.as_csc_matrix(), (example_i, example_j)


def sympix_plot(grid, map, image=None):
    """
    Plots a map to a rectangle. The `image` 2D output array is assumed to
    represent [0, 2pi] x [0, pi], and then the angles are mapped to pixels,
    and the map plotted to those pixels.
    """
    if image is None:
        image = np.zeros((300, 300))

    ny, nx = image.shape
    assert ny % 2 == 0

    phis = (np.linspace(1e-5, 2 * np.pi - 1e-5, image.shape[1]) + np.pi) % (2 * np.pi)

    # `rings` maps line in 2D image to the corresponding ring index in the input map,
    # as counted from north pole
    sphere_thetas = np.concatenate([grid.thetas, np.pi - grid.thetas[::-1]])
    image_thetas = np.linspace(1e-5, np.pi, image.shape[0], endpoint=False)

    rings = np.digitize(image_thetas, sphere_thetas) - 1
    tilesize = grid.tilesize

    for i, iring in enumerate(rings):
        # i is index of image row, iring is the corresponding map ring index
        if iring < 0 or iring >= grid.nrings:
            image[i, :] = np.nan
        else:
            is_north = (iring < grid.nrings // 2)
            iring_as_north = iring if is_north else grid.nrings - iring - 1
            iband = iring_as_north // tilesize
            i_within_band = iring_as_north % tilesize
            ntiles = grid.tile_counts[iband]
            ringlen = tilesize * ntiles
            pixel_borders = np.linspace(0, 2 * np.pi, ringlen, endpoint=False)
            js = np.digitize(phis, pixel_borders) - 1

            offset_from_band_start = js * tilesize + i_within_band
            if is_north:
                offsets = grid.band_offsets[iband] + offset_from_band_start
            else:
                bandsize = ringlen * tilesize
                offsets = grid.band_offsets[iband] + bandsize + offset_from_band_start
            image[i, :] = map[offsets]

    return image


def scatter_to_rings(grid, ringvals):
    """
    Produce a map with a constant value per ring.
    """
    out = np.empty(grid.npix, dtype=ringvals.dtype)
    k = grid.tilesize
    for iband in range(grid.band_pair_count):
        offset = grid.band_offsets[iband]
        ringlen = grid.tile_counts[iband] * k
        bandlen = ringlen * k
        # North
        for iring in range(k):
            t = offset + iring
            out[t:t + bandlen:k] = ringvals[iband * k + iring]
        # South
        for iring in range(k):
            t = offset + bandlen + iring
            out[t:t + bandlen:k] = ringvals[-(iband * k + iring) - 1]
    return out




def udgrade(from_grid, to_grid, x):
    # Tested in scripts/udgrade_sympix.py for now. TODO move into unit tests.
    assert x.shape[0] == from_grid.npix and x.ndim == 1

    assert np.all(from_grid.tile_counts == to_grid.tile_counts)
    if to_grid.tilesize > from_grid.tilesize:
        # ugrade
        factor = to_grid.tilesize // from_grid.tilesize
        # copy along rows
        x_3d = x.reshape((from_grid.npix // from_grid.tilesize**2, from_grid.tilesize, from_grid.tilesize))
        x_3d = np.repeat(x_3d, factor, axis=1)
        x = x_3d.flatten()
        # copy along columns
        x = np.repeat(x, factor)
        return x
    else:
        assert from_grid.tilesize % to_grid.tilesize == 0
        factor = from_grid.tilesize // to_grid.tilesize
        x = x.reshape((from_grid.npix // from_grid.tilesize**2, from_grid.tilesize // factor, factor, from_grid.tilesize // factor, factor))
        return x.sum(axis=4).sum(axis=2).flatten()
