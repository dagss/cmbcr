


from cmbcr import sharp

lmax = 10

from cmbcr.healpix_data import get_ring_weights_T

nside = 8
lmax = int(2 * nside)
weights = get_ring_weights_T(nside)
p = sharp.RealMmajorHealpixPlan(nside, lmax, weights=weights)
#p = sharp.RealMmajorGaussPlan(lmax)
Y = hammer(p.synthesis, p.npix_global, (p.lmax + 1)**2)
YtW = hammer(p.analysis, (p.lmax + 1)**2, p.npix_global)

#YtW = hammer(lambda x: sharp.sh_analysis(2*nside, x), (2*nside+1)**2, 12*nside**2)
#print np.linalg.svd(np.dot(YtW, Y))[1].max() * np.linalg.svd(Y)[1].max()

W = (YtW / Y.T)[0, :]
assert not np.any(np.isinf(W))

#W[100:130] = 0

u, s, v = np.linalg.svd(Y.T * np.sqrt(W)[None, :])
print s.min(), s.max()
#s *= np.sqrt(W)



#1.09
#1.08
