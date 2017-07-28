


lmax = 30
nside = 16

Y = hammer(lambda x: sharp.sh_synthesis(nside, x), 12*nside**2, (lmax + 1)**2)

J = np.dot(Y, Y.T)

clf()
imshow(J, interpolation='nearest')
colorbar()
draw()

