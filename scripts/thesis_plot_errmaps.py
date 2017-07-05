




v0 = sharp.sh_synthesis(nside, benchmarks[-1].err_vecs[1][0])
v1 = sharp.sh_synthesis(nside, benchmarks[-1].err_vecs[2][0])


clf()
from matplotlib import rc
fig = gcf()
#rc('text', usetex=True)
#rc('font', size=14)
fig.set_size_inches(5.5, 3)


mollview(v0, fig=fig.number, cbar=False, title='', xsize=1000, min=v0.min(), max=v0.max())
fig.savefig('/home/dagss/writing/psuedoinv/figures/x0.pdf')

mollview(v0 - v1, fig=fig.number, cbar=False, title='', xsize=1000, min=v0.min(), max=v0.max())
fig.savefig('/home/dagss/writing/psuedoinv/figures/x1.pdf')
mollview(v1, fig=fig.number, cbar=False, title='', xsize=1000, min=v0.min(), max=v0.max())
fig.savefig('/home/dagss/writing/psuedoinv/figures/e1.pdf')


#d0 = sharp.sh_synthesis(nside, benchmarks[0].err_vecs[1][0])
d1 = sharp.sh_synthesis(nside, benchmarks[0].err_vecs[2][0])
mollview(d1, fig=fig.number, cbar=False, title='', xsize=1000, min=v0.min(), max=v0.max())
fig.savefig('/home/dagss/writing/psuedoinv/figures/d1.pdf')




#mollview(v0_s, fig=fig.number, cbar=False, title='', xsize=1000, min=v0_s.min(), max=v0_s.max())
#fig.savefig('/home/dagss/writing/psuedoinv/figures/x0_s.pdf')
#mollview(v0_s - v1_s, fig=fig.number, cbar=False, title='', xsize=1000, min=v0_s.min(), max=v0_s.max())
#fig.savefig('/home/dagss/writing/psuedoinv/figures/x0_s.pdf')
#mollview(v1_s, fig=fig.number, cbar=False, title='', xsize=1000, min=v0_s.min(), max=v0_s.max())
#fig.savefig('/home/dagss/writing/psuedoinv/figures/e1_s.pdf')
