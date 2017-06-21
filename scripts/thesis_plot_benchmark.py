from matplotlib import rc
rc('text', usetex=True)
rc('font', size=18)

clf()
fig = gcf()
fig.set_size_inches(6, 6)

ax = gca()

for bench in benchmarks:
    bench.ploterr()
ax.set_ylim((1e-13, 1))
ax.set_xlabel('Iterations')
ax.set_ylabel(r'$\| {\mathbf{x}}_i - {\mathbf{x}}_{\mathrm{true}} \| / \| {\mathbf{x}}_{\mathrm{true}} \|$')
fig.subplots_adjust(left=0.18, bottom=0.12, right=0.99, top=0.99)
legend()
fig.savefig('/home/dagss/phd/figures/precond_benchmark_multicomp.pdf')
draw()
