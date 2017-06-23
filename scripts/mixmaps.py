

mix = [system.mixing_maps[nu, 0] for nu in range(system.band_count)]

ms = sum([m * (m.sum() / (m**2).sum()) for m in mix])

for m in mix:
    u = m / ms
    print m.max() / m.min(), u.max() / u.min()
