Update:

- Need to a) use V-cycle, b) also restrict in pixel domain what goes psuedo-inv.

- Masking in the mixing map helps too

- Should NOT scale to the identity matrix I + ..., rather keep Si + ...
  In general there is a diagonal scaling that will reduce the ringing / interaction
  as much as possible so that system is block-diagonal in pixel domain as much as possible,
  that's the optimal scaling, although Si seems to work well. Really well in fact.
  Using identity matrix I works.











OLD:

1) Seems like the Z matrix is irrelevant, just adding the identity,

A ~=  M + I

seems to work really well for priors where the method converges

2) For some priors the method doesn't converge. It *does* converge for lcross=1000
for the CMB component. For lcross larger than that there are problems. It is not
with the signal-to-noise-ratio as such, because increasing both lmax and lcross
and you still loose convergence. However, switching to a finer-resolution band
and we can push lcross upwards somewhat. I.e., it's the relationship between
the prior and the inverse-solver that's the problem.

3) It's the presence of low-resolution bands that is the problem. I.e. for lcross=2000
it sort of works for 857, but when adding 143 it stops working.

4) Tried to play around with smoothing the mask or change its size etc., didnt'
really help.

5) It works really well if the prior dominates entirely, but then again the matrix looks
a lot like the identity matrix...

5) Best idea at this point is that there's some problem with ringing

could try to 