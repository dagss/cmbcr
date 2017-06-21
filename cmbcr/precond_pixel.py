import numpy as np

from . import block_matrix
from . import sympix_mg
from . import sympix
from .utils import timed
from .mmajor import lmax_of
from . import sharp


class PixelPreconditioner(object):

    def __init__(self, system, tilesize=8, ninv_factor=2, prior=True):
        lmax = max(system.lmax_list)
        self.grid = sympix.make_sympix_grid(lmax + 1, tilesize, n_start=8)
        grid_ninv = self.grid.with_tilesize(tilesize * ninv_factor)
        
        corner_factor = 0.6#9
        with timed('sympix_csc_neighbours'):
            neighmat, (label_to_i, label_to_j) = sympix.sympix_csc_neighbours(self.grid, lower_only=False,corner_factor=corner_factor)
            neighmat_lower, (label_to_i_lower, label_to_j_lower) = sympix.sympix_csc_neighbours(self.grid, lower_only=True, corner_factor=corner_factor)

        A_sparse = block_matrix.BlockMatrix(neighmat_lower.indptr, neighmat_lower.indices, blockshape=(tilesize**2, tilesize**2))
        with timed('R'):
            for nu in range(system.band_count):
                bl = system.bl_list[nu][:lmax + 1]

                plan_dg = sharp.SymPixGridPlan(grid_ninv, lmax_of(system.winv_ninv_sh_lst[nu]))
                ninv_map = plan_dg.adjoint_analysis(system.winv_ninv_sh_lst[nu])
                R_blocks = sympix_mg.compute_many_YDYt_blocks(
                    grid_ninv, self.grid, bl,
                    np.asarray(label_to_i, dtype=np.int32),
                    np.asarray(label_to_j, dtype=np.int32))
                R = block_matrix.BlockMatrix(neighmat.indptr, neighmat.indices, R_blocks, labels=neighmat.data)
                block_matrix.block_At_D_B(
                    R, R,
                    ninv_map.reshape(grid_ninv.tilesize**2, grid_ninv.ntiles, order='F'),
                    A_sparse)


            k = kp = 0  # DEBUG
            if k == kp and prior:
                Si_blocks = sympix_mg.compute_many_YDYt_blocks(
                    self.grid, self.grid,
                    system.dl_list[k],
                    np.asarray(label_to_i_lower, dtype=np.int32),
                    np.asarray(label_to_j_lower, dtype=np.int32))
                Si_blocks = Si_blocks[:, :, neighmat_lower.data]
                A_sparse.blocks += Si_blocks

                
        self.diagonal_blocks = A_sparse.blocks[:, :, A_sparse.labels[A_sparse.indptr[:-1]]].copy('F')
        block_matrix.block_diagonal_factor(self.diagonal_blocks)

        self.tilesize = tilesize
        self.bs = tilesize**2
        self.plan = sharp.SymPixGridPlan(self.grid, lmax)

    def apply(self, x_lst):
        assert len(x_lst) == 1
        x = x_lst[0].copy()
        x = self.plan.synthesis(x)
        npix = x.shape[0]
        x = x.reshape((self.bs, x.shape[0] // self.bs), order='F')
        block_matrix.block_diagonal_solve(self.diagonal_blocks, x)
        assert not np.any(np.isnan(x))
        x = x.reshape(npix, order='F')
        x = self.plan.adjoint_synthesis(x)
        return [x]
