#%%
from .hputil import *
from .roche import fvmodel_y
import jax.numpy as jnp
from jax import jit, vmap

@jit
def kernel(vels, fluxmap, velmap, cosgmap, vmac):
    """kernel function
    
        Args:
            vels: velocities at which kernel is evaluated
            fluxmap: flux of each pixel
            velmap: velocity of each poxel
            cosgmap: cosgamma of each pixel
            vmac: macroturbulence velocity

        Returns:
            kernel (same shape as vels)
    
    """
    dvij = (vels[:, None] - velmap) / vmac
    singmap = jnp.sqrt(1. - cosgmap**2)
    Mv = jnp.sum( (jnp.exp(-(dvij/cosgmap)**2) + jnp.exp(-(dvij/singmap)**2))*fluxmap, axis=1 )
    Mv /= jnp.sum(fluxmap)
    return Mv

# kernel mapped for multiple healpix maps
kernel_vmap = jit(vmap(kernel, (None,0,0,0,None), 0))

@jit
def params_to_kernel(vels, minfo, phases, params):
    """kernels at multiple phases
    
        Args:
            vels: velocities at which kernel is evaluated
            minfo: minfo of healpix map
            params: inc, u1, u2, q, a, y, vmac, vsini

        Returns:
            kernels of shape (len(phases),len(vels))?
    
    """
    inc, u1, u2, q, a, y, vmac, vsini = params
    Xf, vel_weight, cosg, r = fvmodel_y(minfo, phases, inc, u1, u2, q, a, y)
    return kernel_vmap(vels, Xf, vsini*vel_weight, cosg, vmac)
