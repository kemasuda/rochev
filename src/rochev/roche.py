__all__ = ["fvmodel_y"]

#%%
import numpy as np
from .hputil import *
import jax.numpy as jnp
from jax import jit
from jax.lax import scan
from scipy.integrate import quad


@jit
def limbdark(cosg, u1, u2):
    """limb-darkened intensity
    
        Args:
            cosg: foreshoretening factor 
            u1, u2: quadratic limb-darkening coefficients

        Returns:
            intensity (same shape as cosg)

    """
    return 1. - (1. - cosg)*(u1 + u2*(1. - cosg))


@jit
def fvmodel_y(minfo, phases, inc, u1, u2, q, a, y, omega=1.):
    """flux, velocity, foreshoretening, radial distance of pixels
    
        Args:
            minfo: healpix map info
            phases: orbital phases
            inc: incliantion (radian)
            u1, u2: quadratic limb-darkening coefficients
            q: mass ratio (secondary / primary)
            a: semi-major axis / primary's radius
            y: gravity-darkening parameter
            
        Returns:
            Xf: flux of each pixel (0 if invisible)
            V: LOS velocity of each pixel
            cosg: foreshoretening of each pixel
            r: distance of each pixel from the center of the primary
    
    """
    thetas, phis, npix = minfo
    cost, sint = jnp.cos(thetas), jnp.sin(thetas)
    cosp, sinp = jnp.cos(phis), jnp.sin(phis)
    cosi, sini = jnp.cos(inc), jnp.sin(inc)
    ivec0, ivec1, ivec2 = jnp.ones_like(phases), jnp.cos(phases), jnp.sin(phases)

    cospsi = sint * cosp
    fsync = 0.5 * omega**2 / a**3 * (1.+q)
    C = 1. + q/jnp.sqrt(a*a + 1.) + fsync
    def get_r(r, i):
        ret = 1./(C - q/jnp.sqrt(a*a-2*a*r*cospsi+r*r) + q*r*cospsi/a/a - fsync*sint*sint*r*r)
        return (ret, None)
    rinit = jnp.ones_like(cospsi)*0.8
    r, _ = scan(get_r, rinit, jnp.arange(5))

    r2 = r*r
    d3 = jnp.sqrt(a*a-2*a*r*cospsi+r2)**3.
    Fj = -1./r2 - q*r/d3
    Fjrot = Fj + 2*fsync*r
    Gj = q*a/d3 - q/a/a
    gx = ivec1[:,None] * (Fjrot * sint * cosp + Gj) + ivec2[:,None] * (-Fjrot * sint * sinp)
    gy = ivec1[:,None] * (Fjrot * sint * sinp) + ivec2[:,None] * (Fjrot * sint * cosp + Gj)
    gz = ivec0[:,None] * (Fj * cost)
    gnorm = jnp.sqrt(gx*gx + gy*gy + gz*gz)
    #T = T0 * gnorm**beta

    cosg = (-gy*sini - gz*cosi)/gnorm

    nx = sint*(ivec1[:,None]*cosp-ivec2[:,None]*sinp)
    ny = sint*(ivec2[:,None]*cosp+ivec1[:,None]*sinp)
    nz = ivec0[:,None] * cost
    cosbeta = (-nx*gx - ny*gy - nz*gz)/gnorm

    fnorm = 1. / npix*4 / (1. - u1/3. - u2/6.)
    #int_planck = normalized_intensity(T, T0, wavaa_ref)
    int_gd = gnorm**y
    int_ld = limbdark(cosg, u1, u2)
    int = fnorm * int_gd * int_ld
    dA = r2 / cosbeta

    Xf = jnp.where(cosg > 0, cosg, 0) * dA * int
    V = (-ivec1[:,None] * sint * cosp + ivec2[:,None] * sint * sinp) * r

    return Xf, V, cosg, r


# limb-darkening profile from direct interpolation of the model grids
'''
with open("limbdark.pkl", 'rb') as f:
	ldc = dill.load(f)
roche.xgrid = jnp.array(ldc['mu'])
roche.ygrid = jnp.array(ldc['wavs'])
roche.zgrid = jnp.array(ldc['int']).T
'''
xgrid = None
ygrid = None
zgrid = None

@jit
def limbdark_ginterp(xin, yin):
    """interpolated limb-darkening profile
    
        Args:
            xin: mu
            yin: wavelength

        Returns:
            intensity
    """
    ix, iy = jnp.searchsorted(xgrid, xin), jnp.searchsorted(ygrid, yin)
    x1, x2 = xgrid[ix-1], xgrid[ix]
    y1, y2 = ygrid[iy-1], ygrid[iy]
    z11, z12, z21, z22 = zgrid[ix-1,iy-1], zgrid[ix-1,iy], zgrid[ix,iy-1], zgrid[ix,iy]
    return (z11*(x2-xin)*(y2-yin) + z21*(xin-x1)*(y2-yin) + z12*(x2-xin)*(yin-y1) + z22*(xin-x1)*(yin-y1))/(x2-x1)/(y2-y1)


@jit
def fvmodel_ygint(minfo, phases, inc, q, a, y, leff, sigma_ld, omega=1.):
    """flux, velocity, foreshoretening, radial distance of pixels
    
        Args:
            minfo: healpix map info
            phases: orbital phases
            inc: incliantion (radian)
            q: mass ratio (secondary / primary)
            a: semi-major axis / primary's radius
            y: gravity-darkening parameter
            leff: wavelength at which LD profile is evaluated
            sigma_ld: fractional uncertainty in the LD profile
            
        Returns:
            Xf: flux of each pixel (0 if invisible)
            V: LOS velocity of each pixel
            cosg: foreshoretening of each pixel
            r: distance of each pixel from the center of the primary
    
    """
    thetas, phis, npix = minfo
    cost, sint = jnp.cos(thetas), jnp.sin(thetas)
    cosp, sinp = jnp.cos(phis), jnp.sin(phis)
    cosi, sini = jnp.cos(inc), jnp.sin(inc)
    ivec0, ivec1, ivec2 = jnp.ones_like(phases), jnp.cos(phases), jnp.sin(phases)

    cospsi = sint * cosp
    fsync = 0.5 * omega**2 / a**3 * (1.+q)
    C = 1. + q/jnp.sqrt(a*a + 1.) + fsync
    def get_r(r, i):
        ret = 1./(C - q/jnp.sqrt(a*a-2*a*r*cospsi+r*r) + q*r*cospsi/a/a - fsync*sint*sint*r*r)
        return (ret, None)
    rinit = jnp.ones_like(cospsi)*0.8
    r, _ = scan(get_r, rinit, jnp.arange(5))

    r2 = r*r
    d3 = jnp.sqrt(a*a-2*a*r*cospsi+r2)**3.
    Fj = -1./r2 - q*r/d3
    Fjrot = Fj + 2*fsync*r
    Gj = q*a/d3 - q/a/a
    gx = ivec1[:,None] * (Fjrot * sint * cosp + Gj) + ivec2[:,None] * (-Fjrot * sint * sinp)
    gy = ivec1[:,None] * (Fjrot * sint * sinp) + ivec2[:,None] * (Fjrot * sint * cosp + Gj)
    gz = ivec0[:,None] * (Fj * cost)
    gnorm = jnp.sqrt(gx*gx + gy*gy + gz*gz)
    #T = T0 * gnorm**beta

    cosg = (-gy*sini - gz*cosi)/gnorm

    nx = sint*(ivec1[:,None]*cosp-ivec2[:,None]*sinp)
    ny = sint*(ivec2[:,None]*cosp+ivec1[:,None]*sinp)
    nz = ivec0[:,None] * cost
    cosbeta = (-nx*gx - ny*gy - nz*gz)/gnorm

    fnorm = 1./npix*4#/(1.-u1/3.-u2/6.)
    #int_planck = normalized_intensity(T, T0, wavaa_ref)
    int_gd = gnorm**y
    int_ld = limbdark_ginterp(cosg, leff) * (1+sigma_ld)
    int = fnorm * int_gd * int_ld
    dA = r2 / cosbeta

    Xf = jnp.where(cosg > 0, cosg, 0) * dA * int
    V = (-ivec1[:,None] * sint * cosp + ivec2[:,None] * sint * sinp) * r

    return Xf, V, cosg, r