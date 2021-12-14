__all__ = ["fvmodel_y"]

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corner
from .hputil import *
import jax.numpy as jnp
from jax import jit
from jax.lax import scan

#%%
import seaborn as sns
sns.set(style='ticks', font_scale=1.6, font='sans-serif')
#sns.set_palette('colorblind')
plt.rcParams["figure.figsize"] = (18,6)
from matplotlib import rc
rc('text', usetex=True)

#%%
#wavaa_ref = 4598. # G14 center
#wavaa_ref = 6350. # S12 center

#%%
from scipy.integrate import quad
def wavfactor(tpole, wav_low, wav_high, band=None):
    if band is None:
        band = lambda x: 1.
    func_num = lambda x: x**4*np.exp(x)/(np.exp(x)-1)**2*band(x)
    func_den = lambda x: x**3/(np.exp(x)-1)*band(x)
    x_low = 2.397958702 / wav_high / (tpole/6000.)
    x_high = 2.397958702 / wav_low / (tpole/6000.)
    return quad(func_num, x_low, x_high)[0]/quad(func_den, x_low, x_high)[0]

@jit
def normalized_intensity_band(t, wavfactor):
    return 1. + wavfactor * (t-1.)

#%%
"""
wav, tpole = 0.4598, 4500
wf = wavfactor(tpole, 0.4, 0.8)
Iband = normalized_intensity(tpole*1.05, tpole, wf)
wavs = np.linspace(4000, 8000, 50)
Imono = []
for w in wavs:
    Imono.append(normalized_intensity_mono(tpole*1.05, tpole, w))
"""

#%%
@jit
def normalized_intensity(t, tpole, wavaa_ref):
    #wavaa, teff = 4598, 4600
    #wav = wavaa_ref*1e-10*1e6 #um
    wav = wavaa_ref*1e-10*1e6 #um
    x = 2.397958702 / wav / (t/6000.)
    xpole = 2.397958702 / wav / (tpole/6000.)
    return (jnp.exp(xpole)-1.)/(jnp.exp(x)-1.)

@jit
def limbdark(cosg, u1, u2):
    return 1. - (1. - cosg)*(u1 + u2*(1. - cosg))

#%%
@jit
def fvmodel(minfo, phases, inc, u1, u2, q, a, beta, T0, wavaa_ref, omega=1.):
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
    """
    def get_x(x, i):
        invx = C*a - 0.5*omega**2*(1+q)*sint*sint*x*x - q*(1 + (3*cospsi*cospsi-1)/2.*x*x + 0.5*(5*cospsi**3-3*cospsi)*x**3 + (35*cospsi**4 - 30*cospsi**2 + 3)/8.*x**4)
        return (1./invx, None)
    xinit = jnp.ones_like(cospsi)/a
    x, _ = scan(get_x, xinit, jnp.arange(5))
    r = a*x
    """

    r2 = r*r
    d3 = jnp.sqrt(a*a-2*a*r*cospsi+r2)**3.
    Fj = -1./r2 - q*r/d3
    Fjrot = Fj + 2*fsync*r
    Gj = q*a/d3 - q/a/a
    gx = ivec1[:,None] * (Fjrot * sint * cosp + Gj) + ivec2[:,None] * (-Fjrot * sint * sinp)
    gy = ivec1[:,None] * (Fjrot * sint * sinp) + ivec2[:,None] * (Fjrot * sint * cosp + Gj)
    gz = ivec0[:,None] * (Fj * cost)
    gnorm = jnp.sqrt(gx*gx + gy*gy + gz*gz)
    T = T0 * gnorm**beta

    cosg = (-gy*sini - gz*cosi)/gnorm

    nx = sint*(ivec1[:,None]*cosp-ivec2[:,None]*sinp)
    ny = sint*(ivec2[:,None]*cosp+ivec1[:,None]*sinp)
    nz = ivec0[:,None] * cost
    cosbeta = (-nx*gx - ny*gy - nz*gz)/gnorm

    fnorm = 1./npix*4/(1.-u1/3.-u2/6.)
    int_planck = normalized_intensity(T, T0, wavaa_ref)
    int_ld = limbdark(cosg, u1, u2)
    int = fnorm * int_planck * int_ld
    dA = r2 / cosbeta

    Xf = jnp.where(cosg > 0, cosg, 0) * dA * int
    V = (-ivec1[:,None] * sint * cosp + ivec2[:,None] * sint * sinp) * r

    return Xf, V, cosg, r
    #return flux, vel, Xf, Xv, cosg, V, r, T/T0

@jit
def fvmodel_band(minfo, phases, inc, u1, u2, q, a, beta, wavfactor, omega=1.):
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
    T = gnorm**beta

    cosg = (-gy*sini - gz*cosi)/gnorm

    nx = sint*(ivec1[:,None]*cosp-ivec2[:,None]*sinp)
    ny = sint*(ivec2[:,None]*cosp+ivec1[:,None]*sinp)
    nz = ivec0[:,None] * cost
    cosbeta = (-nx*gx - ny*gy - nz*gz)/gnorm

    fnorm = 1./npix*4/(1.-u1/3.-u2/6.)
    #int_planck = normalized_intensity(T, T0, wavaa_ref)
    int_planck = normalized_intensity_band(T, wavfactor)
    int_ld = limbdark(cosg, u1, u2)
    int = fnorm * int_planck * int_ld
    dA = r2 / cosbeta

    Xf = jnp.where(cosg > 0, cosg, 0) * dA * int
    V = (-ivec1[:,None] * sint * cosp + ivec2[:,None] * sint * sinp) * r

    return Xf, V, cosg, r
    #return flux, vel, Xf, Xv, cosg, V, r, T/T0

#%%
@jit
def fvmodel_y(minfo, phases, inc, u1, u2, q, a, y, omega=1.):
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

    fnorm = 1./npix*4/(1.-u1/3.-u2/6.)
    #int_planck = normalized_intensity(T, T0, wavaa_ref)
    int_gd = gnorm**y
    int_ld = limbdark(cosg, u1, u2)
    int = fnorm * int_gd* int_ld
    dA = r2 / cosbeta

    Xf = jnp.where(cosg > 0, cosg, 0) * dA * int
    V = (-ivec1[:,None] * sint * cosp + ivec2[:,None] * sint * sinp) * r

    return Xf, V, cosg, r
    #return flux, vel, Xf, Xv, cosg, V, r, T/T0

#%%
@jit
def limbdark_interp(cosg, wavs, A, sigma_ld):
    nwavs = (wavs - 600.) / 200.
    mus = cosg.T
    L = jnp.array([jnp.ones_like(mus), mus, mus**2, mus**3, mus**4]).T@A
    ld = L@jnp.array([jnp.ones_like(nwavs), nwavs, nwavs**2, nwavs**3, nwavs**4])
    ld = 1. - (1. - ld) * (1. + sigma_ld)
    return ld

#%%
@jit
def fvmodel_yint(minfo, phases, inc, q, a, y, leff, A, sigma_ld, omega=1.):
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
    int_ld = limbdark_interp(cosg, leff, A, sigma_ld)
    int = fnorm * int_gd * int_ld
    dA = r2 / cosbeta

    Xf = jnp.where(cosg > 0, cosg, 0) * dA * int
    V = (-ivec1[:,None] * sint * cosp + ivec2[:,None] * sint * sinp) * r

    return Xf, V, cosg, r
    #return flux, vel, Xf, Xv, cosg, V, r, T/T0

#%%
xgrid = None
ygrid = None
zgrid = None

#%%
@jit
def limbdark_ginterp(xin, yin):
	ix, iy = jnp.searchsorted(xgrid, xin), jnp.searchsorted(ygrid, yin)
	x1, x2 = xgrid[ix-1], xgrid[ix]
	y1, y2 = ygrid[iy-1], ygrid[iy]
	z11, z12, z21, z22 = zgrid[ix-1,iy-1], zgrid[ix-1,iy], zgrid[ix,iy-1], zgrid[ix,iy]
	return (z11*(x2-xin)*(y2-yin) + z21*(xin-x1)*(y2-yin) + z12*(x2-xin)*(yin-y1) + z22*(xin-x1)*(yin-y1))/(x2-x1)/(y2-y1)

#%%
@jit
def fvmodel_ygint(minfo, phases, inc, q, a, y, leff, sigma_ld, omega=1.):
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
    #return flux, vel, Xf, Xv, cosg, V, r, T/T0


#%%
"""
nside = 8
m = create_hpmap(nside)
thetas, phis, npix = hpmap_info(m)
minfo = (thetas, phis, npix)
print (npix)

#%%
x = np.linspace(0, 4*np.pi, 500)
phases = x + 0.5*np.pi

#%%
inc, q, a, omega, beta, u1, u2, T0, wavaa_ref = 0.5*jnp.pi, 0.5*2, 3, 1., 0.071, 0.551, 0, 6000, 5000
inc, q, a, omega, beta, u1, u2, T0, wavaa_ref = 0.5*jnp.pi, 3, 4.25+0*1.25, 1., 0.54/4, 0.5, 0.2, 4500, 6500
inc, q, a, omega, beta, u1, u2, T0, wavaa_ref = 0.5*jnp.pi, 3, 4.25-1.2, 1., 0.08, 0.5, 0.2, 4500, 6500
inc, q, a, omega, beta, u1, u2, y = 0.5*jnp.pi, 2, 4.25, 1., 0.08, 0.5, 0.2, 0.5
a = 3.

#%%
_q = 1./q
a_crit = (0.6*_q**(2./3.) + np.log(1+_q**(1./3.)))/(0.49*_q**(2./3.))
print (a, a_crit)

#%%
#%time Xf, V, cosg, r = fvmodel(minfo, phases, inc, u1, u2, q, a, beta, T0, wavaa_ref, omega=1.)
%time Xf, V, cosg, r = fvmodel_y(minfo, phases, inc, u1, u2, q, a, y, omega=1.)

#%%
import dill
with open("limbdark.pkl", 'rb') as f:
	ldc = dill.load(f)
xgrid = jnp.array(ldc['mu'])
ygrid = jnp.array(ldc['wavs'])
zgrid = jnp.array(ldc['int']).T

#%%
_mu = jnp.linspace(0, 1, 100)
lds = limbdark_ginterp(cosg, 600.)
plt.yscale('log')
plt.plot(cosg[0,:], lds[0,:], '.')

#%%
A = jnp.array([[ 0.34375458,  0.16214916, -0.01556534, -0.02275041,  0.00779085],
        [ 0.14772613,  0.06298939, -0.0395738 , -0.01389925,  0.01052813],
        [ 1.2581505 , -0.08210222, -0.22727184,  0.16953394, -0.03610829],
        [-1.17782734, -0.34702991,  0.49329115, -0.15231467, -0.00314304],
        [ 0.43003743,  0.20471701, -0.21147477,  0.01928441,  0.02107996]])
ld = limbdark_interp(cosg, 600, A, 0.)
ld2 = limbdark_interp(cosg, 600, A, 0.1)

#%%
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.plot(cosg[200,:], ld[200,:], '.')
plt.plot(cosg[200,:], ld2[200,:], '.')

#%%
idx = jnp.abs(thetas-0.5*jnp.pi)<0.01
plt.plot(phis[idx], r[idx], '.')

#%%
plt.plot(jnp.cos(phis[idx]), jnp.sin(phis[idx]), '-')
plt.plot(r[idx]*jnp.cos(phis[idx]), r[idx]*jnp.sin(phis[idx]), '.')

#%%
hp.mollview(Xf[0], cmap=plt.cm.bone, flip='geo', title='$X_f$')
hp.mollview(V[0], cmap=plt.cm.bone, flip='geo', title='$V$')

#%%
flux = jnp.sum(Xf, axis=1)
plt.plot(x, flux/np.min(flux)*0.91)

#%%
plt.plot(x, 20*jnp.sum(V*Xf, axis=1)/flux)

#%%
#END


#%%
cospsi = jnp.sin(thetas) * jnp.cos(phis)
cost, sint = jnp.cos(thetas), jnp.sin(thetas)
cosp, sinp = jnp.cos(phis), jnp.sin(phis)
fsync = omega**2/a**3*0.5*(1+q)
C = 1 + q/jnp.sqrt(a*a+1)+fsync

#r = jnp.ones_like(cospsi)
#get_r = lambda r: 1./(C - q/jnp.sqrt(a*a-2*a*cospsi+r*r) + q*r*cospsi/a/a - fsync*sint*sint*r*r)
#for i in range(5):
#    print (r[0])
#    r = get_r(r)

def get_r(r, i):
    ret = 1./(C - q/jnp.sqrt(a*a-2*a*cospsi+r*r) + q*r*cospsi/a/a - fsync*sint*sint*r*r)
    return (ret, None)
rinit = jnp.ones_like(cospsi)
r, _ = scan(get_r, rinit, jnp.arange(5))

#%%
cosi, sini = jnp.cos(inc), jnp.sin(inc)
ivec0, ivec1, ivec2 = jnp.ones_like(phases), jnp.cos(phases), jnp.sin(phases)
#cosg = ivec0[:,None] * jvec0 + ivec1[:,None] * jvec1 + ivec2[:,None] * jvec2

#%%
r2 = r*r
d3 = jnp.sqrt(a*a-2*a*cospsi+r2)**3.
Fj = -1./r2 - q*r/d3
Fjrot = Fj + 2*fsync*r
Gj = q*a/d3 - q/a/a
gx = ivec1[:,None] * (Fjrot * sint * cosp + Gj) + ivec2[:,None] * (-Fjrot * sint * sinp)
gy = ivec1[:,None] * (Fjrot * sint * sinp) + ivec2[:,None] * (Fjrot * sint * cosp + Gj)
gz = ivec0[:,None] * (Fj * cost)
gnorm = jnp.sqrt(gx*gx + gy*gy + gz*gz)

#%%
T = T0 * gnorm**beta
dF = normalized_intensity(T, T0, wavaa_ref)
#dF = normalized_intensity_band(T, T0, 5.3)
flux_weights = 1./npix*4/(1.-u1/3.-u2/6.) * dF * r2

#%%
cosg = (-gy*sini - gz*cosi)/gnorm

#%%
nx = sint*(ivec1[:,None]*cosp-ivec2[:,None]*sinp)
ny = sint*(ivec2[:,None]*cosp+ivec1[:,None]*sinp)
nz = ivec0[:,None] * cost
cosbeta = (-nx*gx - ny*gy - nz*gz)/gnorm
#np.shape(cosbeta)

#%%
Xf = flux_weights * limbdark(cosg, u1, u2) * jnp.where(cosg > 0, cosg, 0) / cosbeta

#%%
hp.mollview(r-1., cmap=plt.cm.bone, flip='geo', title='$r-1$')

#%%
hp.mollview(gnorm[0], cmap=plt.cm.bone, flip='geo', title='$g$')
hp.mollview(cosg[0], cmap=plt.cm.bone, flip='geo', title='$\cos\gamma$')
hp.mollview(Xf[0], cmap=plt.cm.bone, flip='geo', title='$X_f$')
#hp.mollview(flux_weights[0], cmap=plt.cm.bone, flip='geo', title='$F_j$')

#%%
flux = jnp.sum(Xf, axis=1)

#%%
plt.ylim(0.97, 1.10)
plt.plot(x, flux/flux[125])

#%%
#plt.ylim(0.97, 1.10)
plt.plot(x, flux/np.min(flux)*0.91)

#%%
jnp.min(r[idx])
idx = np.abs(thetas-0.5*np.pi)<0.1
#plt.axes().set_aspect('equal')
plt.plot(r[idx]*np.cos(phis[idx]), r[idx]*np.sin(phis[idx]), '.')
plt.plot(np.cos(phis[idx]), np.sin(phis[idx]), '-', lw=0.5)
print (jnp.min(r[idx]*np.cos(phis[idx])))
"""
