import jax.numpy as jnp
from jax import jit
from jax.lax import scan

@jit
def get_ccfvel_argmax(v0, Xf, vel_weight, cosg, vsini, vmac, vsigma):
    _cosg = jnp.where(jnp.abs(cosg)<1, cosg, cosg/jnp.abs(cosg))
    _sing = jnp.sqrt(1.-_cosg**2)
    dv2 = (vels[:,None,None] - vel_weight*vsini)**2
    sigma2_cos = vsigma**2 + 0.5*vmac**2*_cosg**2
    sigma2_sin = vsigma**2 + 0.5*vmac**2*_sing**2
    dvijk = jnp.exp(-0.5*dv2/sigma2_cos) + jnp.exp(-0.5*dv2/sigma2_sin)
    Cv = jnp.sum(dvijk*Xf, axis=2)
    return vels[jnp.argmax(Cv, axis=0)]


@jit
def get_ccfvel_itrnr(v0, Xf, vel_weight, cosg, vsini, vmac, vsigma, nitr=50):
    v = v0

    _cosg = jnp.where(jnp.abs(cosg)<1, cosg, cosg/jnp.abs(cosg))
    _sing = jnp.sqrt(1.-_cosg**2)
    sigma2_cos = vsigma**2 + 0.5*vmac**2*_cosg**2
    sigma2_sin = vsigma**2 + 0.5*vmac**2*_sing**2
    u = vel_weight*vsini

    def body_fun(v, i):
        dv = v[:,None] - u
        dv2 = dv**2
        exp_cos, exp_sin = jnp.exp(-0.5*dv2/sigma2_cos), jnp.exp(-0.5*dv2/sigma2_sin)
        dvijX = (exp_cos/sigma2_cos + exp_sin/sigma2_sin)*Xf
        v = jnp.sum(u*dvijX, axis=1)/jnp.sum(dvijX, axis=1)
        return (v, None)
    v, _ = scan(body_fun, v, jnp.arange(nitr))

    def body_fun2(v, i):
        dv = v[:,None] - u
        dv2 = dv**2
        exp_cos, exp_sin = jnp.exp(-0.5*dv2/sigma2_cos), jnp.exp(-0.5*dv2/sigma2_sin)
        dvij = -dv/sigma2_cos*exp_cos - dv/sigma2_sin*exp_sin
        dccf = jnp.sum(dvij * Xf, axis=1)
        ddvij = (dv2/sigma2_cos-1.)/sigma2_cos*exp_cos + (dv2/sigma2_sin-1.)/sigma2_sin*exp_sin
        ddccf = jnp.sum(ddvij * Xf, axis=1)
        return (v-dccf/ddccf, None)
    v, _ = scan(body_fun2, v, jnp.arange(2))

    return v
