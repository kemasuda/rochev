from jax import jit

@jit
def lgcoeffs(wavnm):
    """gravity- and limb-darkening parameters from interpolation of Claret, A., & Bloemen, S. 2011, A&A, 529, A75 based on ATLAS model

        Args:
            wavnum: wavelength in nm

        Returns:
            gravity-darkening coefficient
            two limb-darkening coefficients for quadratic law (u1, u2)
    
    """
    x = (wavnm - 600.)/200.
    fy = lambda x: (((0.06690458*x - 0.15430695)*x + 0.09762098)*x - 0.13098096)*x + 0.54212228
    fu1 = lambda x: (((-0.02030541*x - 0.0565222)*x + 0.16281937)*x - 0.29631237)*x + 0.61250245
    fu2 = lambda x: (((0.02157778*x + 0.07181101)*x - 0.18479906)*x + 0.18314384)*x + 0.13359151
    return fy(x), fu1(x), fu2(x)
