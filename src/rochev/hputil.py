import numpy as np
import healpy as hp

def create_hpmap(nside):
    """create healpix map
    
        Args:
            nside (int)

        Returns:
            1D array with length npix determined from nside
    
    """
    npix = hp.nside2npix(nside)
    return np.ones(npix)


def hpmap_info(m):
    """information of healpix map
    
        Args:
            healpix map (1D array)

        Returns:
            1D arrays of theta & phi corresponding to map, number of pixels (npix)
    
    """
    npix = len(m)
    nside = hp.npix2nside(npix)
    thetas, phis = hp.pix2ang(nside, np.arange(npix))
    print ("approximate resolution: %.2fdeg"%(hp.nside2resol(nside, arcmin=True)/60.))
    return thetas, phis, npix


def spot_hpmap(m, spotpos, spotsize):
    """util function to create a map with circular spots
    
        Args:
            m: healpix map (1D array)
            spotpos: theta and phi of spot centers
            spotsize: angular radii of spots

        Returns:
            healpix map
    
    """
    mret = np.ones(len(m))
    spotpos *= np.pi/180.
    spotsize *= np.pi/180.
    thetas, phis, _ = hpmap_info(m)
    for ipix in range(len(m)):
        theta, phi = thetas[ipix], phis[ipix]
        for spot, rad in zip(spotpos, spotsize):
            stheta, sphi = spot[0], spot[1]
            cosd = np.sin(stheta)*np.sin(theta)*np.cos(sphi-phi)+np.cos(stheta)*np.cos(theta)
            if np.arccos(cosd) < rad:
                mret[ipix] = 0.3
    return mret
