__all__ = ['get_blend_ratio', 'get_cat_subset']

import numpy as np
from astropy import units as u

PIX_TO_ARCSEC = 0.05 # arcsec per HST ACS pixel (Koekemoer et. al. 2007)
ARCSEC_TO_PIX = 1/PIX_TO_ARCSEC # HST ACS pixel per arcsec


# get blend ratio for objects in one arcmin^2 around center, greater than size and lower than magnitude
def get_blend_ratio(cosmos, center, psf, ra_range=0.5*u.arcmin, dec_range=0.5*u.arcmin, size=None, mag=None, super_e_mag=None, super_sb_mag=None, super_mixed_mag=None, zcut=None):
    """
    Compute a blending ratio for a given PSF, over a given region in the COSMOS catalog.

    Parameters
    ----------
    cosmos : 2d numpy array
        Complete catalog of objects, formatted as a table.
    center : [float, float]
        Coordinates of the center of the tile [ra, dec] over which the blending ratio is to be computed.
    psf : float
        PSF of the target instrument in arcsec.
    ra_range : u.arcmin, optional
        Right Ascension range of the tile, by default 0.5*u.arcmin
    dec_range : u.arcmin, optional
        Declination range of the tile, by default 0.5*u.arcmin
    size : float, optional
        Size cut in arcsec, by default None. Only objects larger than size will be counted.
    mag : float, optional
        Mag cut in ST Mag for HST Mags, by default None. Only objects with lower magnitude (i.e. brighter objects) will be counted.
    super_e_mag : float, optional
        Mag cut in ST Mag for SuperBIT Elliptical Mags, by default None. Only objects with lower magnitude (i.e. brighter objects) will be counted.
    super_sb_mag : float, optional
        Mag cut in ST Mag for SuperBIT Starburst Mags, by default None. Only objects with lower magnitude (i.e. brighter objects) will be counted.
    zcut : float, optional
        Redshift cut, by default None. Only objects with lower redshift will be counted.

    Returns
    -------
    blend_ratio : float
        ratio of blended objects to all objects in the selected tile.
    """   
    # get objects with the specified size and position
    cat = get_cat_subset(cosmos, center, ra_range=ra_range, dec_range=dec_range, sizecut=size, magcut=mag, super_e_mag=super_e_mag, super_sb_mag=super_sb_mag, super_mixed_mag=super_mixed_mag, zcut=zcut)

    if (cat.shape[0]==0):
        return np.nan
    
    # Convert FWHM to arcsec
    cat[:,9] = cat[:,9]*PIX_TO_ARCSEC
    
    # Convert angular coordinates to arcsec
    cat[:,0] = cat[:,0]*u.degree.to(u.arcsec)
    cat[:,1] = cat[:,1]*u.degree.to(u.arcsec)
    
    # Discard all the sources smaller than 1.2*PSF
    cat_hst = np.copy(cat)
    cat = cat[cat[:,9] >= 1.2*psf]
    
    distance = np.zeros((cat.shape[0], cat.shape[0]))
    blending  = np.zeros((cat.shape[0]), dtype=bool)

    # Determine blending between pairs of objects
    for i in range(0, cat.shape[0]):
        for j in range(i, cat.shape[0]):
            if i!=j:
                distance = np.sqrt(np.square(cat[i, 0]-cat[j, 0]) + np.square(cat[i, 1]-cat[j, 1]))
                # Calculate SuperBIT size
                r1 = np.sqrt(cat[i,9]**2 + psf**2)/2
                r2 = np.sqrt(cat[j,9]**2 + psf**2)/2
                if (distance < r1+r2): 
                    # # Blending Algorithm 1: Blend if distance < radii
                    # if min(r1,r2)==r1:
                    #     blending[i]=True
                    # else:
                    #     blending[j]=True
                        
                    # Blending Algorithm 2: Blend if the two objects overlap by more than 50% 
                    area = overlap_area(distance, r1, r2)
                    overlap1 = area/(np.pi * r1**2)
                    overlap2 = area/(np.pi * r2**2)

                    # Designate the smaller object as blended if needed
                    if (overlap1>0.5 or overlap2>0.5):
                        if (r1<r2):
                            blending[i]=True
                        else:
                            blending[j]=True


    # Count blended objects
    count = np.count_nonzero(blending)
    # Count discarded objects
    count = count + (cat_hst.shape[0] - cat.shape[0])
    
    blend_ratio = count / cat_hst.shape[0]
    return blend_ratio



def overlap_area(d, rad1, rad2):
    """
    Return the area of intersection of two circles.
    The circles have radii R and r, and their centres are separated by d.
    
    Parameters
    ----------
    d : float
        Distance between the centres of the cirles
    rad1 : float
        radius of circle 1
    rad2 : float
        radius of circle 2

    Returns
    -------
    overlap_area : float
        area of overlap between the two circles
    """  
    r = min(rad1, rad2)
    R = max(rad1, rad2)

    if d <= abs(R-r):
        # One circle is entirely enclosed in the other.
        return np.pi * min(R, r)**2
    if d >= r + R:
        # The circles don't overlap at all.
        return 0

    r2, R2, d2 = r**2, R**2, d**2
    alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
    beta = np.arccos((d2 + R2 - r2) / (2*d*R))
    
    overlap_area = r2 * alpha + R2 * beta - 0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta))
        
    return overlap_area


def get_cat_subset(cat,center,ra_range=0.5*u.arcmin,dec_range=0.5*u.arcmin, sizecut=None, magcut=None, super_e_mag=None, super_sb_mag=None, super_mixed_mag=None, zcut=None):
    """ get a square subset the catalogue centered at a given position and a size cut in arcseconds"""

    ra_range = ra_range.to(u.deg).value
    dec_range = dec_range.to(u.deg).value
    
    subset = cat[cat[:,1]<center[1]+dec_range]
    subset = subset[subset[:,1]>center[1]-dec_range]

    subset = subset[subset[:,0]<center[0]+ra_range]
    subset = subset[subset[:,0]>center[0]-ra_range]
    
    if sizecut is not None:
        # sizecut = sizecut*ARCSEC_TO_PIX
        subset = size_cut(subset,sizecut)
    
    if magcut is not None:
        subset = subset[subset[:,5]<=magcut]
    
    if super_e_mag is not None:
        subset = subset[subset[:,11]<=super_e_mag]
    
    if super_sb_mag is not None:
        subset = subset[subset[:,12]<=super_sb_mag]
        
    if super_mixed_mag is not None:
        subset = subset[subset[:,13]<=super_mixed_mag]

    if zcut is not None:
        subset = subset[subset[:,2]<=zcut]
    
    return subset

def size_cut(cat,size):
    """ Perform a size cut on the catalogue"""
    temp = np.copy(cat)
    # return cat[temp[:,9]*PIX_TO_ARCSEC >size]
    return cat[temp[:,9] >size]


def mag_cut(cat,mag):
    return cat[cat[:,5]<=mag]



