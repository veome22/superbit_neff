__all__ = ['get_n_mag', 'get_n_fit', 'n_extrapolation']

import numpy as np
from astropy import units as u
from blend_calc import *
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

PIX_TO_ARCSEC = 0.05 # arcsec per HST ACS pixel (Koekemoer et. al. 2007)
ARCSEC_TO_PIX = 1/PIX_TO_ARCSEC # HST ACS pixel per arcsec

def get_n_mag(MAG_MIN, MAG_MAX, increment, PSF, cosmos):
    mags = np.arange(MAG_MIN, MAG_MAX, increment)

    magcut = np.empty(mags.shape)
#     magcut_err =  np.empty(mags.shape)

    for i in range(mags.shape[0]):
        cat = cosmos
        # perform magcut
        cat = cat[cat[:,5]<=mags[i]]
        n_total = cat.shape[0]
        
        # perform size cut
        cat = cat[cat[:,9] * PIX_TO_ARCSEC > 1.2*PSF]
        n_eff = cat.shape[0]
        
        magcut[i] = n_eff / n_total
#         magcut_err[i] = np.sqrt((1/ n_total) + (1/n_eff))
    
    # Interpolated Function
    f_magcut = interp1d(mags, magcut, fill_value="extrapolate")
    
    
    x = np.arange(MAG_MIN, MAG_MAX-(increment+0.1), 0.1)
    y = 2**x * f_magcut(x)
    
    # Combined function
    f = interp1d(x, y, fill_value="extrapolate")
    return f


def get_n_fit(x, data, f_magcut):

    def extrapolation(x, amp, exp):
        return amp * exp**x * f_magcut(x)

    init_vals = [1, 2]
    return curve_fit(extrapolation, x, data, p0=init_vals)

def n_extrapolation(x, amp, exp, f_magcut):
    return amp * exp**x * f_magcut(x)
    