__all__ = ['stmag_conversion', 'get_stmag', 'get_correction']

import numpy as np


# Get mag from Response Function, Spectrum, Lambda Range, and Scaling Correction to Spectrum
def get_stmag(r, f_lam, lam, correction=1.0):
    """
    Returns the observed ST mag for a given filter and source spectrum

    Args:
        r: Filter Response Function, [unitless]
        f_lam: Source Spectral Flux Density, [ergs/s/cm^2/Angstrom]
        lam: Wavelength range, [Angstrom]
        correction: constant factor to scale f_lam depending on the source

    Returns:
        ST Magnitude
    """
    f_lam = f_lam * correction
    f1 = np.trapz(y=f_lam * r * lam, x=lam)
    f2 = np.trapz(y=r * lam, x=lam) 

    f_exp = f1/f2 # erg s^-1 cm^-2 AA^-1

    mag = -2.5 * np.log10(f_exp) - 21.1
    return mag

# Get the required corrective factor for the spectral flux density
def get_correction(m_exp, m_obs):
    return np.power(10, (m_exp/2.5)-(m_obs/2.5))


# Get SuperBIT magitudes from Observed Mag
def stmag_conversion(mag_obs, z, src_band, target_band, e_template, starb_template):
    """
    Converts ST Mag from one filter to another, assuming both Elliptical and Starburst 
    spectra for the source

    Parameters
    ----------
    mag_obs : float
        ST Mag observed by Source Band, [unitless]

    z : float
        Redshift of source galaxy

    src_band : scipy.interpolate.interpolate.interp1d
        Band we are converting from

    target_band : scipy.interpolate.interpolate.interp1d
        Band we are converting to

    e_template : numpy array
        Template spectral energy density array for an Elliptical Galaxy, organized as follows:
        e_template[:,0] = wavelengths, [Angstrom]
        e_template[:,1] = flux density, [erg/s/cm^2/Angstrom]

    starb_template : numpy array
        Template spectral energy density array for a Starburst Galaxy, organized as follows:
        starb_template[:,0] = wavelengths, [Angstrom]
        starb_template[:,1] = flux density, [erg/s/cm^2/Angstrom]

    Returns:
        ST Magnitude
    """

    # Elliptical Spectrum Template with redshift
    e_wavelength = e_template[:,0] * (1+z)
    e_flux = e_template[:,1]

    # Starburst Spectrum Template with redshift
    starb_wavelength = starb_template[:,0] * (1+z)
    starb_flux = starb_template[:,1]

    # Get Modified Elliptical Spectrum for the given source
    non0_indices = np.nonzero(src_band(e_wavelength))
    lam = e_wavelength[non0_indices]
    f_lam = e_flux[non0_indices]
    r = src_band(lam)
    e_mag_exp = get_stmag(r, f_lam, lam)
    e_corr = get_correction(e_mag_exp, mag_obs)
    src_e_mag = get_stmag(r, f_lam, lam, correction=e_corr)



    # Get Modified Starburst Spectrum
    non0_indices = np.nonzero(src_band(starb_wavelength))
    lam = starb_wavelength[non0_indices]
    f_lam = starb_flux[non0_indices]
    r = src_band(lam)
    starb_mag_exp = get_stmag(r, f_lam, lam)
    starb_corr = get_correction(starb_mag_exp, mag_obs)
    src_starb_mag = get_stmag(r, f_lam, lam, correction=starb_corr)


 

    # Get Elliptical mag for Target Band
    non0_indices = np.nonzero(target_band(e_wavelength))
    lam = e_wavelength[non0_indices]
    f_lam = e_flux[non0_indices]
    r = target_band(e_wavelength)[non0_indices] #unitless
    target_e_mag = get_stmag(r, f_lam, lam, correction=e_corr)

    # Get Starburst mag for Target Band
    non0_indices = np.nonzero(target_band(starb_wavelength))
    lam = starb_wavelength[non0_indices]
    f_lam = starb_flux[non0_indices] #erg s^-1 cm^-2
    r = target_band(starb_wavelength)[non0_indices] #unitless
    target_starb_mag = get_stmag(r, f_lam, lam, correction=starb_corr)


    return target_e_mag, target_starb_mag, src_e_mag, src_starb_mag

