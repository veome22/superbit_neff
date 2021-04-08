__all__ = ['stmag_conversion', 'get_stmag', 'abmag_conversion', 'get_abmag', 'get_correction', 'redshift_sed']

import numpy as np
from astropy import units as u
from astropy import constants as const
import bit_tools as bt

# Get redshifted spectrum by redshifting in Integrated Flux Density space 
def redshift_sed(sed, z):
    """
    Returns the redshifted SED from a source spectrum

    Args:
        sed: Source Spectral Flux Density, [ergs/s/cm^2/Angstrom]
        z: Redshift

    Returns:
        Redshifted SED, [ergs/s/cm^2/Angstrom]
    """
    int_sed = np.copy(sed)
    
    # Compute Integrated SED (ergs cm^-2 s^-1)
    int_sed[:,1] = int_sed[:,0] * int_sed[:,1]
    
    # Redshift integrated SED
    int_sed[:,0] = int_sed[:,0]* (1+z)
    
    # Convert back to flux density units
    int_sed[:,1] = int_sed[:,1] / int_sed[:,0]
    
    return int_sed


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

# Get mag from Response Function, Spectrum, Lambda Range, and Scaling Correction to Spectrum
def get_abmag(r, f_lam, lam, l_pivot, correction=1.0):
    """
    Returns the observed AB mag for a given filter and source spectrum

    Args:
        r: Filter Response Function, [unitless]
        f_lam: Source Spectral Flux Density, [ergs/s/cm^2/Angstrom]
        lam: Wavelength range, [Angstrom]
        l_pivot: Pivot Wavelength of the Filter [Angstrom]
        correction: constant factor to scale f_lam depending on the source

    Returns:
        AB Magnitude
    """
    f_lam = f_lam * correction
    f1 = np.trapz(y=f_lam * r * lam, x=lam)
    f2 = np.trapz(y=r * lam, x=lam) 

    f_exp = f1/f2 # erg s^-1 cm^-2 AA^-1

    stmag = -2.5 * np.log10(f_exp) - 21.1
    abmag = bt.converters.magst_to_magab(stmag, l_pivot * u.AA.to(u.nm))
    return abmag

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
    ----------
    target_e_mag : float
        ST Mag observed in target band, if source is an Elliptical
    
    target_starb_mag
        ST Mag observed in target band, if source is a Starburst
        
    src_e_mag (not returned)
        ST Mag observed in source band, if source is an Elliptical. 
        Should be the same as mag_obs.
        
    src_starb_mag (not returned)
        ST Mag observed in source band, if source is a Starburst. 
        Should be the same as mag_obs.
        
    """

    # Get Elliptical Spectrum Template with redshift
    e_redshift = redshift_sed(e_template, z)
    e_wavelength = e_redshift[:,0]
    e_flux = e_redshift[:,1]
    

    # Get Starburst Spectrum Template with redshift
    starb_redshift = redshift_sed(starb_template, z)
    starb_wavelength = starb_redshift[:,0]
    starb_flux = starb_redshift[:,1]

    
    # Modify the Elliptical Spectrum to satisfy the Observed ST Mag.
    non0_indices = np.nonzero(src_band(e_wavelength))
    lam = e_wavelength[non0_indices]
    f_lam = e_flux[non0_indices]
    r = src_band(lam)
    e_mag_exp = get_stmag(r, f_lam, lam)
    e_corr = get_correction(e_mag_exp, mag_obs)
    src_e_mag = get_stmag(r, f_lam, lam, correction=e_corr)
    
    # Get observed Elliptical ST Mag in Target Band
    non0_indices = np.nonzero(target_band(e_wavelength))
    lam = e_wavelength[non0_indices]
    f_lam = e_flux[non0_indices]
    r = target_band(e_wavelength)[non0_indices]
    target_e_mag = get_stmag(r, f_lam, lam, correction=e_corr)
    
    

    # Modify the Starburst Spectrum to satisfy the Observed ST Mag.
    non0_indices = np.nonzero(src_band(starb_wavelength))
    lam = starb_wavelength[non0_indices]
    f_lam = starb_flux[non0_indices]
    r = src_band(lam)
    starb_mag_exp = get_stmag(r, f_lam, lam)
    starb_corr = get_correction(starb_mag_exp, mag_obs)
    src_starb_mag = get_stmag(r, f_lam, lam, correction=starb_corr)

    # Get observed Starburst ST Mag in Target Band
    non0_indices = np.nonzero(target_band(starb_wavelength))
    lam = starb_wavelength[non0_indices]
    f_lam = starb_flux[non0_indices]
    r = target_band(starb_wavelength)[non0_indices]
    target_starb_mag = get_stmag(r, f_lam, lam, correction=starb_corr)


    return target_e_mag, target_starb_mag#, src_e_mag, src_starb_mag



# Get SuperBIT magitudes from Observed Mag
def abmag_conversion(mag_obs, z, src_band, target_band, e_template, starb_template, src_pivot=1.0, target_pivot=1.0):
    """
    Converts AB Mags from one filter to another, assuming both Elliptical and Starburst 
    spectra for the source

    Parameters
    ----------
    mag_obs : float
        AB Mag observed by Source Band, [unitless]

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
    
    src_pivot : float
        Pivot Wavelength of Source Band [Angstrom]
        
    target_pivot : float
        Pivot Wavelength of Target Band [Angstrom]


    Returns:
    ----------
    target_e_mag : float
        AB Mag observed in target band, if source is an Elliptical
    
    target_starb_mag
        AB Mag observed in target band, if source is a Starburst
        
    src_e_mag (not returned)
        AB Mag observed in source band, if source is an Elliptical. 
        Should be the same as mag_obs.
        
    src_starb_mag (not returned)
        AB Mag observed in source band, if source is a Starburst. 
        Should be the same as mag_obs.
        
    """

    # Get Elliptical Spectrum Template with redshift
    e_redshift = redshift_sed(e_template, z)
    e_wavelength = e_redshift[:,0]
    e_flux = e_redshift[:,1]
    

    # Get Starburst Spectrum Template with redshift
    starb_redshift = redshift_sed(starb_template, z)
    starb_wavelength = starb_redshift[:,0]
    starb_flux = starb_redshift[:,1]

    
    # Modify the Elliptical Spectrum to satisfy the Observed AB Mag in Source Band.
    non0_indices = np.nonzero(src_band(e_wavelength))
    lam = e_wavelength[non0_indices]
    f_lam = e_flux[non0_indices]
    r = src_band(lam)
    e_mag_exp = get_abmag(r, f_lam, lam, src_pivot)
    e_corr = get_correction(e_mag_exp, mag_obs)
    src_e_mag = get_abmag(r, f_lam, lam, src_pivot, correction=e_corr)
    
    # Get observed Elliptical AB Mag in Target Band
    non0_indices = np.nonzero(target_band(e_wavelength))
    lam = e_wavelength[non0_indices]
    f_lam = e_flux[non0_indices]
    r = target_band(e_wavelength)[non0_indices]
    target_e_mag = get_abmag(r, f_lam, lam, target_pivot, correction=e_corr)
    
    

    # Modify the Starburst Spectrum to satisfy the Observed AB Mag.
    non0_indices = np.nonzero(src_band(starb_wavelength))
    lam = starb_wavelength[non0_indices]
    f_lam = starb_flux[non0_indices]
    r = src_band(lam)
    starb_mag_exp = get_abmag(r, f_lam, lam, src_pivot)
    starb_corr = get_correction(starb_mag_exp, mag_obs)
    src_starb_mag = get_abmag(r, f_lam, lam, src_pivot, correction=starb_corr)

    # Get observed Starburst AB Mag in Target Band
    non0_indices = np.nonzero(target_band(starb_wavelength))
    lam = starb_wavelength[non0_indices]
    f_lam = starb_flux[non0_indices]
    r = target_band(starb_wavelength)[non0_indices]
    target_starb_mag = get_abmag(r, f_lam, lam, target_pivot, correction=starb_corr)


    return target_e_mag, target_starb_mag#, src_e_mag, src_starb_mag

