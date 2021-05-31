__all__ = ['delta_bias', 'beta_uncertainty', 'beta_to_sigma_dm_bahamas', 'beta_to_sigma_dm_h14']

import numpy as np


def delta_bias(delta_i, delta_hst, n_hst, n_sb):
    """
    Calculates the expected positional uncertainty in the mass peak for SuperBIT observations

    Parameters
    ----------
    delta_i : float
        Positional uncertainty without shape bias

    delta_hst : float
        Positional uncertainty with HST shape bias

    n_hst : float
        Background galaxy density observed by HST [arcmin^-2]

    n_sb : float
        Background galaxy density observed by SuperBIT [arcmin^-2]

    Returns:
    ----------
    delta_sb : float
        Expected positional uncertainty with SuperBIT shape bias
        
    """
    # Convert bg galaxy densities to arcsec^-2
    n_hst = n_hst/(3600) #arcsec^-2
    n_sb = n_sb/(3600) #arcsec^-2

    # Calculate conversion constants
    a_hst = delta_hst / delta_i 
    a_0 = a_hst * np.sqrt(n_hst)
    a_sb = a_0 / np.sqrt(n_sb)
    
    delta_sb = a_sb * delta_i
    
    return delta_sb


def beta_uncertainty(d_beta_hst, delta_i, delta_hst, n_hst, n_sb):
    """
    Calculates the expected uncertainty in beta for SuperBIT observations

    Parameters
    ----------
    d_beta_hst : float
        Uncertainty in beta for HST
        
    delta_i : float
        Positional uncertainty without shape bias

    delta_hst : float
        Positional uncertainty with HST shape bias

    n_hst : float
        Background galaxy density observed by HST [arcmin^-2]

    n_sb : float
        Background galaxy density observed by SuperBIT [arcmin^-2]

    Returns:
    ----------
    d_beta_sb : float
        Expected beta uncertainty for SuperBIT
        
    """
    # Convert bg galaxy densities to arcsec^-2
    n_hst = n_hst/(3600) #arcsec^-2
    n_sb = n_sb/(3600) #arcsec^-2

    # Calculate conversion constants
    a_hst = delta_hst / delta_i 
    a_0 = a_hst * np.sqrt(n_hst)
    a_sb = a_0 / np.sqrt(n_sb)
    
    # Convert Harvey 14 Results to SuperBIT
    d_beta = d_beta_hst / a_hst    
    d_beta_sb = d_beta * a_sb

    return d_beta_sb


def beta_to_sigma_dm_bahamas(d_beta_hst, d_beta_sb):
    """
    Calculates SIDM cross-section (sigma) uncertainty from beta uncertainty, assuming the beta-sigma relation from BAHAMAS simulation

    Parameters
    ----------
    d_beta_hst : float
        Uncertainty in beta for HST
        
    d_beta_sb : float
        Uncertainty in beta for SuperBIT

    Returns:
    ----------
    d_dm_hst : float
        SIDM cross-section uncertainty for HST (should match the results from the BAHAMAS simulation)
    
    d_dm_sb : float
        Expected SIDM cross-section uncertainty for SuperBIT
        
    """
    
    # Values of constants from BAHAMAS Model
    A = 0.114
    sigma_star = 0.459

    # Calculate uncertainty in sigma_dm using the BAHAMAS model
    d_dm_hst = d_beta_hst * sigma_star / A # HST, BAHAMAS sim fit model
    d_dm_sb = d_beta_sb * sigma_star / A # SuperBIT, BAHAMAS model
    
    return d_dm_hst, d_dm_sb


def beta_to_sigma_dm_h14(d_beta_hst, d_beta_sb, d_dm_hst=1.075):
    """
    Calculates SIDM cross-section (sigma) uncertainty from beta uncertainty, assuming the beta-sigma relation from the Harvey 14 paper

    Parameters
    ----------
    d_beta_hst : float
        Uncertainty in beta for HST
        
    d_beta_sb : float
        Uncertainty in beta for SuperBIT
        
    d_dm_hst : float
        Uncertainty in sigma for HST SuperBIT

    Returns:
    ----------
    d_dm_hst : float
        SIDM cross-section uncertainty for HST 
        
    d_dm_sb : float
        Expected SIDM cross-section uncertainty for SuperBIT
        
    """    
    # Calculate uncertainty in sigma_dm using the H14 model
    h14_const = d_beta_hst / d_dm_hst
    d_dm_sb = d_beta_sb / h14_const
    
    return d_dm_hst, d_dm_sb