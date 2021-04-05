__all__ = ['delta_bias', 'beta_uncertainty', 'beta_to_sigma_dm_bahamas', 'beta_to_sigma_dm_h14']

import numpy as np


def delta_bias(delta_i, delta_hst, n_hst, n_sb):
    
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
    # Convert bg galaxy densities to arcsec^-2
    n_hst = n_hst/(3600) #arcsec^-2
    n_sb = n_sb/(3600) #arcsec^-2

    # Calculate conversion constants
    a_hst = delta_hst / delta_i 
    a_0 = a_hst * np.sqrt(n_hst)
    a_sb = a_0 / np.sqrt(n_sb)
    # Harvey 14 Results
    d_beta = d_beta_hst / a_hst
    d_beta_sb = d_beta * a_sb

    return d_beta_sb


def beta_to_sigma_dm_bahamas(d_beta_hst, d_beta_sb):
    # Values of constants from BAHAMAS Model
    A = 0.114
    sigma_star = 0.459

    # Calculate uncertainty in sigma_dm using the BAHAMAS model
    d_dm_hst = d_beta_hst * sigma_star / A # HST, BAHAMAS sim fit model
    d_dm_sb = d_beta_sb * sigma_star / A # SuperBIT, BAHAMAS model
    
    return d_dm_hst, d_dm_sb


def beta_to_sigma_dm_h14(d_beta_hst, d_beta_sb, d_dm_hst=1.075):
    
    # Calculate uncertainty in sigma_dm using the H14 model
    h14_const = d_beta_hst / d_dm_hst
    d_dm_sb = d_beta_sb / h14_const
    
    return d_dm_hst, d_dm_sb