import dsps
from dsps import load_ssp_templates
from dsps.cosmology import age_at_z, DEFAULT_COSMOLOGY
from dsps import calc_rest_sed_sfh_table_lognormal_mdf
from dsps import calc_rest_sed_sfh_table_met_table
from dsps import calc_ssp_weights_sfh_table_lognormal_mdf
from dsps.sed.stellar_age_weights import _calc_logsm_table_from_sfh_table
from dsps.utils import trapz
from dsps.constants import N_T_LGSM_INTEGRATION, SFR_MIN, T_BIRTH_MIN

from diffsky import diffndhist

import numpy as np
from matplotlib import pyplot as plt

from jax import jit as jjit
from jax import grad
from jax import vmap 
import jax.numpy as jnp
from jax import random

from tqdm.autonotebook import tqdm
import astropy.constants as const
from astropy.constants import L_sun
import astropy.units as u
import numpy.ma as ma
L_SUN_CGS = jnp.array(L_sun.cgs.value)

def get_L_halpha(gal_sfr_table, 
                 gal_lgmet, 
                 gal_lgmet_scatter, 
                 gal_t_table, 
                 ssp_lgmet, 
                 ssp_lg_age_gyr,
                 ssp_halpha_line_luminosity,
                 t_obs
                 ):

    weights, lgmet_weights, age_weights = calc_ssp_weights_sfh_table_lognormal_mdf(gal_t_table,
                                                                                   gal_sfr_table,
                                                                                   gal_lgmet,
                                                                                   gal_lgmet_scatter,
                                                                                   ssp_lgmet,
                                                                                   ssp_lg_age_gyr,
                                                                                   t_obs[0]
                                                                                  )
    #get mass
    lgt_obs = jnp.log10(t_obs)
    lgt_table = jnp.log10(gal_t_table)
    logsm_table = _calc_logsm_table_from_sfh_table(gal_t_table, gal_sfr_table, SFR_MIN)
    logsm_obs = jnp.interp(lgt_obs, lgt_table, logsm_table)
    mstar_obs = jnp.power(10, logsm_obs)

    #convert luminosity [Lsun/Msun] ---> [erg/s]
    L_halpha_Lsun_per_Msun = (ssp_halpha_line_luminosity * weights).sum()
    L_halpha_erg_per_sec = L_halpha_Lsun_per_Msun * (L_SUN_CGS * mstar_obs)
    

    return L_halpha_erg_per_sec, L_halpha_Lsun_per_Msun



get_L_halpha_vmap = jjit(vmap(
    get_L_halpha,
    in_axes=(0, None, None, None, None, None, None, None),
    out_axes=(0, 0)
    ))

def get_halpha_luminosity_func(L_halpha_cgs, sig=0.001, dlgL_bin=0.2, lgL_min=40., lgL_max=45.):
    lg_L_halpha_cgs = jnp.log10(L_halpha_cgs)
    
    sig = jnp.zeros_like(lg_L_halpha_cgs) + sig
    y = jnp.ones_like(lg_L_halpha_cgs)
    
    lgL_bin_edges = jnp.arange(lgL_min, lgL_max, dlgL_bin)
    lgL_bin_lo = lgL_bin_edges[:-1].reshape(lgL_bin_edges[:-1].size,1)
    lgL_bin_hi = lgL_bin_edges[1:].reshape(lgL_bin_edges[1:].size,1)
    
    tw_hist = diffndhist._tw_ndhist_vmap(lg_L_halpha_cgs, sig, lgL_bin_lo, lgL_bin_hi)
    tw_hist_weighted = diffndhist.tw_ndhist_weighted(lg_L_halpha_cgs, sig, y, lgL_bin_lo, lgL_bin_hi)

    return lgL_bin_edges, tw_hist_weighted

def pop_model(theta, 
              ssp_lgmet, 
              ssp_lg_age_gyr,
              ssp_halpha_line_luminosity,
              t_obs,
              lg_sfr_var=0.1, 
              N=10000, 
              gal_lgmet=-1.0, 
              gal_lgmet_scatter=0,
             ):
    lg_sfr_mean = theta["lg_sfr_mean"]
    key = random.PRNGKey(1000)
    lg_sfr_draws = lg_sfr_mean + jnp.sqrt(lg_sfr_var) * random.normal(key, shape=(N,))

    gal_t_table = jnp.linspace(0.05, 13.8, 100) # age of the universe in Gyr
    gal_sfr_tables = jnp.ones((gal_t_table.size, N)) * (10**lg_sfr_draws)# SFR in Msun/yr
    gal_sfr_tables = gal_sfr_tables.T

    L_halpha_cgs, L_halpha_unit = get_L_halpha_vmap(gal_sfr_tables, 
                                                    gal_lgmet, 
                                                    gal_lgmet_scatter, 
                                                    gal_t_table,
                                                    ssp_lgmet, 
                                                    ssp_lg_age_gyr,
                                                    ssp_halpha_line_luminosity, 
                                                    t_obs
                                                   )
    lgL_bin_edges , tw_hist_weighted = get_halpha_luminosity_func(L_halpha_cgs)

    return lgL_bin_edges, tw_hist_weighted, L_halpha_cgs

