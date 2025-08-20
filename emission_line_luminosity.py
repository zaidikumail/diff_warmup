import dsps
from dsps import load_ssp_templates
from dsps.cosmology import age_at_z, DEFAULT_COSMOLOGY
from dsps import calc_rest_sed_sfh_table_lognormal_mdf
from dsps import calc_rest_sed_sfh_table_met_table
from dsps.utils import trapz
import numpy as np
from matplotlib import pyplot as plt

from jax import jit as jjit
from jax import grad
import jax.numpy as jnp

from tqdm.autonotebook import tqdm
import astropy.constants as const
from astropy.constants import L_sun
import astropy.units as u
import numpy.ma as ma


def _get_clipped_sed(wave, sed, wave_lo, wave_hi):
	sel = (wave > wave_lo) & (wave < wave_hi)
	wave_clipped = wave[sel]
	sed_clipped  = sed[sel]

	return wave_clipped, sed_clipped


def _get_masked_sed(continuum_wave, continuum_rest_sed, lo, hi):

	#mask to fit continuum
	mask = (continuum_wave >= lo) & (continuum_wave <= hi)
	continuum_wave_masked = ma.array(continuum_wave, mask = mask).compressed()
	continuum_rest_sed_masked = ma.array(continuum_rest_sed, mask = mask).compressed()

	return mask, continuum_wave_masked, continuum_rest_sed_masked


def _quad_continuum_model(theta, wave):
    c0 = theta["c0"]
    c1 = theta["c1"]
    c2 = theta["c2"]

    return (c2*wave*wave) + (c1*wave) + c0



def _mse(flux_true: jnp.ndarray, flux_pred: jnp.ndarray) -> jnp.float64:
    """Mean squared error function."""
    return jnp.mean(jnp.power(flux_true - flux_pred, 2))



def _mseloss(theta, model, wave, flux_true):
	flux_pred = model(theta, wave)
	return _mse(flux_true, flux_pred)



def _model_optimization_loop(theta, model, loss, wave, flux_true, n_steps=1000, step_size=1e-18):
    dloss = grad(loss)

    #initial continuum_rest_sed
    continuum_rest_sed_initial = model(dict(c0=theta["c0"], c1=theta["c1"], c2=theta["c2"]), wave)

    
    losses = []
    for i in tqdm(range(n_steps)):
        
        grads = dloss(dict(c0=theta["c0"], c1=theta["c1"], c2=theta["c2"]), model, wave, flux_true)
        
        theta["c0"] = theta["c0"] - step_size*grads["c0"]
        theta["c1"] = theta["c1"] - step_size*grads["c1"]
        theta["c2"] = theta["c2"] - step_size*grads["c2"]

        losses.append(loss(dict(c0=theta["c0"], c1=theta["c1"], c2=theta["c2"]), model, wave, flux_true))

    #fitted continuum_rest_sed
    continuum_rest_sed_fit = model(dict(c0=theta["c0"], c1=theta["c1"], c2=theta["c2"]), wave)

    return losses, theta, continuum_rest_sed_initial, continuum_rest_sed_fit


def _get_line_sed(wave, sed, continuum_sed, line_mask):
	"""
	Parameters:
		wave: wavelength in Angstroms
		sed: SED in units of Lsun/Hz/Msun
		continuum_sed: rest-frame continuum fitted sed in units of Lsun/Hz/Msun
		line_mask: boolean mask array to select wave spanning line

	Returns:
		line_wave: wavelength in Angstroms spanning line
		line_sed: sed [Lsun/Hz/Msun] - continuum_sed [Lsun/Hz/Msun]
	"""
	
	line_wave = wave[line_mask]
	line_sed = sed[line_mask] - continuum_sed[line_mask]
	#line_rest_sed_Lnu = line_rest_sed * L_sun.cgs.value #convert from Lsun/Hz --> erg/s/Hz (Lnu)

	return line_wave, line_sed


def _get_integrated_luminosity(wave, sed):
	"""
	Parameters:
		wave - wavelength array in units of Angstrom
		sed - [Lsun/Hz/Msun]

	Returns: 
		integrated_luminosity - integrated sed in units of [Lsun/Msun]

	"""
	freq = const.c.value / (wave*1e-10)
	freq = jnp.flip(freq)

	integrated_luminosity = np.trapezoid(sed, freq)	#[Lsun/Msun]

	return integrated_luminosity





# def _get_Fnu_from_Lnu(rest_sed_Lnu):
# 	#Lnu (rest-frame) to Fnu (absolute) in units of [erg/s/Hz/cm^2]
# 	D_pc = 10 * u.pc
# 	D_cm = D_pc.to(u.cm).value
# 	rest_sed_Fnu = rest_sed_Lnu / (4 * np.pi * (D_cm**2)) #[erg/s/Hz/cm^2]

# 	return rest_sed_Fnu


# def _get_integrated_Flux(wave_AA, rest_sed_Fnu):
# 	"""
# 	Parameters:
# 		wave_AA - wavelength array in units of Angstrom
# 		rest_sed_Fnu - Fnu [erg/s/Hz/cm^2]

# 	Returns: 
# 		integrated_rest_F - integrated Fnu in units of [erg/s/cm^2]

# 	"""
# 	freq_Hz = const.c.value / (wave_AA*1e-10)
# 	freq_Hz = jnp.flip(freq_Hz)

# 	#integrated F
# 	integrated_rest_F = np.trapezoid(rest_sed_Fnu, freq_Hz)	#[erg/s/cm^2]

# 	return integrated_rest_F



def get_emission_line_luminosity(
	wave,
	sed,
	continuum_fit_lo_lo, 
	continuum_fit_lo_hi, 
	continuum_fit_hi_lo, 
	continuum_fit_hi_hi,
	line_lo,
	line_hi
	):

	"""
	Notes:
		"clipped" means sed clipped between continuum_fit_lo_lo and continuum_fit_hi_hi
		"masked", in additon to wavelengths dropped in "clipped", masks line wavelengths for continuum fitting
	"""
	wave_clipped, sed_clipped = _get_clipped_sed(wave, sed, continuum_fit_lo_lo, continuum_fit_hi_hi)

	#mask line wavelengths for continuum fitting
	mask, wave_masked, sed_masked = _get_masked_sed(wave_clipped, sed_clipped, continuum_fit_lo_hi, continuum_fit_hi_lo)

	#initialize qudratic coeeficients close to a flat line at the mean of continuum_rest_sed_masked
	c0_initial =  jnp.mean(sed_masked)
	c1_initial =  1e-18 	#very small
	c2_initial = -3e-18 	#very small

	losses, theta, continuum_sed_initial_masked, continuum_sed_fit_masked = _model_optimization_loop(dict(c0=c0_initial, c1=c1_initial, c2=c2_initial),
		_quad_continuum_model, _mseloss, wave_masked, sed_masked)

	#interpolate continuum rest sed in the line masked region
	continuum_sed_fit_clipped = jnp.interp(wave_clipped, wave_masked, continuum_sed_fit_masked)

	#limit to line wavelengths
	line_mask = (wave_clipped >= line_lo) & (wave_clipped <= line_hi)
	line_wave, line_sed = _get_line_sed(wave_clipped, sed_clipped, continuum_sed_fit_clipped, line_mask)

	line_integrated_luminosity = _get_integrated_luminosity(line_wave, line_sed)

	sed_dict = {"wave_masked" : wave_masked, 
				"continuum_sed_initial_masked" : continuum_sed_initial_masked,
				"continuum_sed_fit_masked" : continuum_sed_fit_masked,
				"line_wave" : line_wave,
				"line_sed": line_sed,
				"line_integrated_luminosity": line_integrated_luminosity
				}

	return losses, theta, sed_dict



# def get_emission_line_Flux(
# 	wave, 
# 	rest_sed, 
# 	continuum_fit_lo_lo, 
# 	continuum_fit_lo_hi, 
# 	continuum_fit_hi_lo, 
# 	continuum_fit_hi_hi, 
# 	line_lo, 
# 	line_center, 
# 	line_hi
# 	):

	
# 	continuum_wave, continuum_rest_sed = _get_clipped_sed(wave, rest_sed, continuum_fit_lo_lo, continuum_fit_hi_hi)


# 	mask, continuum_wave_masked, continuum_rest_sed_masked = _get_masked_sed(continuum_wave, continuum_rest_sed, continuum_fit_lo_hi, continuum_fit_hi_lo)


# 	#initialize qudratic coeeficients close to a flat line at the mean of continuum_rest_sed_masked
# 	c0_initial =  jnp.mean(continuum_rest_sed_masked)
# 	c1_initial =  1e-18 	#very small
# 	c2_initial = -3e-18 	#very small


# 	losses, theta, continuum_rest_sed_initial, continuum_rest_sed_fit = _model_optimization_loop(dict(c0=c0_initial, c1=c1_initial, c2=c2_initial),
# 		_quad_continuum_model, _mseloss, continuum_wave_masked, continuum_rest_sed_masked)

# 	#interpolate continuum rest sed in the line masked region
# 	continuum_rest_sed_fit_interp = jnp.interp(continuum_wave, continuum_wave_masked, continuum_rest_sed_fit)

# 	#limit to line wavelengths
# 	line_mask = (continuum_wave >= line_lo) & (continuum_wave <= line_hi)
# 	line_wave, line_rest_sed, line_rest_sed_Lnu = _get_line_rest_sed(continuum_wave, continuum_rest_sed, continuum_rest_sed_fit_interp, line_mask)

# 	integrated_line_rest_L = _get_integrated_L(line_wave, line_rest_sed_Lnu)

# 	line_rest_sed_Fnu = _get_Fnu_from_Lnu(line_rest_sed_Lnu)
# 	integrated_line_rest_F = _get_integrated_Flux(line_wave, line_rest_sed_Fnu)


# 	sed_dict = {"continuum_wave_masked" : continuum_wave_masked, 
# 				"continuum_rest_sed_initial" : continuum_rest_sed_initial,
# 				"continuum_rest_sed_fit" : continuum_rest_sed_fit,
# 				"line_wave" : line_wave,
# 				"line_rest_sed_Lnu": line_rest_sed_Lnu,
# 				"integrated_line_rest_L": integrated_line_rest_L,
# 				"line_rest_sed_Fnu" : line_rest_sed_Fnu,
# 				"integrated_line_rest_F" : integrated_line_rest_F
# 				}

# 	return losses, theta, sed_dict
