import numpy as np
import pandas as pd
from scipy.special import gamma
import matplotlib.pyplot as plt
import batman
import spiderman as sp
import sys
from lmfit import Model
import astropy.constants as cs
import astropy.units as u
import RECTE
import recte
import emcee
from lmfit import Parameters, minimize
from lmfit.printfuncs import report_fit
from multiprocessing import Pool
from contextlib import closing

sys.path.insert(1, '/Users/bob/Documents/PhD/hstscan/hstscan')
import hstscan.reduction as r
from scipy.optimize import curve_fit

"""
This file contains functions that will help you fit light curves to HST data.
These functions include functions for:
- Divide out of transit/in eclipse method
- Fitting with an exponential ramp model
- Fitting with RECTE charge trap model.

Written by Bob Jacobs @Universiteit van Amsterdam
"""


def calc_WL(df_data, df_errors, Wavelengths, Weights=None, output="numpy", average=False, weighted=False):
    """
    Calculates the White Light light curve over the wavelength range 'Wavelengths' while weighing the data with
    'Weights'.

    :param df_data: a Pandas dataframe with the data
    :param df_errors: a Pandas dataframe with the errors on the data
    :param Wavelengths: numpy 1D-array with the Wavelengths over which to calculate the White light curve
    :param Weights: a numpy 1D array with which to weight the wavelengths. Should have same length as Wavelengths. The
                    mean of this array should be exactly 1.
                    default: None. If None, no weighting is done.
    :param output: string: ["numpy", "pandas"], determines the type of the output objects
    :param average: bool: whether to take the arithmic mean or not.
    :return: WL_LC: a numpy 1D array or Pandas dataframe with the White Light light curve. Its length is equal to the
                    length of the dataframes.
             WL_err: a numpy 1D array or Pandas dataframe with the errors on the white light curve
    """
    outputerror = "Please call this function with the output parameter as either 'numpy' or 'pandas'"
    assert output in ["numpy", "pandas"], outputerror

    if Weights is not None:
        if np.mean(Weights) != 1.:  # Mean of weights should be 1
            Weights = Weights * len(Weights) / np.sum(Weights)
        df_data_weighted = df_data[Wavelengths].multiply(Weights, axis=1)
        df_errors_weighted = df_errors[Wavelengths].multiply(Weights, axis=1)
        WL_LC = np.array(np.sum(df_data_weighted, axis=1))
        WL_err = np.sqrt(np.array(np.sum(df_errors_weighted * df_errors_weighted, axis=1)))
    elif average:
        WL_LC, WL_err = arithmic_mean(df_data[Wavelengths], df_errors[Wavelengths], axis=1, weighted=weighted)
    else:
        WL_LC = np.array(np.sum(df_data[Wavelengths], axis=1))
        WL_err = np.sqrt(np.array(np.sum(df_errors[Wavelengths] * df_errors[Wavelengths], axis=1)))

    if output == "numpy":
        pass
    elif output == "pandas":
        WL_LC = pd.DataFrame(data=WL_LC, columns=[np.mean(Wavelengths)])
        WL_err = pd.DataFrame(data=WL_err, columns=[np.mean(Wavelengths)])

    return WL_LC, WL_err

def arithmic_mean(A, Aerr, axis=0, weighted=True):
    """
    Calculates the weighted mean of array A
    :param A: numpy array with data
    :param Aerr: numpy array with errors
    :param axis: axis over which to calculate the mean. Default: 0, can also be 1
    :param weighted: bool: whether to weigh the data by the errors
    :return: A_average: numpy array with the (weighted) mean of A
             A_average_err: numpy array with the weighted average errors on A
    """
    if len(A.shape) < 2:
        if weighted:
            A_average = np.array(np.sum(A / (Aerr * Aerr)) / np.sum(1. / (Aerr * Aerr)))
        else:
            A_average = np.array(np.mean(A))
        A_average_err = np.sqrt(1. / np.sum(1. / (Aerr * Aerr)))
    else:
        if weighted:
            A_average = np.array(np.sum(A / (Aerr * Aerr), axis=axis) / np.sum(1. / (Aerr * Aerr), axis=axis))
        else:
            A_average = np.array(np.sum(A, axis=axis) / A.shape[axis])
        A_average_err = np.sqrt( list(1. / np.array(np.sum(1. / (Aerr * Aerr), axis=axis))) )
    return A_average, A_average_err



def divide_oot(df_data, df_err, divisor, divisor_err, selection, **kwargs):
    """
    Based on Berta et al. 2012

    Dividing the light curve by the weighted out of transit light curve (exposure per exposure) removes the orbit-long
    systematics associated with Hubble's orbit around the Earth.
    This function should be performed for every orbit

    :param df_data: a Pandas dataframe with the data to be divided
    :param df_err: a Pandas dataframe with the errors on that data
    :param divisor: a numpy array with the white light data to divide with
    :param divisor_err: a numpy array with the errors on the white light data to divide with.
    :param selection: a numpy 1D-array with the selection of indices to divide
    :param Wavelengths: numpy 1D-array with the Wavelengths over which to calculate the White light curve. Not required.
    :return: Doot: a Pandas dataframe similar to the input, restricted to "Wavelengths".
             Doot_err: a Pandas dataframe of errors similar to the input, restricted to "Wavelengths".
    """
    dataframeerror = "The first two input objects are not Pandas Dataframes but {} and {}".format(type(df_data), type(df_err))
    assert isinstance(df_data, pd.DataFrame) and isinstance(df_err, pd.DataFrame), dataframeerror
    Wavelengths = kwargs.get('Wavelengths', df_data.columns)

    data = np.array(df_data[Wavelengths].iloc[selection])
    data_err = np.array(df_err[Wavelengths].iloc[selection])
    length = len(selection)
    if len(divisor) < len(selection):
        extra = divisor[-1] * np.ones(len(selection) - len(divisor))
        extra_err = divisor_err[-1] * np.ones(len(selection) - len(divisor))
        divisor = np.concatenate((divisor, extra))
        divisor_err = np.concatenate((divisor_err, extra_err))
    else:
        divisor = divisor[:length]
        divisor_err = divisor_err[:length]

    # Doot_err = 1./divisor * sqrt( data_err**2 + (divisor_err * data/divisor)**2 )
    # Doot_err = 1./divisor * sqrt( data_err**2 + element**2 ) = Doot_err_sqrt / divisor
    if data.shape[1] == 1:  # df_data only contains one column
        divisor = np.array(divisor).ravel()
        divisor_err = np.array(divisor_err).ravel()
        Doot_data = data.ravel() / divisor
        Doot = pd.DataFrame(data=Doot_data, columns=Wavelengths)
        element = Doot_data * divisor_err
        Doot_err_sqrt = np.sqrt(data_err.ravel() ** 2. + element ** 2.)
        Doot_err = pd.DataFrame(data=(Doot_err_sqrt / divisor), columns=Wavelengths)
    else:
        Doot_data = np.divide(data.T, divisor).T
        Doot = pd.DataFrame(data=Doot_data, columns=Wavelengths)
        element = np.multiply(Doot_data.T, divisor_err).T
        Doot_err_sqrt = np.sqrt(data_err ** 2. + element ** 2.)
        Doot_err = pd.DataFrame(data=np.divide(Doot_err_sqrt.T, divisor).T, columns=Wavelengths)
    return Doot, Doot_err


def scale_LCs(df_data, df_err, Times, T_0, opt, cov):
    """
    After a fit on the transmission data with a linear decay model, we can remove the linear decay from the model by
    scaling the data appropriately.

    :param df_data: Pandas dataframe or numpy 1D array: The data to be scaled
    :param df_err: Pandas dataframe or numpy 1D array: The error on the data to be scaled
    :param Times: numpy 1D array: The array with all the times of the observation data
    :param T_0: float: the time at which the linear decay model has an intercept of exactly 1
    :param opt: list of floats: The optimal parameters obtained from a previous fit to the data.
                                The fourth entry needs to be the linear decay constant
                                The fifth entry needs to be the intercept of the linear decay model
    :param cov: 2D list of floats: The covariance matrix for the optimal parameters of opt.
    :return: df_data_scaled: Pandas dataframe or numpy 1D array: the scaled data. Type is the same as df_data
             df_data_scaled_err: Pandas dataframe or numpy 1D array: the error on the scaled data.
                                 Type is the same as df_data
    """

    opt_err = np.sqrt(np.diag(cov))
    if (type(df_data) == np.ndarray) & (df_data.ndim == 1):  # The data is in a numpy 1D array
        df_data_scaled = df_data * (1 + opt[3] * (Times - T_0)) - opt[4]
        element1 = df_err * (1 + opt[3] * (Times - T_0))
        element2 = df_data * (Times - T_0) * opt_err[3]
    else:  # The data is in pandas dataframe format
        df_data_scaled = df_data.multiply(1 + opt[3] * (Times - T_0), axis=0) - opt[4]
        element1 = df_err.multiply(1 + opt[3] * (Times - T_0), axis=0)
        element2 = df_data.multiply(Times - T_0, axis=0) * opt_err[3]
    #df_err_scaled = np.sqrt(element1 ** 2. + element2 ** 2.)
    df_err_scaled = np.sqrt(element1 ** 2. + element2 ** 2. + opt_err[4] ** 2.)
    return df_data_scaled, df_err_scaled


def normalize_LC(LC, error, selection):
    """
    Normalize a light curve w.r.t. the data points encompassed by 'selection'.
    This is done for scaling a light curve such that the mean of the out-of-transit values is 1.

    :param LC: a numpy 1D-array with the light curve data
    :param error: a numpy 1D-array with the errors on the light curve data
    :param selection: The indices of LC (and error) to which the light curve should be normalized
    :return: LC_norm: a numpy 1D array: the normalized light curve
             error_norm: a numpy 1D-array: the errors on the normalized light curve
    """
    Divisor = np.array(LC[selection])
    Divisor_err = np.array(error[selection])
    Divisor_mean = np.mean(Divisor)
    Divisor_mean_err = np.sqrt(np.sum(Divisor_err ** 2.)) / len(Divisor)
    LC_norm = LC / Divisor_mean
    error_norm = np.sqrt(error ** 2. + Divisor_mean_err ** 2. * LC_norm ** 2.) / Divisor_mean

    return LC_norm, error_norm


def select_oot(data, data_err, selections):
    """
    Select the observations that were taken out of transit using the "selections" parameter. Then we take the mean of
     this.
    This is evaluated at each exposure within an orbit.
    To illustrate:

    First element = np.mean(sec_orbit[0], thi_orbit[0], six_orbit[0])
    Second element = np.mean(sec_orbit[1], thi_orbit[1], six_orbit[1])
    ....
    ....
    nth element = np.mean(sec_orbit[n], thi_orbit[n]) #for n > len(six_orbit) + 2

    :param data: Pandas dataframe: The observed data
    :param data_err: Pandas dataframe: Error on the observed data
    :param selections: A list of 1D-numpy arrays: Each array contains the indices of the rows in "data" that correspond
                                                  to that orbit. The list is as long as the number of orbits.
    :return: data_oot_mean: a Pandas dataframe with the mean of the out-of-transit data
             data_oot_mean_err: a Pandas dataframe with the errors on that data
    """
    orbit_lengths = [len(s) for s in selections]
    maxlen = max(orbit_lengths)
    Selections = sorted(selections, key=len)
    Wavelengths = data.columns

    data_oot_mean = pd.DataFrame(data=np.zeros((maxlen, len(Wavelengths))), columns=Wavelengths)
    data_oot_mean_err = pd.DataFrame(data=np.zeros((maxlen, len(Wavelengths))), columns=Wavelengths)
    for l in Wavelengths:
        Selec = np.zeros((maxlen, len(Selections)))
        Selec_err = np.zeros((maxlen, len(Selections)))
        nr_orbits = np.zeros(maxlen)  # Nr. of orbits that "participate" in the mean
        for i in range(maxlen):
            for j, s in enumerate(Selections):
                if i < len(s):
                    Selec[i, j] = data[l].iloc[s[i]]
                    Selec_err[i, j] = data_err[l].iloc[s[i]]
                    nr_orbits[i] += 1
        Selec[Selec == 0] = np.nan
        Selec_err[Selec_err == 0] = np.nan
        data_oot_mean[l] = np.nanmean(Selec, axis=1)
        data_oot_mean_err[l] = np.sqrt(np.nansum(Selec_err ** 2.)) / nr_orbits

    return data_oot_mean, data_oot_mean_err


def reduced_chi2(xdata, ydata, errors, func, opt):
    """
    Calculate the reduced chi squared value

    :param xdata: list/1-D array: The x-data
    :param ydata: list/1-D array: The y-data
    :param errors: list/1-D array: The errors on the y-data
    :param func: The function to which the data is fitted
    :param opt: The optimal parameters from the fit
    :return: float: chi squared value
    """
    residuals = func(xdata, *opt) - ydata
    N = len(ydata)
    n = len(opt)
    return np.sum(residuals ** 2. / errors ** 2.) / float(N - n)


def linear(xdata, m, b):
    """
    A simple linear function
    :param xdata: (numpy 1D array) data
    :param m: (float) slope
    :param b: (float) intercept
    :return:
    """
    return m * xdata + b


def Func_exp_eclipse(t, depth_f, depth_r, t0, inc, ecc, aRs, u1, u2, phi, V2_F, V2_R, f0_F, f0_R, Rorb1_F, Rorb1_R,
                     Rorb2_F, Rorb2_R, tau_F, tau_R, SatF, V1_F, V1_R, V3_F, V3_R, Forward, sp_params, orbit_times,
                     satellite_time, Include, noexcl, Stel_puls_phase=0., Stel_puls_amp=0., Harm_amp=0., exptime=90.,
                     incl_first_orbit=False, fit=True):
        """
        A function with an exponential ramp model for each orbit for a light curve with an eclipse.
        The first orbit should already have been discarded before applying this function.

        :param t: (np.array) The observation times of the exposures (BJD)
        :param depth_f: (float) Eclipse depth of forward scanned data
        :param depth_r: (float) Eclipse depth of reverse scanned data
        :param t0: (float) Mid-transit time offset (days)
        :param inc: (float) orbital inclination in degrees
        :param ecc: (float) orbital eccentricity
        :param aRs: (float) semi-major axis in stellar radii
        :param u1: (float) first limb darkening parameter (if u2==0, limb darkening law is assumed to be linear)
        :param u2: (float) second limb darkening parameter for a quadratic limb darkening law
        :param phi: (float) Phase offset of the thermal emission peak in days
        :param V2_F: (float) HST systematics linear ramp for forward scanned data
        :param V2_R: (float) HST systematics linear ramp for reverse scanned data
        :param f0_F: (float) Stellar flux in forward scan (mean is not necessarily 1)
        :param f0_R: (float) Stellar flux in reverse scan
        :param Rorb1_F: (float) Forward scanned amplitude of the ramp of the first (non-discarded) orbit.
        :param Rorb1_R: (float) Reverse scanned amplitude of the ramp of the first orbit.
        :param Rorb2_F: (float) Forward scanned amplitude of the ramp of all remaining orbits.
        :param Rorb2_R: (float) Reverse scanned amplitude of the ramp of all remaining orbits.
        :param tau_F: (float) Forward scanned time scale of the ramps
        :param tau_R: (float) Reverse scanned time scale of the ramps
        :param SatF: (float) Flux added to a single exposure by a satellite crossing
        :param V1_F: (float) Sinoid amplitude in forward scanned data
        :param V1_R: (float) Sinoid amplitude in reverse scanned data
        :param V3_F: (float) Amplitude of second order polynomial baseline HST systematics in forward scanned data
        :param V3_R: (float) Amplitude of second order polynomial baseline HST systematics in reverse scanned data
        :param Forward: (numpy boolarray) Array of True/False on whether an exposure is Forward scanned.
                        This has the same length as 't'
        :param sp_params: (object) An instance of the spiderman ModelParams function
        :param orbit_times: (2D list) List of time ranges (in days) of the orbits such that all start times of an orbit
                           fall into this range. 2 orbit example: orbit_times = [[-1.e-6, 0.066], [0.066, 0.132]]
                           This list should include *all* orbit, even the discarded first orbit.
        :param satellite_time: the time of the exposure in which a satellite crossing occurs. This time should be in 't'
        :param Include: (numpy boolarray) Array of True/False on whether an exposure is included in the fitting
                        procedure. One may want to exclude e.g. egress when fitting. This only happens if noexcl
                        is set to False
        :param noexcl: (bool) Whether to exclude some data in the return function. Data that is included/excluded is
                        managed by the 'Include' input parameter
        :param Stel_puls_phase: (float) The phase of the stellar pulsations
        :param Stel_puls_amp: (float) Amplitude of stellar pulsations relative to the baked-in values in Stellar_signal
        :param Harm_amp: (float) Amplitude of first and secondary harmonics relative to the baked-in values in
                         Stellar_signal
        :param exptime: (float) Exposure time in seconds
        :param incl_first_orbit: (bool) Whether the first orbit of the visit is included in the data. (default=False)
        :param fit: Whether the call to this function is used to fit data. (else it's assumed to plot the data)

        :return:
               if fit == True:
                   if noexcl == False:
                       numpy array of length (nr. True in Include) with full astrophysical lightcurve (exlcudes
                       telescope systematics)
                   else:
                       numpy array of length t with full astrophysical lightcurve
               else:
                   4 arrays:
                   - numpy array of length t with full astrophysical lightcurve (excludes telescope systematics)
                   - numpy array of length t with stellar lightcurve
                   - numpy array of length t with astrophysical lightcurve * telescope systematics (excluding orbit ramps)
                     (i.e. what is observed)
                   - numpy array of length t with only planetary light curve
        """
        depth_ = [depth_f, depth_r]
        V1_ = [V1_F, V1_R]
        V2_ = [V2_F, V2_R]
        V3_ = [V3_F, V3_R]
        f0_ = [f0_F, f0_R]
        Rorb1_ = [Rorb1_F, Rorb1_R]
        Rorb2_ = [Rorb2_F, Rorb2_R]
        tau_ = [tau_F, tau_R]
        Direction_ = [Forward, ~Forward]
        t = np.array(t)

        if incl_first_orbit:
            k = 0
        else:
            k = 1

        sp_params.inc = inc
        sp_params.ecc = ecc
        sp_params.a = aRs
        if u2 == 0.:
            sp_params.limb_dark = "linear"  # limb darkening model #don't care
            sp_params.u = [u1]  # stellar limb darkening coefficients
        else:
            sp_params.limb_dark = "quadratic"  # limb darkening model #don't care
            sp_params.u = [u1, u2]  # stellar limb darkening coefficients
        #Adjust the time of secondary eclipse to the new time offset
        m = batman.TransitModel(sp_params, t + t0)
        sp_params.t_secondary = m.get_t_secondary(sp_params)


        #Make boolean masks for the first orbit and all other orbits.
        orbits = []
        for ot in orbit_times:
            orbits.append((t >= ot[0]) & (t < ot[1]))
        Firstorbit = orbits[k]
        #if not fit:
        #    print t[Firstorbit]
        #    print Firstorbit
        Otherorbits = np.any(orbits[k+1:], axis=0)
        torb1 = t[Firstorbit] - t[Firstorbit][0]
        torb2 = []
        for orb in orbits[k+1:]:
            torb2.extend(t[orb] - t[orb][0])
        torb2 = np.array(torb2)

        psi_s = np.zeros_like(t)
        S = np.zeros_like(t)
        f_m = np.zeros_like(t, dtype='complex128')  #complex, such that we're able to do complex calculations which can
                                                    #help finding the global minimum in some fitting methods.
        baseline_w_eclipse = np.zeros_like(t)



        for depth, V1, V2, V3, f0, Rorb1, Rorb2, tau, Direction in zip(depth_, V1_, V2_, V3_, f0_, Rorb1_, Rorb2_, tau_, Direction_):
            #Forward or reverse

            psi_p = Planetary_signal(t, phi, depth, V1, t0, sp_params, fit=fit)
            psi_s = Stellar_signal(t, Stel_puls_amp1, Stel_puls_phase1, Stel_puls_amp2, Stel_puls_phase2, Harm_amp, t0, sp_params)
            S[Firstorbit] = ((1 + V2 * (t[Firstorbit] - t[0]) + V3 * (t[Firstorbit] - t[0])**2.) *
                             (1 - Rorb1 * np.exp(-torb1 / tau)))  #Systematics
            S[Otherorbits] = ((1 + V2 * (t[Otherorbits] - t[0]) + V3 * (t[Otherorbits] - t[0])**2.) *
                              (1 - Rorb2 * np.exp(-torb2 / tau)))  #Systematics

            f_m[Direction] = f0 * S[Direction] * (psi_s[Direction] + psi_p[Direction])
            baseline_w_eclipse[Direction] = ((1 + V2 * (t[Direction] - t[0]) + V3 * (t[Direction] - t[0])**2.) *
                                             (psi_s[Direction] + psi_p[Direction]))

            satellite = t == satellite_time
            f_m[satellite] += SatF

        if fit:
            if noexcl:
                return np.real(f_m)
            else:
                return np.real(f_m[Include])
        else:
            if np.sum(Forward) != 0.:
                depth = depth_f
                V1 = V1_F
            else:
                depth = depth_r
                V1 = V1_R
            entire_phase = 1 + Planetary_signal(t, phi, depth, V1, t0, sp_params)
            return np.real(f_m), np.real(psi_s), np.real(baseline_w_eclipse), np.real(entire_phase)


def Func_exp_transit(t, depth_f, depth_r, t0, inc, ecc, aRs, u1, u2, phi, V2_F, V2_R, f0_F, f0_R, Rorb1_F, Rorb1_R,
                     Rorb2_F, Rorb2_R, tau_F, tau_R, SatF, V1_F, V1_R, V3_F, V3_R, Forward, sp_params, orbit_times,
                     satellite_time, Include, noexcl, Stel_puls_phase=0., Stel_puls_amp=0., Harm_amp=0., exptime=90.,
                     incl_first_orbit=False, fit=True):
        """
        A function with an exponential ramp model for each orbit for a transit light curve.
        The first orbit should already have been discarded before applying this function.

        :param t: (np.array) The observation times of the exposures (BJD)
        :param depth_f: (float) Eclipse depth of forward scanned data
        :param depth_r: (float) Eclipse depth of reverse scanned data
        :param t0: (float) Mid-transit time offset (days)
        :param inc: (float) orbital inclination in degrees
        :param ecc: (float) orbital eccentricity
        :param aRs: (float) semi-major axis in stellar radii
        :param u1: (float) first limb darkening parameter (if u2==0, limb darkening law is assumed to be linear)
        :param u2: (float) second limb darkening parameter for a quadratic limb darkening law
        :param phi: (float) Phase offset of the thermal emission peak in days
        :param V2_F: (float) HST systematics linear ramp for forward scanned data
        :param V2_R: (float) HST systematics linear ramp for reverse scanned data
        :param f0_F: (float) Stellar flux in forward scan (mean is not necessarily 1)
        :param f0_R: (float) Stellar flux in reverse scan
        :param Rorb1_F: (float) Forward scanned amplitude of the ramp of the first (non-discarded) orbit.
        :param Rorb1_R: (float) Reverse scanned amplitude of the ramp of the first orbit.
        :param Rorb2_F: (float) Forward scanned amplitude of the ramp of all remaining orbits.
        :param Rorb2_R: (float) Reverse scanned amplitude of the ramp of all remaining orbits.
        :param tau_F: (float) Forward scanned time scale of the ramps
        :param tau_R: (float) Reverse scanned time scale of the ramps
        :param SatF: (float) Flux added to a single exposure by a satellite crossing
        :param V1_F: (float) Sinoid amplitude in forward scanned data
        :param V1_R: (float) Sinoid amplitude in reverse scanned data
        :param V3_F: (float) Amplitude of second order polynomial baseline HST systematics in forward scanned data
        :param V3_R: (float) Amplitude of second order polynomial baseline HST systematics in reverse scanned data
        :param Forward: (numpy boolarray) Array of True/False on whether an exposure is Forward scanned.
                        This has the same length as 't'
        :param sp_params: (object) An instance of the spiderman ModelParams function
        :param orbit_times: (2D list) List of time ranges (in days) of the orbits such that all start times of an orbit
                           fall into this range. 2 orbit example: orbit_times = [[-1.e-6, 0.066], [0.066, 0.132]]
                           This list should include *all* orbit, even the discarded first orbit.
        :param satellite_time: the time of the exposure in which a satellite crossing occurs. This time should be in 't'
        :param Include: (numpy boolarray) Array of True/False on whether an exposure is included in the fitting
                        procedure. One may want to exclude e.g. egress when fitting. This only happens if noexcl
                        is set to False
        :param noexcl: (bool) Whether to exclude some data in the return function. Data that is included/excluded is
                        managed by the 'Include' input parameter
        :param Stel_puls_phase: (float) The phase of the stellar pulsations
        :param Stel_puls_amp: (float) Amplitude of stellar pulsations relative to the baked-in values in Stellar_signal
        :param Harm_amp: (float) Amplitude of first and secondary harmonics relative to the baked-in values in
                         Stellar_signal
        :param exptime: (float) Exposure time in seconds
        :param incl_first_orbit: (bool) Whether the first orbit of the visit is included in the data. (default=False)
        :param fit: Whether the call to this function is used to fit data. (else it's assumed to plot the data)

        :return:
               if fit == True:
                   if noexcl == False:
                       numpy array of length (nr. True in Include) with full astrophysical lightcurve (exlcudes
                       telescope systematics)
                   else:
                       numpy array of length t with full astrophysical lightcurve
               else:
                   4 arrays:
                   - numpy array of length t with full astrophysical lightcurve (excludes telescope systematics)
                   - numpy array of length t with stellar lightcurve
                   - numpy array of length t with astrophysical lightcurve * telescope systematics (excluding orbit ramps)
                     (i.e. what is observed)
                   - numpy array of length t with only planetary light curve
        """
        if incl_first_orbit:
            k = 0
        else:
            k = 1

        depth_ = [depth_f, depth_r]
        V1_ = [V1_F, V1_R]
        V2_ = [V2_F, V2_R]
        V3_ = [V3_F, V3_R]
        f0_ = [f0_F, f0_R]
        Rorb1_ = [Rorb1_F, Rorb1_R]
        Rorb2_ = [Rorb2_F, Rorb2_R]
        tau_ = [tau_F, tau_R]
        Direction_ = [Forward, ~Forward]
        t = np.array(t)

        sp_params.inc = inc
        sp_params.ecc = ecc
        sp_params.a = aRs
        if sp_params.limb_dark == 'linear':
            sp_params.u = [u1]  # stellar limb darkening coefficients
        elif sp_params.limb_dark == "quadratic":
            sp_params.u = [u1, u2]  # stellar limb darkening coefficients
        else:
            pass


        #Make boolean masks for the first orbit and all other orbits.
        orbits = []
        for ot in orbit_times:
            orbits.append((t >= ot[0]) & (t < ot[1]))
        Firstorbit = orbits[k]
        Otherorbits = np.any(orbits[k+1:], axis=0)
        torb1 = t[Firstorbit] - t[Firstorbit][0]
        torb2 = []
        for orb in orbits[k+1:]:
            torb2.extend(t[orb] - t[orb][0])
        torb2 = np.array(torb2)

        psi_s = np.zeros_like(t)
        S = np.zeros_like(t)
        f_m = np.zeros_like(t, dtype='complex128')  #complex, such that we're able to do complex calculations which can
                                                    #help finding the global minimum in some fitting methods.
        baseline_w_transit = np.zeros_like(t)

        for depth, V1, V2, V3, f0, Rorb1, Rorb2, tau, Direction in zip(depth_, V1_, V2_, V3_, f0_, Rorb1_, Rorb2_, tau_, Direction_):
            #Forward or reverse
            sp_params.rp = np.sqrt(depth)

            m = batman.TransitModel(sp_params, t + t0)
            psi_p = m.light_curve(sp_params)
            psi_s = Stellar_signal(t, Stel_puls_amp, Stel_puls_phase, Harm_amp, t0, sp_params)
            S[Firstorbit] = ((1 + V2 * (t[Firstorbit] - t[0]) + V3 * (t[Firstorbit] - t[0])**2.) *
                             (1 - Rorb1 * np.exp(-torb1 / tau)))  #Systematics
            S[Otherorbits] = ((1 + V2 * (t[Otherorbits] - t[0]) + V3 * (t[Otherorbits] - t[0])**2.) *
                              (1 - Rorb2 * np.exp(-torb2 / tau)))  #Systematics

            f_m[Direction] = f0 * S[Direction] * (psi_s[Direction] * psi_p[Direction])
            baseline_w_transit[Direction] = ((1 + V2 * (t[Direction] - t[0]) + V3 * (t[Direction] - t[0])**2.) *
                                             (psi_s[Direction] * psi_p[Direction]))

            satellite = t == satellite_time
            f_m[satellite] += SatF

        if fit:
            if noexcl:
                return np.real(f_m)
            else:
                return np.real(f_m[Include])
        else:
            if np.sum(Forward) != 0.:
                sp_params.rp = np.sqrt(depth_f)
            else:
                sp_params.rp = np.sqrt(depth_r)
            m = batman.TransitModel(sp_params, t + t0)
            entire_phase = m.light_curve(sp_params)
            return np.real(f_m), np.real(psi_s), np.real(baseline_w_transit), np.real(entire_phase)


def Func_recte_eclipse(t, depth_f, depth_r, t0, inc, ecc, aRs, u1, u2, phi, V2_F, V2_R, f0_F, f0_R, E0_s, E0_f, Delta_Es,
              Delta_Ef, SatF, V1_F, V1_R, V3_F, V3_R, Forward, sp_params, orbit_times, satellite_time, Include,
              exptime, noexcl, Stel_puls_phase=0., Stel_puls_amp=0., Harm_amp=0., incl_first_orbit=False, fit=True):
        """

        :param t: The observation times of the exposures (BJD)
        :param depth_f: Eclipse depth of forward scanned data
        :param depth_r: Eclipse depth of reverse scanned data
        :param t0: Mid-transit time (days)
        :param inc: orbital inclination
        :param phi: Phase offset of the thermal emission peak in days
        :param V1_F: Sinoid amplitude in forward scanned data
        :param V1_R: Sinoid amplitude in reverse scanned data
        :param V2_F: HST systematics linear ramp for forward scanned data
        :param V2_R: HST systematics linear ramp for reverse scanned data
        :param f0_F: Stellar flux in forward scan
        :param f0_R: Stellar flux in reverse scan
        :param E0_s: Trapped slow electrons before observations
        :param E0_f: Trapped fast electrons before observations
        :param Delta_Es: Change in slow trapped electrons between orbits
        :param Delta_Ef: Change in fast trapped electrons between orbits
        :param SatF: Flux induced by the satellite crossing
        :param fit: Whether the call to this function is used to fit the data.
        :return:
        """
        depth_ = [depth_f, depth_r * f0_R / f0_F]
        V1_ = [V1_F, V1_R]
        V2_ = [V2_F, V2_R]
        V3_ = [V3_F, V3_R]
        f0_ = [f0_F, f0_F]
        Direction_ = [Forward, ~Forward]
        t = np.array(t)

        sp_params.inc = inc
        sp_params.ecc = ecc
        sp_params.a = aRs
        if u2 == 0.:
            sp_params.limb_dark = "linear"  # limb darkening model #don't care
            sp_params.u = [u1]  # stellar limb darkening coefficients
        else:
            sp_params.limb_dark = "quadratic"  # limb darkening model #don't care
            sp_params.u = [u1, u2]  # stellar limb darkening coefficients
        # Adjust the time of secondary eclipse to the new time offset
        m = batman.TransitModel(sp_params, t + t0)
        sp_params.t_secondary = m.get_t_secondary(sp_params)


        psi_s = np.zeros_like(t)
        baseline_w_eclipse = np.zeros_like(t, dtype='complex128')


        for depth, V1, V2, V3, f0, Direction in zip(depth_, V1_, V2_, V3_, f0_, Direction_):
            #Forward or reverse
            t_secondary = sp_params.t_secondary.copy() + t0
            #print sp_params.t_secondary, t0
            if not t[0] < t_secondary < t[-1]:
                if t[0] < t_secondary + sp_params.per < t[-1]:
                    t_secondary += sp_params.per
                elif t[0] < t_secondary + 2 * sp_params.per < t[-1]:
                    t_secondary += 2 * sp_params.per
                elif t[0] < t_secondary - sp_params.per < t[-1]:
                    t_secondary -= sp_params.per
                elif t[0] < t_secondary - 2 * sp_params.per < t[-1]:
                    t_secondary -= 2 * sp_params.per
            #print V1, V1 * (24e-6) * (t[0] - t_secondary), t_secondary

            psi_p = Planetary_signal(t, phi, depth, V1, t0, sp_params, fit=fit)
            psi_s = Stellar_signal(t, Stel_puls_amp, Stel_puls_phase, Harm_amp, t0, sp_params)
            S = 1 + V2 * (t - t[0]) + V3 * (t - t[0])**2.  #Systematics
            baseline_w_eclipse[Direction] = f0 * S[Direction] * (psi_s[Direction] + psi_p[Direction])

            satellite = t == satellite_time
            baseline_w_eclipse[satellite] += SatF

        f_m = RECTE.RECTE(baseline_w_eclipse, (t - t[0]) * u.d.to(u.s), exptime = exptime,
                          trap_pop_s=E0_s, trap_pop_f=E0_f, dTrap_s=Delta_Es, dTrap_f=Delta_Ef)
        f_m[~Forward] = f_m[~Forward] * f0_R / f0_F
        if fit:
            if noexcl:
                return f_m / exptime
            else:
                return f_m[Include] / exptime
        else:
            if np.sum(Forward) != 0.:
                depth = depth_f
                V1 = V1_F
            else:
                depth = depth_r
                V1 = V1_R
            entire_phase = 1 + Planetary_signal(t, phi, depth, V1, t0, sp_params)
            return np.real(f_m / exptime), np.real(psi_s), np.real(baseline_w_eclipse), np.real(entire_phase)


def Func_recte_transit(t, depth_f, depth_r, t0, inc, ecc, aRs, u1, u2, phi, V2_F, V2_R, f0_F, f0_R, E0_s, E0_f, Delta_Es,
              Delta_Ef, SatF, V1_F, V1_R, V3_F, V3_R, Forward, sp_params, orbit_times, satellite_time, Include,
              exptime, noexcl, Stel_puls_phase=0., Stel_puls_amp=0., Harm_amp=0., incl_first_orbit=False, fit=True):
        """

        :param t: The observation times of the exposures (BJD)
        :param depth_f: Eclipse depth of forward scanned data
        :param depth_r: Eclipse depth of reverse scanned data
        :param t0: Mid-transit time offset (days)
        :param inc: orbital inclination
        :param phi: Phase offset of the thermal emission peak in days
        :param V1_F: Sinoid amplitude in forward scanned data
        :param V1_R: Sinoid amplitude in reverse scanned data
        :param V2_F: HST systematics linear ramp for forward scanned data
        :param V2_R: HST systematics linear ramp for reverse scanned data
        :param f0_F: Stellar flux in forward scan
        :param f0_R: Stellar flux in reverse scan
        :param E0_s: Trapped slow electrons before observations
        :param E0_f: Trapped fast electrons before observations
        :param Delta_Es: Change in slow trapped electrons between orbits
        :param Delta_Ef: Change in fast trapped electrons between orbits
        :param SatF: Flux induced by the satellite crossing
        :param fit: Whether the call to this function is used to fit the data.
        :return:
        """
        depth_ = [depth_f, depth_r * f0_R / f0_F]
        V1_ = [V1_F, V1_R]
        V2_ = [V2_F, V2_R]
        V3_ = [V3_F, V3_R]
        f0_ = [f0_F, f0_F]
        Direction_ = [Forward, ~Forward]
        t = np.array(t)

        sp_params.inc = inc
        sp_params.ecc = ecc
        sp_params.a = aRs
        if u2 == 0.:
            sp_params.limb_dark = "linear"  # limb darkening model #don't care
            sp_params.u = [u1]  # stellar limb darkening coefficients
        else:
            sp_params.limb_dark = "quadratic"  # limb darkening model #don't care
            sp_params.u = [u1, u2]  # stellar limb darkening coefficients
        # Adjust the time of secondary eclipse to the new time offset


        psi_s = np.zeros_like(t)
        baseline_w_eclipse = np.zeros_like(t, dtype='complex128')


        for depth, V1, V2, V3, f0, Direction in zip(depth_, V1_, V2_, V3_, f0_, Direction_):
            #Forward or reverse
            sp_params.rp = np.sqrt(depth)

            m = batman.TransitModel(sp_params, t + t0)
            psi_p = m.light_curve(sp_params)

            psi_s = Stellar_signal(t, Stel_puls_amp, Stel_puls_phase, Harm_amp, t0, sp_params)
            S = 1 + V2 * (t - t[0]) + V3 * (t - t[0])**2.  #Systematics
            baseline_w_eclipse[Direction] = f0 * S[Direction] * (psi_s[Direction] * psi_p[Direction])

            satellite = t == satellite_time
            baseline_w_eclipse[satellite] += SatF

        f_m = RECTE.RECTE(baseline_w_eclipse, (t - t[0]) * u.d.to(u.s), exptime = exptime,
                          trap_pop_s=E0_s, trap_pop_f=E0_f, dTrap_s=Delta_Es, dTrap_f=Delta_Ef)
        f_m[~Forward] = f_m[~Forward] * f0_R / f0_F
        if fit:
            if noexcl:
                return f_m / exptime
            else:
                return f_m[Include] / exptime
        else:
            if np.sum(Forward) != 0.:
                sp_params.rp = np.sqrt(depth_f)
            else:
                sp_params.rp = np.sqrt(depth_r)
            m = batman.TransitModel(sp_params, t + t0)
            entire_phase = m.light_curve(sp_params)
            return np.real(f_m / exptime), np.real(psi_s), np.real(baseline_w_eclipse), np.real(entire_phase)


def calc_eclipse_times(sp_params):
    """
    Calculates the ingress and egress times of eclipse.
    Is only valid for NON-ELLIPTIC ORIBTS
    Follows https://pdfs.semanticscholar.org/4572/1b4859970b57496d036396c1b4abe889d4d7.pdf
    :param sp_params: (object) An instance of the spiderman ModelParams function
    :return: - (float) time of end of eclipse (and start of egress) (BJD)
             - (float) time of start of eclipse (and end of ingress) (BJD)
             - (float) time of end of egress (BJD)
             - (float) time of start of ingress (BJD)
    """
    b = (sp_params.a * np.cos(sp_params.inc * np.pi / 180.) * (1 - sp_params.ecc ** 2.) /
         (1 - sp_params.ecc * np.sin(sp_params.w * np.pi / 180.)))
    DT_eclipse_tot = sp_params.per / np.pi * np.arcsin(np.sqrt((1 + sp_params.rp) ** 2. - b ** 2.) /
                                                        (np.sin(sp_params.inc * np.pi / 180.) * sp_params.a))
    DT_eclispe_full = sp_params.per / np.pi * np.arcsin(np.sqrt((1 - sp_params.rp) ** 2. - b ** 2.) /
                                                            (np.sin(sp_params.inc * np.pi / 180.) * sp_params.a))
    t_ecl_e = sp_params.t_secondary + 0.5 * DT_eclispe_full
    t_ecl_s = sp_params.t_secondary - 0.5 * DT_eclispe_full
    t_egr_e = sp_params.t_secondary + 0.5 * DT_eclipse_tot
    t_ing_s = sp_params.t_secondary - 0.5 * DT_eclipse_tot
    return t_ecl_e, t_ecl_s, t_egr_e, t_ing_s


def Planetary_phase_signal(Times, phi, depth, V1, sp_params, fit=True):
    """
    Calculate the phase signal of the planet
    :param Times: (numpy 1D array) array of times
    :param phi: (float) phase signal offset (days)
    :param depth: (float) eclipse depth
    :param V1: (float) amplitude of phase signal (Flux_day - Flux_night)
    :param sp_params: (object) An instance of the spiderman ModelParams function
    :param fit: (bool) Whether this is used in a fit (deprecated)

    :return: (numpy 1D array) of length 'Times' with planetary phase signal
    """
    sina = 2 * np.pi * (1. / sp_params.per)
    sine = np.sin(sina * Times + phi - sina * sp_params.t_secondary + 0.5 * np.pi)
    return V1 * sine + depth - V1


def Planetary_signal(t, phi, depth, V1, t0, sp_params, calc_time_delay=True, fit=True):
    """
    Calculate the planetary flux as a function of time t

    :param t: (numpy 1D array) with the times of the exposures
    :param phi: (float) phase offset in BJD
    :param depth: (float) eclipse depth
    :param V1: (float) sine amplitude (Flux_day - Flux_night)
    :param t0: (float) Mid-transit time offset (min)
    :param sp_params: (object) An instance of the spiderman ModelParams function
    :param calc_time_delay: (bool) Whether to include the effect of time dilation from the distance star-planet
    :param fit: (bool) Whether this is used in a fit (deprecated)

    :return: (numpy 1D array) of length t with the planetary flux
    """

    This_phase = (t > sp_params.t_secondary + t0 - 0.5 * sp_params.per) & (t < sp_params.t_secondary + t0 + 0.5 * sp_params.per)

    psi_p = np.zeros_like(t)
    if calc_time_delay:
        m_temp = batman.TransitModel(sp_params, t + t0, transittype='secondary')
        time_delay = calc_lighttime_delay_at_time(sp_params, m_temp, sp_params.t0 - t0)
    else:
        time_delay = 0.

    sp_params.fp = depth
    m = batman.TransitModel(sp_params, t + t0 - time_delay, transittype='secondary')
    lc = m.light_curve(sp_params)

    psi_p = lc - 1

    return psi_p - depth

def calc_lighttime_delay_at_time(sp_params, m, t_trans):
    """
    Calculate the time dilation between the planet at its transit position and the position at t_a_peri, a time after
     periastron.

    :return: (numpy 1D array) of length t with the planetary flux
    :param sp_params: (object) spiderman/batman parameters object
    :param m: (object) Batman TransitModel object for a transit/eclipse model without
    :param t_trans: (float) The time of mid-transit for this planet.
    :return: (numpy 1D array) time delay in days
    """
    true_anomaly = m.get_true_anomaly()

    mtemp = batman.TransitModel(sp_params, np.array([t_trans]))
    true_anomaly_transit = mtemp.get_true_anomaly()[0]

    if sp_params.ecc > 0.:
        d_planet_time = (sp_params.a * sp_params.r_s * cs.R_sun.value * (1 - sp_params.ecc**2.)
                         / (1 + sp_params.ecc * np.cos(true_anomaly)))
        d_planet_transit = (sp_params.a * sp_params.r_s * cs.R_sun.value * (1 - sp_params.ecc**2.)
                            / (1 + sp_params.ecc * np.cos(true_anomaly_transit)))
    else:
        d_planet_time = sp_params.a * sp_params.r_s * cs.R_sun.value
        d_planet_transit = sp_params.a * sp_params.r_s * cs.R_sun.value

    Angle = true_anomaly - true_anomaly_transit  #angle between transit-line-of-sight and position at time Times

    return (d_planet_time * -np.cos(Angle) + d_planet_transit) / cs.c.value / 24. / 60. / 60.



def Stellar_signal(t, Stel_puls_amp, Stel_puls_phase, Harm_amp, t0, sp_params):
    """
    Calculate the Stellar signal.
    To do this, we use a few extra parameters that should be in the sp_params instance.
    They aren't standard for the spider parameters, so you will need to add them.
    If you are working with the hstscan pipeline, you can put them into reduction.py.

    :param t:
    :param Stel_puls_amp:
    :param Stel_puls_phase:
    :param Harm_amp:
    :param t0:
    :param sp_params:
    :return:
    """
    if Stel_puls_amp != 0.:
        if not np.all([hasattr(sp_params, 'pulse_alpha'), hasattr(sp_params, 'pulse_beta'),
                       hasattr(sp_params, 'pulse_Pi')]):
            print "The planet instance does not have the right parameters for stellar pulsations."
            print "Check the KELT-9b example in reduction.py"
        alpha = Stel_puls_amp * sp_params.pulse_alpha
        beta = Stel_puls_amp * sp_params.pulse_beta
        Pi = sp_params.pulse_Pi
    else:
        alpha, beta = 0, 0
        Pi = 1.
    if Harm_amp != 0.:
        if not np.all([hasattr(sp_params, 'harm_A1'), hasattr(sp_params, 'harm_A2'), hasattr(sp_params, 'harm_B2'),
                       hasattr(sp_params, 'harm_A3'), hasattr(sp_params, 'harm_B3')]):
            print "The planet instance does not have the right parameters for harmonics."
            print "Check the KELT-9b example in reduction.py"
            A1 = sp_params.harm_A1
            A2 = sp_params.harm_A2
            B2 = sp_params.harm_B2
            A3 = sp_params.harm_A3
            B3 = sp_params.harm_B3
    else:
        A1, A2, B2, A3, B3 = 0., 0., 0., 0., 0.

    ksi = (t - sp_params.t_secondary - Stel_puls_phase / (2 * np.pi)) / Pi


    Phi = (t - sp_params.t0 + t0) / sp_params.per

    psi_s_1 = A1 * np.sin(2 * np.pi * Phi)
    psi_s_2 = B2 * np.cos(2 * np.pi * 2 * Phi) + A2 * np.sin(2 * np.pi * 2 * Phi)
    psi_s_3 = B3 * np.cos(2 * np.pi * 3 * Phi) + A3 * np.sin(2 * np.pi * 3 * Phi)

    Theta = (1.e6 + alpha * np.sin(2 * np.pi * ksi) + beta * np.cos(2 * np.pi * ksi)) / 1.e6
    Harmonics = (1.e6 + Harm_amp * (psi_s_1 + psi_s_2 + psi_s_3)) / 1.e6
    return Theta * Harmonics


def area_of_circular_segment(d, r):
    """
    Calculates the area of a segment of a circle.
    This is useful for calculating what part of the planet is not occulted by the star.
    Here the star is assumed to be much larger than the planet.

    :param d:
    :param r:
    :return:
    """
    return r * (r * np.arccos(d / r) - d * np.sqrt(1 - d**2. / r**2.))


def lnprior(theta, parameters, names):
    """
    Set the priors. There are three types of priors:
    -flat (default). If value theta is between its prior_min and prior_max for every parameter, zero is returned.
                     If not, -infinity is returned to turn the MCMC towards the prior values
    -gaussian. A gaussian prior is added for each parameter that has a gaussian prior. This to make the expected value
               more likely
    -gamma. A gamma function is used here. This is basically a gaussian function that is cut off at zero. This is useful
            for parameters that are close to zero.

    :param theta: (list) of parameter values
    :param parameters: (dict) of parameter settings (incl prior type)
    :param
    """
    prior = 0.
    for i,name in enumerate(names):
        if parameters[name]['prior_min'] < theta[i] < parameters[name]['prior_max']:
            if parameters[name]['prior'] == 'gaussian':
                this_prior = np.exp(-0.5 * (theta[i] - parameters[name]['prior_mu'])**2 /
                                    (parameters[name]['prior_sigma']**2) )
                if 'prior_strength' in parameters[name].keys():
                    this_prior *= parameters[name]['prior_strength']
                if prior == 0.:
                    prior = this_prior
                else:
                    prior *= this_prior
            elif parameters[name]['prior'] == 'gamma':
                Theta = np.sqrt(parameters[name]['prior_sigma'] ** 2. / parameters[name]['prior_mu'])
                k = parameters[name]['prior_mu'] / Theta
                x = np.linspace(1.e-5, 0.001, 100)
                f = 1. / (gamma(k) * Theta ** k) * x ** (k - 1.) * np.exp(-x / Theta)
                this_prior = 1.e-3 / (gamma(k) * Theta ** k) * theta[i] ** (k - 1.) * np.exp(- theta[i] / Theta) / np.trapz(f[1:], x[1:])
                if prior == 0.:
                    prior = this_prior
                else:
                    prior *= this_prior
        else:
            return -np.inf

    return prior

def lnlike(pars, Times, data, Forward, sp_params, orbit_times, satellite_time, exc_egress, t0_not_yet_set, sigma,
           same_depth, same_Rorb, fix_stel_puls_phase, Stel_puls_phase):
    if same_depth and same_Rorb and not fix_stel_puls_phase:
        (depth_F, deltat, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R, Rorb1_F, tau_F, Stel_puls_phase) = pars
        depth_R = depth_F
        Rorb1_R = Rorb1_F
        tau_R = tau_F
    if same_depth and same_Rorb and fix_stel_puls_phase:
        (depth_F, deltat, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R, Rorb1_F, tau_F) = pars
        depth_R = depth_F
        Rorb1_R = Rorb1_F
        tau_R = tau_F
    elif same_depth:
        (depth_F, deltat, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R, Rorb1_F, Rorb1_R, tau_F, tau_R,
         Stel_puls_phase) = pars
        depth_R = depth_F
    else:
        (depth_F, depth_R, deltat, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R, Rorb1_F, Rorb1_R, tau_F, tau_R, Stel_puls_phase) = pars
    model = Func_exp_eclipse(Times, depth_F, depth_R,
                                    deltat, sp_params.inc,
                                    sp_params.ecc, sp_params.a, sp_params.u[0], sp_params.u[1],
                                    0.,
                                    V2_F, V2_R,
                                    Stellar_flux_F, Stellar_flux_R, Rorb1_F,
                                    Rorb1_R, 0., 0., tau_F, tau_R,
                                    0., 0., 0., 0., 0.,
                                    Forward, sp_params, orbit_times, satellite_time, exc_egress, t0_not_yet_set,
                                    Stel_puls_phase=Stel_puls_phase, Stel_puls_amp=1.,#pars['Stel_puls_amp'],
                                    Harm_amp=0.)#pars['Harmonics_amplitude'])

    return -0.5 * (np.sum((data - model) ** 2. / sigma ** 2.))


def lnlike_c(pars, Times, data, Forward, sp_params, orbit_times, satellite_time, exc_egress, t0_not_yet_set, sigma,
           same_depth, deltat, inc, Rorb1_F, Rorb1_R, tau_F, tau_R, Stel_pulse_phase):
    if same_depth:
        (depth_F, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R) = pars
        depth_R = depth_F
    else:
        (depth_F, depth_R, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R) = pars
    model = Func_exp_eclipse(Times, depth_F, depth_R,
                                    deltat, inc,
                                    sp_params.ecc, sp_params.a, sp_params.u[0], sp_params.u[1], 0.,
                                    V2_F, V2_R,
                                    Stellar_flux_F, Stellar_flux_R, Rorb1_F,
                                    Rorb1_R, 0., 0., tau_F, tau_R,
                                    0., 0., 0., 0., 0.,
                                    Forward, sp_params, orbit_times, satellite_time, exc_egress, t0_not_yet_set,
                                    Stel_puls_phase=Stel_pulse_phase, Stel_puls_amp=1.,#pars['Stel_puls_amp'],
                                    Harm_amp=0.)#pars['Harmonics_amplitude'])

    return -0.5 * (np.sum((data - model) ** 2. / sigma ** 2.))

def lnlike_no_abs_flux(pars, Times, data, Forward, sp_params, orbit_times, satellite_time, exc_egress, t0_not_yet_set,
                       sigma, Stellar_flux_F, Stellar_flux_R, same_depth, same_Rorb, fix_stel_puls_phase, Stel_puls_phase):
    if same_depth and same_Rorb and not fix_stel_puls_phase:
        (depth_F, deltat, V2_F, V2_R, Rorb1_F, tau_F, Stel_puls_phase) = pars
        depth_R = depth_F
        Rorb1_R = Rorb1_F
        tau_R = tau_F
    if same_depth and same_Rorb and fix_stel_puls_phase:
        (depth_F, deltat, V2_F, V2_R, Rorb1_F, tau_F) = pars
        depth_R = depth_F
        Rorb1_R = Rorb1_F
        tau_R = tau_F
    elif same_depth:
        (depth_F, deltat, V2_F, V2_R, Rorb1_F, Rorb1_R, tau_F, tau_R, Stel_puls_phase) = pars
        depth_R = depth_F
    else:
        (depth_F, depth_R, deltat, V2_F, V2_R, Rorb1_F, Rorb1_R, tau_F, tau_R, Stel_puls_phase) = pars
    model = Func_exp_eclipse(Times, depth_F, depth_R,
                                    deltat, sp_params.inc,
                                    sp_params.ecc, sp_params.a, sp_params.u[0], sp_params[1], 0.,
                                    V2_F, V2_R,
                                    Stellar_flux_F, Stellar_flux_R, Rorb1_F,
                                    Rorb1_R, 0., 0., tau_F, tau_R,
                                    0., 0., 0., 0., 0.,
                                    Forward, sp_params, orbit_times, satellite_time, exc_egress, t0_not_yet_set,
                                    Stel_puls_phase=Stel_puls_phase, Stel_puls_amp=1.,#pars['Stel_puls_amp'],
                                    Harm_amp=0.)#pars['Harmonics_amplitude'])

    return -0.5 * (np.sum((data - model) ** 2. / sigma ** 2.))


def lnlike_multi_color(pars, Times, data, Forward, sp_params, orbit_times, satellite_time, exc_egress, t0_not_yet_set,
                     sigma, deltat, same_depth, same_Rorb, fix_stel_puls_phase, Stel_puls_phase, no_t0_inc,
                       avg_wave_bin, debug=False):
    if no_t0_inc:
        (depth_R, depth_G, depth_B, V2_F_R, V2_R_R, V2_F_G, V2_R_G, V2_F_B, V2_R_B,
         Stellar_flux_F_R, Stellar_flux_R_R, Stellar_flux_F_G, Stellar_flux_R_G, Stellar_flux_F_B, Stellar_flux_R_B,
         Rorb1_m, Rorb1_b, tau_m, tau_b, Stel_puls_phase) = pars
        depth = [depth_R, depth_G, depth_B]
        V2_F = [V2_F_R, V2_F_G, V2_F_B]
        V2_R = [V2_R_R, V2_R_G, V2_R_B]
        Stellar_flux_F = [Stellar_flux_F_R, Stellar_flux_F_G, Stellar_flux_F_B]
        Stellar_flux_R = [Stellar_flux_R_R, Stellar_flux_R_G, Stellar_flux_R_B]
        times = [Times[0], Times[1], Times[2]]
        Data = [data[0], data[1], data[2]]
        Sigma = [sigma[0], sigma[1], sigma[2]]

    else:
        (depth_R, depth_G, depth_B, deltat, V2_F_R, V2_R_R, V2_F_G, V2_R_G, V2_F_B, V2_R_B,
         Stellar_flux_F_R, Stellar_flux_R_R, Stellar_flux_F_G, Stellar_flux_R_G, Stellar_flux_F_B, Stellar_flux_R_B,
         Rorb1_m, Rorb1_b, tau_m, tau_b, Stel_puls_phase) = pars
        depth = [depth_R, depth_G, depth_B]
        V2_F = [V2_F_R, V2_F_G, V2_F_B]
        V2_R = [V2_R_R, V2_R_G, V2_R_B]
        Stellar_flux_F = [Stellar_flux_F_R, Stellar_flux_F_G, Stellar_flux_F_B]
        Stellar_flux_R = [Stellar_flux_R_R, Stellar_flux_R_G, Stellar_flux_R_B]
        times = [Times[0], Times[1], Times[2]]
        Data = [data[0], data[1], data[2]]
        Sigma = [sigma[0], sigma[1], sigma[2]]
    resids = 0
    for wave, Times_c, data_c, sigma_c,  d, V2_f, V2_r, Stel_f_F, Stel_f_R, in zip(avg_wave_bin, times, Data, Sigma, depth, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R):
        tau = linear(wave, tau_m, tau_b)
        Rorb = linear(wave, Rorb1_m, Rorb1_b)
        if tau <= 0.:  #Couldn't process this into a prior because otherwise the below function would crash.
            return -np.inf
        if Rorb <= 0.:
            return -np.inf
        model = Func_exp_eclipse(Times_c, d, d,
                                    deltat, sp_params.inc,
                                    sp_params.ecc, sp_params.a, sp_params.u[0], sp_params[1], 0.,
                                    V2_f, V2_r,
                                    Stel_f_F, Stel_f_R, Rorb,
                                    Rorb, 0., 0., tau, tau,
                                    0., 0., 0., 0., 0.,
                                    Forward, sp_params, orbit_times, satellite_time, exc_egress, t0_not_yet_set,
                                    Stel_puls_phase=Stel_puls_phase, Stel_puls_amp=1.,#pars['Stel_puls_amp'],
                                    Harm_amp=0.)#pars['Harmonics_amplitude'])
        resids += np.sum((data_c - model) ** 2. / sigma_c ** 2.)
    #print "dd", d, d
    #print "t, i", deltat, sp_params.inc, 0.
    #print "V2", V2_f, V2_r
    #print "Stel_f", Stel_f_F, Stel_f_R
    #print "Rorb, 00, tau", Rorb, Rorb, 0., 0., tau, tau
    #print "model", d, model
    #plt.scatter(Times_c, model)
    #plt.title(resids)
    #plt.show()
    #print tau, tau_b, tau_m
    #print "resids", resids
    #if tau < 0.01:
    #    print "{}, {}".format((data_c - model) ** 2. / sigma_c ** 2., resids)#, data_c, model
    #print avg_wave_bin
    return -0.5 * (np.sum(resids))


def lnlike_no_t0_inc(pars, Times, data, Forward, sp_params, orbit_times, satellite_time, exc_egress, t0_not_yet_set,
                     sigma, deltat, same_depth, same_Rorb, fix_stel_puls_phase, Stel_puls_phase):
    if same_depth and same_Rorb and not fix_stel_puls_phase:
        (depth_F, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R, Rorb1_F, tau_F, Stel_puls_phase) = pars
        depth_R = depth_F
        Rorb1_R = Rorb1_F
        tau_R = tau_F
    elif same_depth and same_Rorb and fix_stel_puls_phase:
        (depth_F, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R, Rorb1_F, tau_F) = pars
        depth_R = depth_F
        Rorb1_R = Rorb1_F
        tau_R = tau_F
    elif same_depth:
        (depth_F, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R, Rorb1_F, Rorb1_R, tau_F, tau_R,
         Stel_puls_phase) = pars
        depth_R = depth_F
    else:
        (depth_F, depth_R, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R, Rorb1_F, Rorb1_R, tau_F, tau_R, Stel_puls_phase) = pars
    model = Func_exp_eclipse(Times, depth_F, depth_R,
                                    deltat, sp_params.inc,
                                    sp_params.ecc, sp_params.a, sp_params.u[0], sp_params[1], 0.,
                                    V2_F, V2_R,
                                    Stellar_flux_F, Stellar_flux_R, Rorb1_F,
                                    Rorb1_R, 0., 0., tau_F, tau_R,
                                    0., 0., 0., 0., 0.,
                                    Forward, sp_params, orbit_times, satellite_time, exc_egress, t0_not_yet_set,
                                    Stel_puls_phase=Stel_puls_phase, Stel_puls_amp=1.,#pars['Stel_puls_amp'],
                                    Harm_amp=0.)#pars['Harmonics_amplitude'])
    return -0.5 * (np.sum((data - model) ** 2. / sigma ** 2.))

def lnlike_RECTE(pars, Times, data, Forward, sp_params, orbit_times, satellite_time, exc_egress, t0_not_yet_set, sigma,
                 exptime, same_depth):
    if same_depth:
        (depth_F, deltat, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R, E0_s, E0_f, Delta_Es, Delta_Ef, Satellite_flux,
         Stel_puls_phase) = pars
        depth_R = depth_F
    else:
        (depth_F, depth_R, deltat, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R, E0_s, E0_f, Delta_Es, Delta_Ef,
         Satellite_flux, Stel_puls_phase) = pars
    model = Func_recte_eclipse(Times, depth_F, depth_R,
                        deltat, sp_params.inc, 
                        sp_params.ecc, sp_params.a, sp_params.u[0], sp_params.u[1], 0.,
                        V2_F, V2_R,
                        Stellar_flux_F, Stellar_flux_R,
                        E0_s, E0_f, Delta_Es, Delta_Ef,
                        Satellite_flux, 0., 0., 0., 0.,
                        Forward, sp_params, orbit_times, satellite_time, exc_egress, exptime, t0_not_yet_set,
                        Stel_puls_phase=Stel_puls_phase, Stel_puls_amp=1.,  # pars['Stel_puls_amp'],
                        Harm_amp=0.)  # pars['Harmonics_amplitude'])

    return -0.5 * (np.sum((data - model) ** 2. / sigma ** 2.))

def lnlike_RECTE_RGB(pars, Times, data, Forward, sp_params, orbit_times, deltat, satellite_time, exc_egress, t0_not_yet_set, sigma,
                 exptime):
    (depth_R, depth_G, depth_B,  V2_F_R, V2_R_R, V2_F_G, V2_R_G, V2_F_B, V2_R_B, Stellar_flux_F_R,
     Stellar_flux_R_R, Stellar_flux_F_G, Stellar_flux_R_G, Stellar_flux_F_B, Stellar_flux_R_B, E0_s_R, E0_s_G, E0_s_B,
     E0_f_R, E0_f_G, E0_f_B, Delta_Es_R, Delta_Es_G, Delta_Es_B, Delta_Ef_R, Delta_Ef_G, Delta_Ef_B, Satellite_flux_R,
     Satellite_flux_G, Satellite_flux_B, Stel_puls_phase) = pars

    data_R, data_G, data_B = data
    sigma_R, sigma_G, sigma_B = sigma

    resids = 0
    model_R = Func_recte_eclipse(Times[0], depth_R, depth_R,
                        deltat, sp_params.inc, 
                        sp_params.ecc, sp_params.a, sp_params.u[0], sp_params.u[1], 0.,
                        V2_F_R, V2_R_R,
                        Stellar_flux_F_R, Stellar_flux_R_R,
                        E0_s_R, E0_f_R, Delta_Es_R, Delta_Ef_R,
                        Satellite_flux_R, 0., 0., 0., 0.,
                        Forward, sp_params, orbit_times, satellite_time, exc_egress, exptime, t0_not_yet_set,
                        Stel_puls_phase=Stel_puls_phase, Stel_puls_amp=1.,  # pars['Stel_puls_amp'],
                        Harm_amp=0.)  # pars['Harmonics_amplitude'])
    model_G = Func_recte_eclipse(Times[1], depth_G, depth_G,
                        deltat, sp_params.inc, 
                        sp_params.ecc, sp_params.a, sp_params.u[0], sp_params.u[1], 0.,
                        V2_F_G, V2_R_G,
                        Stellar_flux_F_G, Stellar_flux_R_G,
                        E0_s_G, E0_f_G, Delta_Es_G, Delta_Ef_G,
                        Satellite_flux_G, 0., 0., 0., 0.,
                        Forward, sp_params, orbit_times, satellite_time, exc_egress, exptime, t0_not_yet_set,
                        Stel_puls_phase=Stel_puls_phase, Stel_puls_amp=1.,  # pars['Stel_puls_amp'],
                        Harm_amp=0.)  # pars['Harmonics_amplitude'])
    model_B = Func_recte_eclipse(Times[2], depth_B, depth_B,
                        deltat, sp_params.inc, 
                        sp_params.ecc, sp_params.a, sp_params.u[0], sp_params.u[1], 0.,
                        V2_F_B, V2_R_B,
                        Stellar_flux_F_B, Stellar_flux_R_B,
                        E0_s_B, E0_f_B, Delta_Es_B, Delta_Ef_B,
                        Satellite_flux_B, 0., 0., 0., 0.,
                        Forward, sp_params, orbit_times, satellite_time, exc_egress, exptime, t0_not_yet_set,
                        Stel_puls_phase=Stel_puls_phase, Stel_puls_amp=1.,  # pars['Stel_puls_amp'],
                        Harm_amp=0.)  # pars['Harmonics_amplitude'])


    resids = np.sum((data_R[exc_egress] - model_R[exc_egress])**2. / sigma_R[exc_egress] **2.) + \
             np.sum((data_G[exc_egress] - model_G[exc_egress])**2. / sigma_G[exc_egress] **2.) + \
             np.sum((data_B[exc_egress] - model_B[exc_egress])**2. / sigma_B[exc_egress] **2.)

    return -0.5 * resids


def lnlike_no_t0_inc_RECTE(pars, Times, data, Forward, sp_params, orbit_times, satellite_time, exc_egress, t0_not_yet_set,
                           sigma, deltat, exptime, same_depth, Stel_puls_phase, fix_stel_puls_phase):
    if fix_stel_puls_phase and same_depth:
        (depth_F, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R, E0_s, E0_f, Delta_Es, Delta_Ef, Satellite_flux) = pars
        depth_R = depth_F
    elif fix_stel_puls_phase and not same_depth:
        (depth_F, depth_R, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R, E0_s, E0_f, Delta_Es, Delta_Ef, Satellite_flux) = pars
    elif same_depth:
        (depth_F, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R, E0_s, E0_f, Delta_Es, Delta_Ef, Satellite_flux,
         Stel_puls_phase) = pars
        depth_R = depth_F
    else:
        (depth_F, depth_R, V2_F, V2_R, Stellar_flux_F, Stellar_flux_R, E0_s, E0_f, Delta_Es, Delta_Ef, Satellite_flux,
         Stel_puls_phase) = pars
    model = Func_recte_eclipse(Times, depth_F, depth_R,
                       deltat, sp_params.inc, 
                       sp_params.ecc, sp_params.a, sp_params.u[0], sp_params.u[1], 0.,
                       V2_F, V2_R,
                       Stellar_flux_F, Stellar_flux_R,
                       E0_s, E0_f, Delta_Es, Delta_Ef,
                       Satellite_flux, 0., 0., 0., 0.,
                       Forward, sp_params, orbit_times, satellite_time, exc_egress, exptime, t0_not_yet_set,
                       Stel_puls_phase = Stel_puls_phase, Stel_puls_amp = 1.,  # pars['Stel_puls_amp'],
                       Harm_amp = 0.)  # pars['Harmonics_amplitude'])

    return -0.5 * (np.sum((data[exc_egress] - model[exc_egress]) ** 2. / sigma[exc_egress] ** 2.))

def lnprob(theta, x, y, no_t0_inc, no_abs_flux, same_depth, same_Rorb, fix_stel_puls_phase, Forward, sp_params, orbit_times, satellite_time,
           exc_egress, t0_not_yet_set, sigma, parameters, names, deltat, FluxF, FluxR, Stel_puls_phase, avg_wave_bins, exptime = 90., fittype='exp'):
            '''
            Log of the probability distribution
            The chance that a walker moves to any given point
            '''
            lp = lnprior(theta, parameters, names)
            if not np.isfinite(lp):
                return -np.inf
            if fittype == 'RGB':
                return lp + lnlike_multi_color(theta, x, y, Forward, sp_params, orbit_times, satellite_time, exc_egress,
                                               t0_not_yet_set, sigma, deltat, same_depth, same_Rorb,
                                               fix_stel_puls_phase, Stel_puls_phase, no_t0_inc, avg_wave_bins)
            elif fittype == 'RECTE_RGB':
                return lp + lnlike_RECTE_RGB(theta, x, y, Forward, sp_params, orbit_times, deltat, satellite_time, exc_egress,
                                               t0_not_yet_set, sigma, exptime)
            if no_t0_inc:
                if fittype == 'RECTE':
                    return lp + lnlike_no_t0_inc_RECTE(theta, x, y, Forward, sp_params, orbit_times, satellite_time,
                                                       exc_egress, t0_not_yet_set, sigma, deltat, exptime, same_depth,
                                                       Stel_puls_phase, fix_stel_puls_phase)
                else:
                    return lp + lnlike_no_t0_inc(theta, x, y, Forward, sp_params, orbit_times, satellite_time, exc_egress,
                                             t0_not_yet_set, sigma, deltat, same_depth, same_Rorb, fix_stel_puls_phase, Stel_puls_phase)
            elif no_abs_flux:
                assert fittype != 'RECTE', "RECTE version is not implemented yet"
                return lp + lnlike_no_abs_flux(theta, x, y, Forward, sp_params, orbit_times, satellite_time, exc_egress,
                                               t0_not_yet_set, sigma, FluxF, FluxR, same_depth, same_Rorb, fix_stel_puls_phase, Stel_puls_phase)
            else:
                if fittype == 'RECTE':
                    return lp + lnlike_RECTE(theta, x, y, Forward, sp_params, orbit_times, satellite_time, exc_egress,
                                             t0_not_yet_set, sigma, exptime, same_depth)
                else:
                    return lp + lnlike(theta, x, y, Forward, sp_params, orbit_times, satellite_time, exc_egress,
                                       t0_not_yet_set, sigma, same_depth, same_Rorb, fix_stel_puls_phase, Stel_puls_phase)


def lnprob_c(theta, x, y, no_t0_inc, no_abs_flux, same_depth, Forward, sp_params, orbit_times, satellite_time,
             exc_egress, t0_not_yet_set, sigma, parameters, names, deltat,  inc, Flux_F, Flux_R, Rorb1_F, Rorb1_R,
             tau_F, tau_R, stel_pulse_phase, avg_wave_bins, exptime=90., fittype='exp'):
    '''
    Log of the probability distribution
    The chance that a walker moves to any given point
    '''
    lp = lnprior(theta, parameters, names)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_c(theta, x, y, Forward, sp_params, orbit_times, satellite_time, exc_egress,
                            t0_not_yet_set, sigma, same_depth, deltat, inc, Rorb1_F, Rorb1_R, tau_F, tau_R, stel_pulse_phase)



class PlanetParams:
    def __init__(self, a, t0, per, t_secondary, w, rp, ecc, inc):
        self.a = a
        self.t0 = t0
        self.per = per
        self.t_secondary = t_secondary
        self.w = w
        self.rp = rp
        self.ecc = ecc
        self.inc = inc


class Planet:
    def __init__(self, name):
        reload(sp.web)
        self.name = name
        self.params = r.custom_transit_params(name)
        self.sp_params = sp.ModelParams(brightness_model="lambertian")
        self.sp_params.n_layers = 5
        for key in dir(self.params):
            if not key.startswith('_'):
                val = self.params.__dict__[key]
                if type(val) is int:
                    val = float(val)
                self.sp_params.__dict__[key] = val
        self.sp_params.p_u1 = 0  # Planetary limb darkening parameter
        self.sp_params.p_u2 = 0  # Planetary limb darkening parameter




    def MCMC_exponential(self, Times, data, errors, orbit_times, Forward, exptime=90, t0=0., deltat=0., a=10.,
                               phi=1000., inc=1000., sattime=1.e6, l1 = 1.1e-6, l2=1.7e-6,
                                   Flux_F=1., Flux_R=1., stel_pulse_phase=0., fix_pulse_phase=False, no_t0_inc=False,
                                   no_abs_flux=False, same_depth=False, nsteps = 7000, nburn = 1000,
                                   Rorb1_F_estimate = 0.0002, Rorb1_F_estimate_sigma = 5.e-5,
                                   Rorb1_R_estimate = 0.0002, Rorb1_R_estimate_sigma = 5.e-5, set_deltat=False,
                                   tau_F_estimate = 0.0002, tau_R_estimate=0.0002, tau_F_estimate_sigma=5.e-5,
                                   tau_R_estimate_sigma=5.e-5,
                                   same_Rorb=False, fix_stel_puls_phase=False, priortype_Rorb='uniform'):
        """
        MCMC the light curve of your planet.

        :param Times: (1D numpy array) of the times of exposure in BJD
        :param data: (1D numpy array) of the flux in electron per seconds per pixel
        :param errors: (1D numpy array) of errorbars in same units as data
        :param orbit_times: (2D list) List of time ranges (in days) of the orbits such that all start times of an orbit
                           fall into this range. 2 orbit example: orbit_times = [[-1.e-6, 0.066], [0.066, 0.132]]
                           This list should include *all* orbit, even the discarded first orbit.
        :param Forward: (numpy boolarray) Array of True/False on whether an exposure is Forward scanned.
                        This has the same length as 'Times'
        :param exptime: (float) exposure time of each exposure in seconds
        :param bounds:
        :param verbose:
        :param t0:
        :param tau:
        :param a:
        :param phi:
        :param inc:
        :param sattime: The time (BJD) of the exposure in which the sattelite crossed.
        :param prev_opt:
        :param prev_err:
        :param Firstramp:
        :param l1:
        :param l2:
        :param secondtau:
        :param sine: Whether to include a sine (phase curve) in the fit.
        :param harmonics: Whether to include first, secondary and tertiary harmonics.
        :return:
        """
        assert Times[1] - Times[0] < 1. and Times[0] > 0., "Please enter Times in BJD"

        self.sp_params.l1 = l1  # The starting wavelength in meters
        self.sp_params.l2 = l2  # The ending wavelength in meters
        self.Forward = Forward
        self.satellite_time = sattime
        self.exptime = exptime
        self.incl_eclipse = True
        self.t0_not_yet_set = True

        if t0 != 0.:  # Set the mid transit time if given.
            self.sp_params.t0 = t0
            self.sp_params.t_secondary = self.sp_params.t0 + self.sp_params.per / 2. * \
                                         (1 + 4 * self.sp_params.ecc * np.cos(self.sp_params.w* np.pi / 180. ))
        if a != 10.:
            self.sp_params.a = a
        if phi != 1000.:
            self.phi = phi
        if inc != 1000.:
            self.sp_params.inc = inc
        i0 = self.sp_params.inc

        self.orbit_times = orbit_times
        self.orbit_times_orig = orbit_times.copy()

        #self.exc_egress = ~((Times >= self.orbit_times[-2][0]) & (Times <= self.orbit_times[-2][1]))
        Excl_sat = Times != sattime

        Excl_egress = np.ones_like(Excl_sat, dtype=bool)
        Excl_egress[51:62] = False
        self.exc_egress = Excl_egress

        self.func = Func_exp_eclipse

        self.parameters = {}
        self.parameters['depth_F'] = {'value': 0.0016, 'prior_min':0., 'prior_max':0.1, 'prior': 'uniform',
                                      'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        self.parameters['depth_R'] = {'value': 0.0016, 'prior_min':0., 'prior_max':0.1, 'prior':'uniform',
                                      'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        if set_deltat:
            self.parameters['deltat'] = {'value':deltat, 'prior_min':deltat-0.00005, 'prior_max':deltat+0.00005, 'prior':'uniform',
                                         'walker_locs':'flat'}
        else:
            self.parameters['deltat'] = {'value': 0.001, 'prior_min':-0.05, 'prior_max':0.05, 'prior': 'uniform',
                                     'walker_locs':'flat'}
        #self.parameters['inc']= {'value': 87.2, 'prior_min':85.2, 'prior_max':89.2, 'walker_locs':'flat'}
        self.parameters['V2_F'] = {'value': 0., 'prior_min':-0.5, 'prior_max':0.5, 'prior':'uniform',
                                   'walker_locs':'gaussian', 'walker_sigma':0.002, 'sigma_type':'absolute'}
        self.parameters['V2_R'] = {'value': 0., 'prior_min':-0.5, 'prior_max':0.5, 'prior':'uniform',
                                   'walker_locs':'gaussian', 'walker_sigma':0.002, 'sigma_type':'absolute'}
        self.parameters['Stellar_flux_F'] = {'value': 1., 'prior_min':0.9, 'prior_max':1.1, 'prior': 'uniform',
                                             'walker_locs':'gaussian', 'walker_sigma':0.001, 'sigma_type':'absolute'}
        self.parameters['Stellar_flux_R'] = {'value': 1., 'prior_min':0.9, 'prior_max':1.1, 'prior': 'uniform',
                                             'walker_locs':'gaussian', 'walker_sigma':0.001, 'sigma_type':'absolute'}
        if priortype_Rorb == 'uniform':
            self.parameters['Rorb1_F'] = {'value': 0.0005, 'prior_min':0., 'prior_max':0.01, 'prior':'uniform',
                                          'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        elif priortype_Rorb in ['gamma', 'gaussian']:
            self.parameters['Rorb1_F'] = {'value': Rorb1_F_estimate, 'prior_min':0., 'prior_max':0.01, 'prior':priortype_Rorb,
                                          'prior_mu':Rorb1_F_estimate, 'prior_sigma': Rorb1_F_estimate_sigma,
                                          'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        #self.parameters['Rorb1_R'] = {'value': 0.0005, 'prior_min':0., 'prior_max':0.01, 'prior':'uniform',
        #                              'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        #self.parameters['Rorb1_R'] = {'value': 0.0005, 'prior_min':0., 'prior_max':0.01, 'prior':'gaussian',
        #                              'prior_mu':Rorb1_R_estimate, 'prior_sigma':Rorb1_R_estimate_sigma,
        #                              'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        if priortype_Rorb == 'uniform':
            self.parameters['tau_F'] = {'value': tau_F_estimate, 'prior_min':0., 'prior_max':0.1, 'prior':'uniform',
                                        #'prior_min': tau_F_estimate - 1.e-5, 'prior_max':tau_F_estimate + 1.e-5, 'prior':'uniform',
                                        #'walker_locs':'gaussian', 'walker_sigma':0.5e-5, 'sigma_type':'relative'}
                                        'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        elif priortype_Rorb in ['gamma', 'gaussian']:
            self.parameters['tau_F'] = {'value': tau_F_estimate, 'prior_min':0., 'prior_max':0.1, 'prior':priortype_Rorb,
                                        'prior_mu':tau_F_estimate, 'prior_sigma':tau_F_estimate_sigma,
                                        'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
            #self.parameters['tau_R'] = {'value': 0.01, 'prior_min':0., 'prior_max':0.1, 'prior':'uniform',
            #                            'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        #self.parameters['tau_R'] = {'value': 0.01, 'prior_min':0., 'prior_max':0.1, 'prior':'gaussian',
        #                            'prior_mu':0.005, 'prior_sigma':0.002,
        #                            'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        #self.parameters['Satellite_flux'] = {'value': 0.01, 'prior_min':0., 'prior_max':0.1, 'walker_locs':'flat'}
        if fix_pulse_phase:
            self.parameters['Stel_puls_phase'] = {'value':stel_pulse_phase, 'prior_min':stel_pulse_phase-0.0005, 'prior_max':stel_pulse_phase+0.0005, 'prior':'gaussian',
                                                  'prior_mu':0., 'prior_sigma':0.001, 'walker_locs':'flat'}
        else:
            self.parameters['Stel_puls_phase'] = {'value': stel_pulse_phase, 'prior_min':-0.5, 'prior_max':0.5, 'prior':'gaussian',
                                              'prior_mu':0., 'prior_sigma':0.1, 'walker_locs':'flat'}


        if no_t0_inc and not same_depth and not same_Rorb:
            ndim, nwalkers = 11, 110  # choose the number of walkers, ndim here is the number of fit parameters
            names = ['depth_F', 'depth_R', 'V2_F', 'V2_R', 'Stellar_flux_F', 'Stellar_flux_R',
                     'Rorb1_F', 'Rorb1_R', 'tau_F', 'tau_R', 'Stel_puls_phase']
        elif no_abs_flux and same_depth:
            ndim, nwalkers = 9, 90
            names = ['depth_F', 'deltat', 'V2_F', 'V2_R', 'Rorb1_F', 'Rorb1_R', 'tau_F', 'tau_R',
                     'Stel_puls_phase']
        elif no_abs_flux and not same_depth:
            ndim, nwalkers = 10, 100
            names = ['depth_F', 'depth_R', 'deltat', 'V2_F', 'V2_R', 'Rorb1_F', 'Rorb1_R', 'tau_F', 'tau_R',
                     'Stel_puls_phase']
        elif same_depth and same_Rorb and no_t0_inc and not fix_stel_puls_phase:
            ndim, nwalkers = 8, 100
            names = ['depth_F', 'V2_F', 'V2_R', 'Stellar_flux_F', 'Stellar_flux_R',
                     'Rorb1_F', 'tau_F', 'Stel_puls_phase']
        elif same_depth and same_Rorb and no_t0_inc and fix_stel_puls_phase:
            ndim, nwalkers = 7, 100
            names = ['depth_F', 'V2_F', 'V2_R', 'Stellar_flux_F', 'Stellar_flux_R',
                     'Rorb1_F', 'tau_F']
        elif same_depth and not same_Rorb:
            ndim, nwalkers = 11, 110
            names = ['depth_F', 'deltat', 'V2_F', 'V2_R', 'Stellar_flux_F', 'Stellar_flux_R',
                     'Rorb1_F', 'Rorb1_R', 'tau_F', 'tau_R', 'Stel_puls_phase']
        else:
            ndim, nwalkers = 12, 120
            names = ['depth_F', 'depth_R', 'deltat', 'V2_F', 'V2_R', 'Stellar_flux_F', 'Stellar_flux_R',
                     'Rorb1_F', 'Rorb1_R', 'tau_F', 'tau_R', 'Stel_puls_phase']

        pos = []
        for name in names:
            if self.parameters[name]['walker_locs'] == 'flat':
                pos.append(np.random.uniform(low=self.parameters[name]['prior_min'],
                                                high=self.parameters[name]['prior_max'], size=nwalkers))
            elif self.parameters[name]['walker_locs'] == 'gaussian':
                if self.parameters[name]['sigma_type'] == 'relative':
                    pos.append(self.parameters[name]['value'] * (np.ones(nwalkers) + np.random.randn(nwalkers) *
                                                                 self.parameters[name]['walker_sigma']))
                elif self.parameters[name]['sigma_type'] == 'absolute':
                    pos.append(self.parameters[name]['value'] + np.random.randn(nwalkers) * self.parameters[name]['walker_sigma'])
                else:
                    print "Warning: incorrect value entred for 'sigma_type' for parameter ", name
            else:i
                print "Warning: incorrect value entred for 'walker_locs' for parameter ", name


        pos = np.array(pos).T

        # Now use emcee to run an MCMC for your setup


        with closing(Pool(processes=8)) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(Times, data, no_t0_inc, no_abs_flux, same_depth,
                                                                      same_Rorb, fix_stel_puls_phase,
                                                                      self.Forward, self.sp_params, self.orbit_times,
                                                                      self.satellite_time, self.exc_egress,
                                                                      self.t0_not_yet_set, errors,
                                                                      self.parameters, names, deltat, Flux_F, Flux_R,
                                                                      stel_pulse_phase, [1.4]),
                                            pool=pool)
                                                                #threads=8)
            samples = sampler.run_mcmc(pos, nsteps, progress=True)
            samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
        print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        return samples, sampler


    def Fit_with_exponential_lmfit(self, Times, data, errors, orbit_times, Forward, exptime=90, 
                       verbose=True, t0=0., deltat=0., a=10., ecc=-1., phi=1000., inc=1000., sattime=1.e6,
                       sine=False, stel_pulse=False, harmonics=False,
                       separate_depth=True, fix_pulse_amp=False, puls_amp=1., fix_limb_dark=False, limb_dark='linear',
                       fix_inclination=True, fix_eccentricity=True, fix_aRs=True, disable_first_ramp=False,
                       disable_second_ramp=False, polynomial=False, same_Rorb=True, nosat=False, Transit=False,
                       exc_indices=[],
                       fitting_method='least_squares', incl_first_orbit=False):
        """
        Fit a lightcurve with an exponential systematics fit and a Levenberg-Marquardt-ish algorithm (using package 
         lmfit)
        This assumes the first orbit has already been discarded. So, when I refer to the first orbit below, I really
         refer to the second orbit.

         The fitting algorithm used here is the python lmfit package (https://lmfit.github.io/lmfit-py/)

        :param Times: (1D numpy array) of the times of exposure in BJD
        :param data: (1D numpy array) of the flux. The out of transit mean should be 1.
        :param errors: (1D numpy array) of errorbars in same units as data
        :param orbit_times: (2D list) List of time ranges (in days) of the orbits such that all start times of an orbit
                           fall into this range. 2 orbit example: orbit_times = [[-1.e-6, 0.066], [0.066, 0.132]]
                           This list should include *all* orbit, even the discarded first orbit.
        :param Forward: (numpy boolarray) Array of True/False on whether an exposure is Forward scanned.
                        This has the same length as 'Times'
        :param exptime: (float) exposure time of each exposure in seconds
        :param verbose: (bool) Whether to print out anything (incl plots)
        :param t0: (float) The mid-transit time of the exoplanet in BJD. If using Transit=False, this needs to be the
                   mid-transit time preceding the eclipse data.
        :param deltat: (float) mid-time transit offset that will be used if that parameter is fixed
        :param a: (float) a/Rs that will be used if that parameter is fixed and a!=10.
        :param ecc: (float) eccentricity that will be used if that parameter is fixed and ecc!=-1.
        :param phi: (float) phase offset in days that will be used if that parameter is fixed and phi!=1000.
        :param inc: (float) orbital inclination in degrees that will be used if that parameter is fixed and inc!=1000.
        :param sattime: The time (BJD) of the exposure in which a satellite crossed. (this time stamp should be in the
                        Times array).
        :param sine: (bool) Whether to include a sine (as phase curve) into the lightcurve of the planet
        :param stel_pulse: (bool) Whether to include stellar pulsations into the light curve of the planet (requires
                           parameters pulse_alpha, pulse_beta and pulse_Pi to be added to the spiderman parameters).
        :param harmonics: (bool) Whether to include harmonics into the phase curve of the planet. (requires parameters
                          harm_A1, harm_A2, harm_A3, harm_B2 and harm_B3 to be added to the spiderman parameters).
        :param separate_depth: (bool) Whether to separate the eclipse/transit depth parameters into a forward depth and
                               a reverse scanned depth. (also works for the depth of the sine phase curve).
        :param fix_pulse_amp: (bool) Whether to fix the amplitude of the stellar pulsations.
        :param puls_amp: (float) The amplitude to which the stellar pulsations would be fixed.
        :param fix_limb_dark: (bool) Whether to fix the stellar limb darkening. Is automatically disabled if fitting an
                              eclipse.
        :param limb_dark: (string) Either 'linear' or 'quadratic'. The limb darkening law to follow.
        :param fix_inclination: (bool) Whether to fix the orbital inclination
        :param fix_eccentricity: (bool) Whether to fix the orbital eccentricity
        :param fix_aRs: (bool) Whether to fix a/Rs
        :param disable_first_ramp: (bool) Whether to disable the orbit-ramp that is specific to the first orbit-only.
                                     When disabled, the orbit-ramp for the first orbit will take the same form as the
                                     second orbit.
        :param disable_second_ramp: (bool) Whether to disable the orbit-ramp of all orbits except for the first (this
                                    should be used if you're fitting a light curve for which you have divided out the
                                    ramp model already (the divisor should not include the first orbit)
        :param polynomial: (bool) Whether to include a polynomial to the HST systematics baseline
        :param same_Rorb: (bool) Whether to have the same orbit-ramp parameters for forward and reverse scanned data.
        :param nosat: (bool) Whether to ignore any possible satellite crossings
        :param Transit: (bool) Whether to fit a transit or an eclipse.
        :param exc_indices: (list) List of indices you want to exclude in the fitting algorithm. They will still be plot
        :param fitting_method: (string) The fitting algorithm to use. Default: least_squares. For a comprehensive list
                               of options, see https://lmfit.github.io/lmfit-py/fitting.html#the-minimize-function
                               under method. least_squares is the default because this can (by default) give error-
                               estimates to the fitted parameters.
        :param incl_first_orbit: (bool) Whether the first orbit of the visit is included in the data. (default=False)

        :return:
                 - (list) of optimal parameters. Also includes parameters that were not fit, but were kept fixed
                 - (list) of errors on those optimal parameters
                 - (float) chi-squared
                 - (float) Bayesian Information Criterion
        """
        assert Times[1] - Times[0] < 1. and Times[0] > 0., "Please enter Times in BJD"

        self.Forward = Forward
        self.satellite_time = sattime
        self.exptime = exptime
        self.incl_eclipse = True
        self.incl_first_orbit = incl_first_orbit

        #Exclude some data for the fit, but not for plotting
        self.exc_array = np.ones_like(Times, dtype=bool)
        if len(exc_indices) > 0:
            self.exc_array[exc_indices] = False
            self.noexcl = False
        else:
            self.noexcl = True


        #If all data is scanned in the same direction, we need to disable the reverse scanned function
        if np.sum(self.Forward) == 0. or np.sum(self.Forward) == len(Times):
            self.Forward = np.ones_like(Times, dtype=bool)
            max_Forward = np.max(data[self.Forward])
            max_Reverse = np.max(data[self.Forward])
            onlyForward = True
        else:
            max_Forward = np.max(data[self.Forward])
            max_Reverse = np.max(data[~self.Forward])
            onlyForward = False

        if t0 != 0.:  # Set the mid transit time if given.
            self.sp_params.t0 = t0
            m = batman.TransitModel(self.sp_params, Times)
            self.sp_params.t_secondary = m.get_t_secondary(self.sp_params)
        if a != 10.:  #Set a/Rs etc. if given
            self.sp_params.a = a
        if ecc != -1.:
            self.sp_params.ecc = ecc
        if phi != 1000.:
            self.phi = phi
        if inc != 1000.:
            self.sp_params.inc = inc
        self.sp_params.limb_dark = limb_dark
        i0 = self.sp_params.inc

        #If the default limb darkening parameters are set for linear, just have the second limb darkening parameter be 0
        if self.sp_params.limb_dark == "linear":
            self.sp_params.u = [self.sp_params.u[0], 0.]

        self.orbit_times = orbit_times
        self.orbit_times_orig = orbit_times.copy()

        #First fit for t0 and inclination. We'll exclude egress later.
        if Transit:
            if verbose:
                print "Fitting a transit"
            self.func = Func_exp_transit
        else:
            if verbose:
                print "Fitting an eclipse"
            self.func = Func_exp_eclipse

        fit_params_t0 = Parameters()
        #For every parameter: set vary=True if you want to include this parameter in your fit
        if separate_depth and not onlyForward:
            fit_params_t0.add('depth_F', value=0.0016, max=0.1, min=0.0, vary=True)
            fit_params_t0.add('depth_R', value=0.0016, max=0.1, min=0.0, vary=True)
        else:
            fit_params_t0.add('depth_F', value=0.0016, max=0.1, min=0.0, vary=True)
            fit_params_t0.add('depth_R', expr='depth_F')
        if deltat !=0:
            fit_params_t0.add('Mid_transit_time_offset', value=deltat, vary=False)
        else:
            fit_params_t0.add('Mid_transit_time_offset', value=0, max=0.01, min=-0.01)
        if fix_inclination:
            fit_params_t0.add('Inclination', value=i0, vary=False)
        else:
            fit_params_t0.add('Inclination', value=i0, max=i0 + 2., min=i0 - 1.)
        if fix_eccentricity:
            fit_params_t0.add('Eccentricity', value=self.sp_params.ecc, vary=False)
        else:
            fit_params_t0.add('Eccentricity', value=self.sp_params.ecc, max=max(self.sp_params.ecc * 5, 1),
                              min=min(self.sp_params.ecc * 5, 0))
        if fix_aRs:
            fit_params_t0.add('aRs', value=self.sp_params.a, vary=False)
        else:
            fit_params_t0.add('aRs', value=self.sp_params.a, max=self.sp_params.a * 1.2, min=self.sp_params.a * 0.8)
        if fix_limb_dark or not Transit:
            if len(self.sp_params.u) == 1. and not limb_dark == 'linear':
                print "Only one limb darkening parameter was given, reverting to linear limb darkening"
                fit_params_t0.add('u1', value=self.sp_params.u[0], vary=False)
                fit_params_t0.add('u2', value=0., vary=False)
                self.sp_params.limb_dark = 'linear'
            else:
                fit_params_t0.add('u1', value=self.sp_params.u[0], vary=False)
                fit_params_t0.add('u2', value=self.sp_params.u[1], vary=False)
        else:
            fit_params_t0.add('u1', value=self.sp_params.u[0], max=1., min=0.)
            if limb_dark == 'linear':
                fit_params_t0.add('u2', value=0., vary=False)
            else:
                if len(self.sp_params.u) < 2.:
                    u2 = 0
                else:
                    u2 = self.sp_params.u[1]
                fit_params_t0.add('u2', value=u2, max=1., min=0.)
        if sine:
            fit_params_t0.add('Phase_offset', value=0., max=0.3, min=-0.3)
        else:
            fit_params_t0.add('Phase_offset', value=0., vary=False)
        fit_params_t0.add('V2_F', value=-0.003, max=0.5, min=-0.5)
        if onlyForward:
            fit_params_t0.add('V2_R', expr='V2_F')
        else:
            fit_params_t0.add('V2_R', value=-0.003, max=0.5, min=-0.5)
        fit_params_t0.add('Stellar_flux_F', value=max_Forward, max=2. * np.min(data), min=0.5 * np.min(data))
        if onlyForward:
            fit_params_t0.add('Stellar_flux_R', expr='Stellar_flux_F')
        else:
            fit_params_t0.add('Stellar_flux_R', value=max_Reverse, max=2. * np.min(data), min=0.5 * np.min(data))
        if disable_first_ramp:
            fit_params_t0.add('Rorb1_F', value=0., vary=False)
            fit_params_t0.add('Rorb1_R', value=0., vary=False)
        else:
            fit_params_t0.add('Rorb1_F', value=0.0025, max=0.1, min=0.)
            if same_Rorb or onlyForward:
                fit_params_t0.add('Rorb1_R', expr='Rorb1_F')
            else:
                fit_params_t0.add('Rorb1_R', value=0.0025, max=0.1, min=0.)
        if disable_second_ramp:
            fit_params_t0.add('Rorb2_F', value=0., vary=False)
            fit_params_t0.add('Rorb2_R', value=0., vary=False)
        else:
            fit_params_t0.add('Rorb2_F', value=0.002, max=0.1, min=0.)
            if same_Rorb or onlyForward:
                fit_params_t0.add('Rorb2_R', expr='Rorb2_F')
            else:
                fit_params_t0.add('Rorb2_R', value=0.002, max=0.1, min=0.)
        if disable_second_ramp and disable_first_ramp:
            fit_params_t0.add('tau_F', value=0.005, vary=False)
            fit_params_t0.add('tau_R', value=0.005, vary=False)
        else:
            fit_params_t0.add('tau_F', value=0.005, max=0.1, min=0.)
            if same_Rorb or onlyForward:
                fit_params_t0.add('tau_R', expr='tau_F')
            else:
                fit_params_t0.add('tau_R', value=0.005, max=0.1, min=0.)
        if nosat:
            fit_params_t0.add('Satellite_flux', value=0., vary=False)
        else:
            fit_params_t0.add('Satellite_flux', value=0., max=10., min=-2.)
        if sine:
            if separate_depth and not onlyForward:
                fit_params_t0.add('V1_F', value=0.0005, min=0, max=0.1)
                fit_params_t0.add('V1_R', value=0.0005, min=0, max=0.1)
            else:
                fit_params_t0.add('V1_F', value=0.0005, min=0, max=0.1)
                fit_params_t0.add('V1_R', expr='V1_F')
        else:
            fit_params_t0.add('V1_F', value=0., vary=False)
            fit_params_t0.add('V1_R', value=0., vary=False)
        if polynomial:
            fit_params_t0.add('V3_F', value=0., min=-1., max=1.)
            if onlyForward:
                fit_params_t0.add('V3_R', expr='V3_F')
            else:
                fit_params_t0.add('V3_R', value=0., min=-1., max=1.)
        else:
            fit_params_t0.add('V3_F', value=0., vary=False)
            fit_params_t0.add('V3_R', value=0., vary=False)
        if stel_pulse:
            if fix_pulse_amp:
                fit_params_t0.add('Stel_puls_phase', value=0., min=-0.5, max=0.5)
                fit_params_t0.add('Stel_puls_amp', value=puls_amp, vary=False)
            else:
                fit_params_t0.add('Stel_puls_phase', value=0., min=-0.5, max=0.5)
                fit_params_t0.add('Stel_puls_amp', value=puls_amp, min=0., max=2.)
        else:
            fit_params_t0.add('Stel_puls_phase', value=0., vary=False)
            fit_params_t0.add('Stel_puls_amp', value=0., vary=False)
        if harmonics:
            fit_params_t0.add('Harmonics_amplitude', value=1., min=0., max=3.)
        else:
            fit_params_t0.add('Harmonics_amplitude', value=0., vary=False)
        if disable_first_ramp:
            fit_params_t0['Rorb1_F'].expr = 'Rorb2_F'
            fit_params_t0['Rorb1_R'].expr = 'Rorb2_R'

        def residual(pars, Times, data=None):
            pars['V1_R'].max = 0.5 * pars['depth_R'].value
            pars['V1_F'].max = 0.5 * pars['depth_F'].value
            model = self.func(Times, pars['depth_F'], pars['depth_R'],
                                            pars['Mid_transit_time_offset'], pars['Inclination'], pars['Eccentricity'],
                                            pars['aRs'], pars['u1'], pars['u2'], pars['Phase_offset'],
                                            pars['V2_F'], pars['V2_R'],
                                            pars['Stellar_flux_F'], pars['Stellar_flux_R'], pars['Rorb1_F'],
                                            pars['Rorb1_F'], pars['Rorb2_F'], pars['Rorb2_F'], pars['tau_F'],
                                            pars['tau_F'],
                                            pars['Satellite_flux'], pars['V1_F'], pars['V1_R'],
                                            pars['V3_F'], pars['V3_R'],
                                            self.Forward, self.sp_params, self.orbit_times, self.satellite_time,
                                            self.exc_array, self.noexcl,
                                            Stel_puls_phase=pars['Stel_puls_phase'],
                                            Stel_puls_amp=pars['Stel_puls_amp'],
                                            Harm_amp=pars['Harmonics_amplitude'])
            if data is None:
                return model
            if self.noexcl:
                return model - data
            else:
                return model - data[self.exc_array]

        out = minimize(residual, fit_params_t0, args=(Times,), kws={'data': data}, method=fitting_method)
        if verbose:
            report_fit(out, show_correl=False)
        opt2 = [out.params[key].value for key in out.params.keys()]
        err2 = [out.params[key].stderr for key in out.params.keys()]
        chi2 = out.chisqr
        BIC = out.bic

        self.orbit_times = self.orbit_times_orig
        if verbose:
            self._Plot(Times, data, errors, opt2, Transit=Transit)
            self.Forward = Forward



        return opt2, err2, chi2, BIC   #chi2 is on the full data range including egress



    def Fit_with_recte_lmfit(self, Times, data, errors, orbit_times, Forward, exptime=90,
                       verbose=True, t0=0., deltat=0., a=10., ecc=-1., phi=1000., inc=1000., sattime=1.e6,
                       sine=False, stel_pulse=False, harmonics=False,
                       separate_depth=True, fix_pulse_amp=False, puls_amp=1., fix_limb_dark=False, limb_dark='linear',
                       fix_inclination=True, fix_eccentricity=True, fix_aRs=True,
                       polynomial=False, nosat=False, Transit=False, exc_indices=[],
                       incl_first_orbit=False, fitting_method='least_squares'):
        """
        Fit a lightcurve with an RECTE charge trap model and a Levenberg-Marquardt-ish algorithm (using package lmfit)

         The fitting algorithm used here is the python lmfit package (https://lmfit.github.io/lmfit-py/)

        :param Times: (1D numpy array) of the times of exposure in BJD
        :param data: (1D numpy array) of the flux. The out of transit mean should be 1.
        :param errors: (1D numpy array) of errorbars in same units as data
        :param orbit_times: (2D list) List of time ranges (in days) of the orbits such that all start times of an orbit
                           fall into this range. 2 orbit example: orbit_times = [[-1.e-6, 0.066], [0.066, 0.132]]
                           This list should NOT INCLUDE THE FIRST ORBIT if that has been excluded from 'Times' and
                           'data' already.
        :param Forward: (numpy boolarray) Array of True/False on whether an exposure is Forward scanned.
                        This has the same length as 'Times'
        :param exptime: (float) exposure time of each exposure in seconds
        :param verbose: (bool) Whether to print out anything (incl plots)
        :param t0: (float) The mid-transit time of the exoplanet in BJD. If using Transit=False, this needs to be the
                   mid-transit time preceding the eclipse data.
        :param deltat: (float) mid-time transit offset that will be used if that parameter is fixed
        :param a: (float) a/Rs that will be used if that parameter is fixed and a!=10.
        :param ecc: (float) eccentricity that will be used if that parameter is fixed and ecc!=-1.
        :param phi: (float) phase offset in days that will be used if that parameter is fixed and phi!=1000.
        :param inc: (float) orbital inclination in degrees that will be used if that parameter is fixed and inc!=1000.
        :param sattime: The time (BJD) of the exposure in which a satellite crossed. (this time stamp should be in the
                        Times array).
        :param sine: (bool) Whether to include a sine (as phase curve) into the lightcurve of the planet
        :param stel_pulse: (bool) Whether to include stellar pulsations into the light curve of the planet (requires
                           parameters pulse_alpha, pulse_beta and pulse_Pi to be added to the spiderman parameters).
        :param harmonics: (bool) Whether to include harmonics into the phase curve of the planet. (requires parameters
                          harm_A1, harm_A2, harm_A3, harm_B2 and harm_B3 to be added to the spiderman parameters).
        :param separate_depth: (bool) Whether to separate the eclipse/transit depth parameters into a forward depth and
                               a reverse scanned depth. (also works for the depth of the sine phase curve).
        :param fix_pulse_amp: (bool) Whether to fix the amplitude of the stellar pulsations.
        :param puls_amp: (float) The amplitude to which the stellar pulsations would be fixed.
        :param fix_limb_dark: (bool) Whether to fix the stellar limb darkening. Is automatically disabled if fitting an
                              eclipse.
        :param limb_dark: (string) Either 'linear' or 'quadratic'. The limb darkening law to follow.
        :param fix_inclination: (bool) Whether to fix the orbital inclination
        :param fix_eccentricity: (bool) Whether to fix the orbital eccentricity
        :param fix_aRs: (bool) Whether to fix a/Rs
        :param polynomial: (bool) Whether to include a polynomial to the HST systematics baseline
        :param nosat: (bool) Whether to ignore any possible satellite crossings
        :param Transit: (bool) Whether to fit a transit or an eclipse.
        :param exc_indices: (list) List of indices you want to exclude in the fitting algorithm. They will still be plot
        :param fitting_method: (string) The fitting algorithm to use. Default: least_squares. For a comprehensive list
                               of options, see https://lmfit.github.io/lmfit-py/fitting.html#the-minimize-function
                               under method. least_squares is the default because this can (by default) give error-
                               estimates to the fitted parameters.

        :return:
                 - (list) of optimal parameters. Also includes parameters that were not fit, but were kept fixed
                 - (list) of errors on those optimal parameters
                 - (float) chi-squared
                 - (float) Bayesian Information Criterion
        """
        assert Times[1] - Times[0] < 1. and Times[0] > 0., "Please enter Times in BJD"

        self.Forward = Forward
        self.satellite_time = sattime
        self.exptime = exptime
        self.incl_eclipse = True
        self.incl_first_orbit = incl_first_orbit

        #Exclude some data for the fit, but not for plotting
        self.exc_array = np.ones_like(Times, dtype=bool)
        if len(exc_indices) > 0:
            self.exc_array[exc_indices] = False
            self.noexcl = False
        else:
            self.noexcl = True


        #If all data is scanned in the same direction, we need to disable the reverse scanned function
        if np.sum(self.Forward) == 0. or np.sum(self.Forward) == len(Times):
            self.Forward = np.ones_like(Times, dtype=bool)
            max_Forward = np.max(data[self.Forward])
            max_Reverse = np.max(data[self.Forward])
            onlyForward = True
        else:
            max_Forward = np.max(data[self.Forward])
            max_Reverse = np.max(data[~self.Forward])
            onlyForward = False

        if t0 != 0.:  # Set the mid transit time if given.
            self.sp_params.t0 = t0
            m = batman.TransitModel(self.sp_params, Times)
            self.sp_params.t_secondary = m.get_t_secondary(self.sp_params)
        if a != 10.:  #Set a/Rs etc. if given
            self.sp_params.a = a
        if ecc != -1.:
            self.sp_params.ecc = ecc
        if phi != 1000.:
            self.phi = phi
        if inc != 1000.:
            self.sp_params.inc = inc
        self.sp_params.limb_dark = limb_dark
        i0 = self.sp_params.inc

        #If the default limb darkening parameters are set for linear, just have the second limb darkening parameter be 0
        if self.sp_params.limb_dark == "linear":
            self.sp_params.u = [self.sp_params.u[0], 0.]

        self.orbit_times = orbit_times
        self.orbit_times_orig = orbit_times.copy()


        #First fit for t0 and inclination. We'll exclude egress later.
        if Transit:
            if verbose:
                print "Fitting a transit"
            self.func = Func_recte_transit
        else:
            if verbose:
                print "Fitting an eclipse"
            self.func = Func_recte_eclipse

        fit_params_t0 = Parameters()
        #For every parameter: set vary=True if you want to include this parameter in your fit

        if separate_depth and not onlyForward:
            fit_params_t0.add('depth_F', value=0.0016, max=0.1, min=0.0, vary=True)
            fit_params_t0.add('depth_R', value=0.0016, max=0.1, min=0.0, vary=True)
        else:
            fit_params_t0.add('depth_F', value=0.0016, max=0.1, min=0.0, vary=True)
            fit_params_t0.add('depth_R', expr='depth_F')
        if deltat !=0:
            fit_params_t0.add('Mid_transit_time_offset', value=deltat, vary=False)
        else:
            fit_params_t0.add('Mid_transit_time_offset', value=0., max=0.01, min=-0.01)
        if fix_inclination:
            fit_params_t0.add('Inclination', value=i0, vary=False)
        else:
            fit_params_t0.add('Inclination', value=i0, max=i0 + 2., min=i0 - 1.)
        if fix_eccentricity:
            fit_params_t0.add('Eccentricity', value=self.sp_params.ecc, vary=False)
        else:
            fit_params_t0.add('Eccentricity', value=self.sp_params.ecc, max=max(self.sp_params.ecc * 5, 1),
                              min=min(self.sp_params.ecc * 5, 0))
        if fix_aRs:
            fit_params_t0.add('aRs', value=self.sp_params.a, vary=False)
        else:
            fit_params_t0.add('aRs', value=self.sp_params.a, max=self.sp_params.a * 1.2, min=self.sp_params.a * 0.8)
        if fix_limb_dark or not Transit:
            if len(self.sp_params.u) == 1. and not limb_dark == 'linear':
                print "Only one limb darkening parameter was given, reverting to linear limb darkening"
                fit_params_t0.add('u1', value=self.sp_params.u[0], vary=False)
                fit_params_t0.add('u2', value=0., vary=False)
                self.sp_params.limb_dark = 'linear'
            else:
                fit_params_t0.add('u1', value=self.sp_params.u[0], vary=False)
                fit_params_t0.add('u2', value=self.sp_params.u[1], vary=False)
        else:
            fit_params_t0.add('u1', value=self.sp_params.u[0], max=1., min=0.)
            if limb_dark == 'linear':
                fit_params_t0.add('u2', value=0., vary=False)
            else:
                if len(self.sp_params.u) < 2.:
                    u2 = 0
                else:
                    u2 = self.sp_params.u[1]
                fit_params_t0.add('u2', value=u2, max=1., min=0.)
        if sine:
            fit_params_t0.add('Phase_offset', value=0., max=0.3, min=-0.3)
        else:
            fit_params_t0.add('Phase_offset', value=0., vary=False)
        fit_params_t0.add('V2_F', value=-0.003, max=0.5, min=-0.5)
        if onlyForward:
            fit_params_t0.add('V2_R', expr='V2_F')
        else:
            fit_params_t0.add('V2_R', value=-0.003, max=0.5, min=-0.5)
        fit_params_t0.add('Stellar_flux_F', value=max_Forward, max=2. * np.min(data), min=0.5 * np.min(data))
        if onlyForward:
            fit_params_t0.add('Stellar_flux_R', expr='Stellar_flux_F')
        else:
            fit_params_t0.add('Stellar_flux_R', value=max_Reverse, max=2. * np.min(data), min=0.5 * np.min(data))
        fit_params_t0.add('E0_s', value=1000., max=1525.38, min=0.)
        fit_params_t0.add('E0_f', value=100., max=162.38, min=0.)
        fit_params_t0.add('Delta_Es', value=100., max=1525.38, min=0.)
        fit_params_t0.add('Delta_Ef', value=80., max=162.38, min=0.)
        if nosat:
            fit_params_t0.add('Satellite_flux', value=0., vary=False)
        else:
            fit_params_t0.add('Satellite_flux', value=0., max=10., min=-2.)
        if sine:
            if separate_depth and not onlyForward:
                fit_params_t0.add('V1_F', value=0.0005, min=0, max=0.1)
                fit_params_t0.add('V1_R', value=0.0005, min=0, max=0.1)
            else:
                fit_params_t0.add('V1_F', value=0.0005, min=0, max=0.1)
                fit_params_t0.add('V1_R', expr='V1_F')
        else:
            fit_params_t0.add('V1_F', value=0., vary=False)
            fit_params_t0.add('V1_R', value=0., vary=False)
        if polynomial:
            fit_params_t0.add('V3_F', value=0., min=-1., max=1.)
            if onlyForward:
                fit_params_t0.add('V3_R', expr='V3_F')
            else:
                fit_params_t0.add('V3_R', value=0., min=-1., max=1.)
        else:
            fit_params_t0.add('V3_F', value=0., vary=False)
            fit_params_t0.add('V3_R', value=0., vary=False)
        if stel_pulse:
            if fix_pulse_amp:
                fit_params_t0.add('Stel_puls_phase', value=0., min=-0.5, max=0.5)
                fit_params_t0.add('Stel_puls_amp', value=puls_amp, vary=False)
            else:
                fit_params_t0.add('Stel_puls_phase', value=0., min=-0.5, max=0.5)
                fit_params_t0.add('Stel_puls_amp', value=puls_amp, min=0., max=2.)
        else:
            fit_params_t0.add('Stel_puls_phase', value=0., vary=False)
            fit_params_t0.add('Stel_puls_amp', value=0., vary=False)
        if harmonics:
            fit_params_t0.add('Harmonics_amplitude', value=1., min=0., max=3.)
        else:
            fit_params_t0.add('Harmonics_amplitude', value=0., vary=False)


        def residual(pars, Times, data=None):
            pars['V1_R'].max = 0.5 * pars['depth_R'].value
            pars['V1_F'].max = 0.5 * pars['depth_F'].value
            model = self.func(Times, pars['depth_F'], pars['depth_R'],
                                            pars['Mid_transit_time_offset'], pars['Inclination'], pars['Eccentricity'],
                                            pars['aRs'], pars['u1'], pars['u2'], pars['Phase_offset'],
                                            pars['V2_F'], pars['V2_R'],
                                            pars['Stellar_flux_F'], pars['Stellar_flux_R'], pars['E0_s'],
                                            pars['E0_f'], pars['Delta_Es'], pars['Delta_Ef'],
                                            pars['Satellite_flux'], pars['V1_F'], pars['V1_R'],
                                            pars['V3_F'], pars['V3_R'],
                                            self.Forward, self.sp_params, self.orbit_times, self.satellite_time,
                                            self.exc_array, self.exptime ,self.noexcl,
                                            Stel_puls_phase=pars['Stel_puls_phase'],
                                            Stel_puls_amp=pars['Stel_puls_amp'],
                                            Harm_amp=pars['Harmonics_amplitude'])
            if data is None:
                return model
            if self.noexcl:
                return model - data
            else:
                return model - data[self.exc_array]

        out = minimize(residual, fit_params_t0, args=(Times,), kws={'data': data}, method=fitting_method)
        if verbose:
            report_fit(out, show_correl=False)
        opt2 = [out.params[key].value for key in out.params.keys()]
        err2 = [out.params[key].stderr for key in out.params.keys()]
        chi2 = out.chisqr
        BIC = out.bic

        self.orbit_times = self.orbit_times_orig
        if verbose:
            self._Plot(Times, data, errors, opt2, Transit=Transit)
            self.Forward = Forward



        return opt2, err2, chi2, BIC   #chi2 is on the full data range including egress



    def MCMC_with_exponential_lmfit_color(self, Times, data, errors, orbit_times, Forward, exptime=90, bounds=True,
                                   darkening_law='linear', verbose=True, u=0., t0=0., deltat=0., a=10., phi=1000.,
                                   inc=87.2, sattime=1.e6, prev_opt=[], prev_err=[], l1 = 1.1e-6, l2=1.7e-6,
                                   Flux_F=1., Flux_R=1., secondtau=False, sine=True, stel_pulse=False, harmonics=False,
                                   fix_pulse_amp=False, fix_pulse_phas=False, limit_sine=True,
                                   disable_second_ramp=False, polynomial=False, no_t0_inc=False,
                                   stel_pulse_phase=0.,
                                   Rorb1_F=0., Rorb1_R=0., tau_F=0., tau_R=0.,
                                   no_abs_flux=False, same_depth=False, nsteps = 7000, nburn = 1000, same_Rorb=False):
        """
        Fit Kelt-9b's secondary eclipse with an exponential systematics fit and a Levenberg-Marquardt algorithm

        :param Times: in BJD
        :param data: in electron per seconds per pixel
        :param errors:
        :param orbit_times: in BJD
        :param Forward: (np.array of booleans) with len=len(Times). True if exposure is forward scanned
        :param exptime: (float) exposure time of each exposure in seconds
        :param bounds:
        :param verbose:
        :param t0:
        :param tau:
        :param a:
        :param phi:
        :param inc:
        :param sattime: The time (BJD) of the exposure in which the sattelite crossed.
        :param prev_opt:
        :param prev_err:
        :param Firstramp:
        :param l1:
        :param l2:
        :param secondtau:
        :param sine: Whether to include a sine (phase curve) in the fit.
        :param harmonics: Whether to include first, secondary and tertiary harmonics.
        :return:
        """
        assert Times[1] - Times[0] < 1. and Times[0] > 0., "Please enter Times in BJD"

        self.sp_params.l1 = l1  # The starting wavelength in meters
        self.sp_params.l2 = l2  # The ending wavelength in meters
        self.Forward = Forward
        self.satellite_time = sattime
        self.exptime = exptime
        self.incl_eclipse = True
        self.t0_not_yet_set = True

        if t0 != 0.:  # Set the mid transit time if given.
            self.sp_params.t0 = t0
            self.sp_params.t_secondary = self.sp_params.t0 + self.sp_params.per / 2. * \
                                         (1 + 4 / np.pi * self.sp_params.ecc * np.cos(self.sp_params.w* np.pi / 180. ))
        if a != 10.:
            self.sp_params.a = a
        if phi != 1000.:
            self.phi = phi
        if inc != 1000.:
            self.sp_params.inc = inc
        i0 = self.sp_params.inc

        self.orbit_times = orbit_times
        self.orbit_times_orig = orbit_times.copy()
        #self.exc_egress = ~((Times >= self.orbit_times[-2][0]) & (Times <= self.orbit_times[-2][1]))
        Excl_sat = Times != sattime

        Excl_egress = np.ones_like(Excl_sat, dtype=bool)
        Excl_egress[51:62] = False
        self.exc_egress = Excl_egress

        #First fit for t0 and inclination. We'll exclude egress later.
        self.func = Func_exp_eclipse

        self.parameters = {}
        self.parameters['depth_F'] = {'value': 0.0016, 'prior_min':0., 'prior_max':0.1, 'prior': 'uniform',
                                      'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        self.parameters['depth_R'] = {'value': 0.0016, 'prior_min':0., 'prior_max':0.1, 'prior':'uniform',
                                      'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        #self.parameters['deltat'] = {'value': 0.001, 'prior_min':-0.05, 'prior_max':0.05, 'prior': 'uniform',
        #                             'walker_locs':'flat'}
        #self.parameters['inc']= {'value': 87.2, 'prior_min':85.2, 'prior_max':89.2, 'walker_locs':'flat'}
        self.parameters['V2_F'] = {'value': 0., 'prior_min':-0.5, 'prior_max':0.5, 'prior':'uniform',
                                   'walker_locs':'gaussian', 'walker_sigma':0.002, 'sigma_type':'absolute'}
        self.parameters['V2_R'] = {'value': 0., 'prior_min':-0.5, 'prior_max':0.5, 'prior':'uniform',
                                   'walker_locs':'gaussian', 'walker_sigma':0.002, 'sigma_type':'absolute'}
        self.parameters['Stellar_flux_F'] = {'value': 1., 'prior_min':0.9, 'prior_max':1.1, 'prior': 'uniform',
                                             'walker_locs':'gaussian', 'walker_sigma':0.001, 'sigma_type':'absolute'}
        self.parameters['Stellar_flux_R'] = {'value': 1., 'prior_min':0.9, 'prior_max':1.1, 'prior': 'uniform',
                                             'walker_locs':'gaussian', 'walker_sigma':0.001, 'sigma_type':'absolute'}
        #self.parameters['Rorb1_F'] = {'value': 0.0005, 'prior_min':0., 'prior_max':0.1, 'prior':'uniform',
        #                              'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        #self.parameters['Rorb1_R'] = {'value': 0.0005, 'prior_min':0., 'prior_max':0.1, 'prior':'uniform',
        #                              'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        #self.parameters['tau_F'] = {'value': 0.0005, 'prior_min':0., 'prior_max':0.1, 'prior':'uniform',
        #                            'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        #self.parameters['tau_R'] = {'value': 0.0005, 'prior_min':0., 'prior_max':0.1, 'prior':'uniform',
        #                            'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        #self.parameters['Satellite_flux'] = {'value': 0.01, 'prior_min':0., 'prior_max':0.1, 'walker_locs':'flat'}
        #self.parameters['Stel_puls_phase'] = {'value': 0., 'prior_min':-0.5, 'prior_max':0.5, 'prior':'gaussian',
        #                                      'prior_mu':0., 'prior_sigma':0.1, 'walker_locs':'flat'}

        if same_depth:
            ndim, nwalkers = 5, 50  # choose the number of walkers, ndim here is the number of fit parameters
            names = ['depth_F', 'V2_F', 'V2_R', 'Stellar_flux_F', 'Stellar_flux_R']
        else:
            ndim, nwalkers = 6, 60  # choose the number of walkers, ndim here is the number of fit parameters
            names = ['depth_F', 'depth_R', 'V2_F', 'V2_R', 'Stellar_flux_F', 'Stellar_flux_R']

        pos = []
        for name in names:
            if self.parameters[name]['walker_locs'] == 'flat':
                pos.append(np.random.uniform(low=self.parameters[name]['prior_min'],
                                                high=self.parameters[name]['prior_max'], size=nwalkers))
            elif self.parameters[name]['walker_locs'] == 'gaussian':
                if self.parameters[name]['sigma_type'] == 'relative':
                    pos.append(self.parameters[name]['value'] * (np.ones(nwalkers) + np.random.randn(nwalkers) *
                                                                 self.parameters[name]['walker_sigma']))
                elif self.parameters[name]['sigma_type'] == 'absolute':
                    pos.append(self.parameters[name]['value'] + np.random.randn(nwalkers) * self.parameters[name]['walker_sigma'])
                else:
                    print "Warning: incorrect value entred for 'sigma_type' for parameter ", name
            else:
                print "Warning: incorrect value entred for 'walker_locs' for parameter ", name


        pos = np.array(pos).T

        #print ndim, nwalkers
        #print self.parameters.keys()
        if same_Rorb:
            Rorb1_R = Rorb1_F
            tau_R = tau_F
        # Now use emcee to run an MCMC for your setup
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_c, args=(Times, data, no_t0_inc, no_abs_flux, same_depth,
                                                                      self.Forward, self.sp_params, self.orbit_times,
                                                                      self.satellite_time, self.exc_egress,
                                                                      self.t0_not_yet_set, errors,
                                                                      self.parameters, names, deltat, inc, Flux_F, Flux_R,
                                                                      Rorb1_F, Rorb1_R, tau_F, tau_R, stel_pulse_phase, [1.4]),
                                                                threads=8)
        samples = sampler.run_mcmc(pos, nsteps)
        samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
        print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        return samples, sampler


    def MCMC_with_recte_lmfit(self, Times, data, errors, orbit_times, Forward, exptime=90, bounds=True, darkening_law='linear',
                       verbose=True, u=0., t0=0., deltat=0., a=10., phi=1000., inc=87.2, sattime=1.e6, prev_opt=[], prev_err=[],
                       l1 = 1.1e-6, l2=1.7e-6, secondtau=False, sine=True, stel_pulse=False, harmonics=False,
                       fix_pulse_amp=False, limit_sine=True, skip_t0_inc=False, polynomial=False, no_t0_inc=False,
                             no_abs_flux=False, same_depth=False, fix_deltat=False, fix_pulse_phase=False,
                             Stel_puls_phase=0.,
                             nsteps=7000, nburn=1000):
        """
        MCMC Kelt-9b's secondary eclipse with RECTE and a Levenberg-Marquardt algorithm

        :param Times: in BJD
        :param data: in electron per seconds per pixel
        :param errors:
        :param orbit_times: in BJD
        :param Forward: (np.array of booleans) with len=len(Times). True if exposure is forward scanned
        :param exptime: (float) exposure time of each exposure in seconds
        :param bounds:
        :param verbose:
        :param t0:
        :param tau:
        :param a:
        :param phi:
        :param inc:
        :param sattime: The time (BJD) of the exposure in which the sattelite crossed.
        :param prev_opt:
        :param prev_err:
        :param Firstramp:
        :param l1:
        :param l2:
        :param secondtau:
        :param sine: Whether to include a sine (phase curve) in the fit.
        :param harmonics: Whether to include first, secondary and tertiary harmonics.
        :return:
        """
        assert Times[1] - Times[0] < 1. and Times[0] > 0., "Please enter Times in BJD"

        self.sp_params.l1 = l1  # The starting wavelength in meters
        self.sp_params.l2 = l2  # The ending wavelength in meters
        self.Forward = Forward
        self.satellite_time = sattime
        self.exptime = exptime
        self.incl_eclipse = True
        self.t0_not_yet_set = True

        if t0 != 0.:  # Set the mid transit time if given.
            self.sp_params.t0 = t0
            self.sp_params.t_secondary = self.sp_params.t0 + self.sp_params.per / 2. * \
                                         (1 + 4 / np.pi * self.sp_params.ecc * np.cos(self.sp_params.w* np.pi / 180. ))
        if a != 10.:
            self.sp_params.a = a
        if phi != 1000.:
            self.phi = phi
        if inc != 1000.:
            self.sp_params.inc = inc
        i0 = self.sp_params.inc

        self.orbit_times = orbit_times
        self.orbit_times_orig = orbit_times.copy()
        #self.exc_egress = ~((Times >= self.orbit_times[-2][0]) & (Times <= self.orbit_times[-2][1]))
        Excl_sat = Times != sattime

        Excl_egress = np.ones_like(Excl_sat, dtype=bool)
        #Excl_egress[51:62] = False  #This is an example of what you could exclude
        self.exc_egress = Excl_egress

        #First fit for t0 and inclination. We'll exclude egress later.
        self.func = self.Fit_recte_WL_t0_i_noFirst


        self.parameters = {}
        self.parameters['depth_F'] = {'value': 0.0016, 'prior_min':0., 'prior_max':0.1, 'prior':'uniform',
                                      'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        self.parameters['depth_R'] = {'value': 0.0016, 'prior_min':0., 'prior_max':0.1, 'prior':'uniform',
                                      'walker_locs':'gaussian', 'walker_sigma':0.5, 'sigma_type':'relative'}
        self.parameters['deltat'] = {'value': 0.001, 'prior_min':-0.05, 'prior_max':0.05, 'prior':'uniform',
                                    'walker_locs':'flat'}
        #self.parameters['inc']= {'value': 87.2, 'prior_min':85.2, 'prior_max':89.2, 'walker_locs':'flat'}
        self.parameters['V2_F'] = {'value': -0.004, 'prior_min':-0.5, 'prior_max':0.5, 'prior':'uniform',
                                   'walker_locs':'gaussian', 'walker_sigma':0.003, 'sigma_type':'absolute'}
        self.parameters['V2_R'] = {'value': -0.004, 'prior_min':-0.5, 'prior_max':0.5, 'prior':'uniform',
                                   'walker_locs':'gaussian', 'walker_sigma':0.003, 'sigma_type':'absolute'}
        self.parameters['Stellar_flux_F'] = {'value': np.max(data[self.Forward]),
                                             'prior_min':0.9 * np.max(data[self.Forward]),
                                             'prior_max':1.1 * np.max(data[self.Forward]),
                                             'prior':'uniform',
                                             'walker_locs':'gaussian', 'walker_sigma':0.001, 'sigma_type':'relative'}
        self.parameters['Stellar_flux_R'] = {'value': np.max(data[~self.Forward]),
                                             'prior_min':0.9 * np.max(data[~self.Forward]),
                                             'prior_max':1.1 * np.max(data[~self.Forward]),
                                             'prior':'uniform',
                                             'walker_locs':'gaussian', 'walker_sigma':0.001, 'sigma_type':'relative'}
        self.parameters['E0_s'] = {'value': 1000., 'prior_min':0., 'prior_max':1525.38,
                                   'prior':'uniform', 'walker_locs':'gaussian', 'walker_sigma':100.,
                                   'sigma_type':'absolute'}
        self.parameters['E0_f'] = {'value': 100., 'prior_min':0., 'prior_max':162.38, 'prior':'uniform',
                                   'walker_locs':'gaussian', 'walker_sigma':10., 'sigma_type':'absolute'}
        #self.parameters['Delta_Es'] = {'value': 100., 'prior_min':0., 'prior_max':1525.38, 'prior':'uniform',
        #                               'walker_locs':'flat', 'walker_sigma':100., 'sigma_type':'absolute'}
        #self.parameters['Delta_Ef'] = {'value': 100., 'prior_min':0., 'prior_max':162.38, 'prior':'uniform',
        #                               'walker_locs':'flat', 'walker_sigma':10., 'sigma_type':'absolute'}
        self.parameters['Delta_Es'] = {'value': 90., 'prior_min':0., 'prior_max':1525.38, 'prior':'uniform',
                                       'walker_locs':'gaussian', 'walker_sigma':50., 'sigma_type':'absolute'}
        self.parameters['Delta_Ef'] = {'value': 90., 'prior_min':0., 'prior_max':162.38, 'prior':'uniform',
                                       'walker_locs':'gaussian', 'walker_sigma':30., 'sigma_type':'absolute'}
        self.parameters['Satellite_flux'] = {'value': 1., 'prior_min':-2., 'prior_max':20., 'prior':'uniform',
                                             'walker_locs':'flat'}
        self.parameters['Stel_puls_phase'] = {'value': Stel_puls_phase, 'prior_min':-0.5, 'prior_max':0.5, 'prior':'gaussian',
                                              'prior_mu':0., 'prior_sigma':0.05, 'walker_locs':'gaussian', 'walker_sigma':0.1, 'sigma_type':'absolute'}


        pos = []
        if fix_deltat and fix_pulse_phase and same_depth:
            ndim, nwalkers = 10, 100
            names = ['depth_F', 'V2_F', 'V2_R', 'Stellar_flux_F', 'Stellar_flux_R',
                     'E0_s', 'E0_f', 'Delta_Es', 'Delta_Ef', 'Satellite_flux']
        elif fix_deltat and fix_pulse_phase and not same_depth:
            ndim, nwalkers = 11, 100
            names = ['depth_F', 'depth_R', 'V2_F', 'V2_R', 'Stellar_flux_F', 'Stellar_flux_R',
                     'E0_s', 'E0_f', 'Delta_Es', 'Delta_Ef', 'Satellite_flux']
        elif same_depth and not fix_deltat and not fix_pulse_phase:
            ndim, nwalkers = 12, 120
            names = ['depth_F', 'deltat', 'V2_F', 'V2_R', 'Stellar_flux_F', 'Stellar_flux_R',
                     'E0_s', 'E0_f', 'Delta_Es', 'Delta_Ef', 'Satellite_flux', 'Stel_puls_phase']
        else:
            ndim, nwalkers = 13, 130
            names = ['depth_F', 'depth_R', 'deltat', 'V2_F', 'V2_R', 'Stellar_flux_F', 'Stellar_flux_R',
                     'E0_s', 'E0_f', 'Delta_Es', 'Delta_Ef', 'Satellite_flux', 'Stel_puls_phase']
        for name in names:
            if self.parameters[name]['walker_locs'] == 'flat':
                pos.append(np.random.uniform(low=self.parameters[name]['prior_min'],
                                                high=self.parameters[name]['prior_max'], size=nwalkers))
            elif self.parameters[name]['walker_locs'] == 'gaussian':
                if self.parameters[name]['sigma_type'] == 'relative':
                    pos.append(self.parameters[name]['value'] * (np.ones(nwalkers) + np.random.randn(nwalkers) *
                                                                 self.parameters[name]['walker_sigma']))
                elif self.parameters[name]['sigma_type'] == 'absolute':
                    pos.append(self.parameters[name]['value'] + np.random.randn(nwalkers) * self.parameters[name]['walker_sigma'])
                else:
                    print "Warning: incorrect value entred for 'sigma_type' for parameter ", name
            else:
                print "Warning: incorrect value entred for 'walker_locs' for parameter ", name

        pos = np.array(pos).T


        # Now use emcee to run an MCMC for your setup
        with closing(Pool(processes=8)) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(Times, data, no_t0_inc, no_abs_flux, same_depth,
                                                                          False, fix_pulse_phase,
                                                                      self.Forward, self.sp_params, self.orbit_times,
                                                                      self.satellite_time, self.exc_egress,
                                                                      self.t0_not_yet_set, errors,
                                                                      self.parameters, names, deltat, 1., 1.,
                                                                          Stel_puls_phase, [1.4],
                                                                      self.exptime, 'RECTE'),
                                            pool=pool)
                                        #threads=8)
            samples = sampler.run_mcmc(pos, nsteps, progress=True)
            samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
        print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        return samples, sampler



    def _Plot(self, Times, data, errors, opt, ylim_lo=0.99, ylim_hi=1.003, xlim_lo=0., xlim_hi=100.,
              overridebox=False, Transit=False):
        """
        Plot the data with the fit

        :param Times: list: The times of the exposures
        :param data: list: The brightnesses of the exposures
        :param errors: list: The errors on the brightnesses of the exposures
        :param opt: list: list of all variables to be put in the fitted function
        :param ylim_low: float: lower limit of the y-axis in the plot
        :param ylim_hi: float: upper limit of the y-axis in the plot
        :param ylim_low: float: lower limit of the y-axis in the plot
        :param ylim_hi: float: upper limit of the y-axis in the plot
        :param overridebox: bool: Whether to override the above settings on the limits of the viewing box of the plots
        """
        if not overridebox:
            if min(Times) > xlim_hi:
                xlim_hi = max(Times)
            else:
                xlim_hi = min(max(Times), xlim_hi)
            xlim_lo = max(min(Times), xlim_lo)
            ylim_hi = min(max(data + errors), ylim_hi)
            ylim_lo = max(min(data - errors), ylim_lo)

            # Make the plot box slightly larger to view it in its entirety:
            ylim_hi += 0.1 * (ylim_hi - ylim_lo)
            ylim_lo -= 0.1 * (ylim_hi - ylim_lo)
            xlim_hi += 0.1 * (xlim_hi - xlim_lo)
            xlim_lo -= 0.1 * (xlim_hi - xlim_lo)

        fit_opt = opt[:-3]
        expected, baseline_exp, base_eclipse_exp, eclipse_exp = self.func(Times, *fit_opt,
                                                                          Forward=self.Forward,
                                                                          sp_params=self.sp_params,
                                                                          orbit_times=self.orbit_times,
                                                                          satellite_time=self.satellite_time,
                                                                          Include=self.exc_array,
                                                                          exptime=self.exptime,
                                                                          noexcl=self.noexcl,
                                                                          Stel_puls_phase=opt[-3],
                                                                          Stel_puls_amp=opt[-2], Harm_amp=opt[-1],
                                                                          fit=False)
        Residuals = data - expected
        T_new = np.linspace(min(Times), max(Times), 1000)
        if Transit:
            T_new_orbit = T_new.copy()
        else:
            #Plot the full phase curve
            T_new_orbit = np.linspace(self.sp_params.t_secondary - 0.5 * self.sp_params.per, 0.5 * self.sp_params.per + self.sp_params.t_secondary, 1000)
        Forward_old = self.Forward.copy()
        self.Forward = np.ones_like(T_new, dtype=bool)



        fig = plt.figure(figsize=(10, 18))
        plt.subplot(411)
        plt.scatter(Times, expected, color='k', alpha=0.3, label='Fit')
        plt.scatter(Times[Forward_old * self.exc_array], data[Forward_old * self.exc_array], color='b',
                    label='Forward')
        plt.scatter(Times[Forward_old * ~self.exc_array], data[Forward_old * ~self.exc_array], color='b',
                    label='Forward excluded from fit', alpha=0.3)
        plt.scatter(Times[~Forward_old * self.exc_array], data[~Forward_old * self.exc_array], color='g',
                    label='Reverse')
        plt.scatter(Times[~Forward_old * ~self.exc_array], data[~Forward_old * ~self.exc_array], color='g',
                    label='Reverse excluded from fit', alpha=0.3)
        plt.legend()
        datadif = max(data) - min(data)
        plt.gca().set_ylim(min(data) - 0.1 * datadif, max(data) + 0.1 * datadif)

        plt.xlabel('BJD since first observation')
        plt.ylabel('Average electron count per second')
        if len(opt) == 5:
            plt.title('Fit to divide-oot data varying R_p, t_0, u and linear decay')
        if len(opt) == 4:
            plt.title('Fit to divide-oot data varying R_p, t_0 and linear decay')
        if len(opt) == 2:
            plt.title('Fit to divide-oot data varying R_p and t_0')

        plt.gca().set_xlim(xlim_lo, xlim_hi)

        fig.subplots_adjust(hspace=0)

        plt.subplot(412)
        plt.errorbar(Times[Forward_old * self.exc_array], (Residuals[Forward_old * self.exc_array]) / np.max(expected)
                     * 1.e6, yerr=(errors[Forward_old * self.exc_array]) / np.max(expected) * 1.e6, color='b', fmt='o',
                     markersize=2)
        plt.errorbar(Times[Forward_old * ~self.exc_array], (Residuals[Forward_old * ~self.exc_array]) / np.max(expected)
                     * 1.e6, yerr=(errors[Forward_old * ~self.exc_array]) / np.max(expected) * 1.e6, color='b', fmt='o',
                     markersize=2, alpha=0.3)
        plt.errorbar(Times[~Forward_old * self.exc_array], (Residuals[~Forward_old * self.exc_array]) / np.max(expected)
                     * 1.e6, yerr=(errors[~Forward_old * self.exc_array]) / np.max(expected) * 1.e6, color='g', fmt='o',
                     markersize=2)
        plt.errorbar(Times[~Forward_old * ~self.exc_array], (Residuals[~Forward_old * ~self.exc_array]) /
                     np.max(expected) * 1.e6, yerr=(errors[~Forward_old * ~self.exc_array]) / np.max(expected) * 1.e6,
                     color='g', fmt='o', markersize=2, alpha=0.3)

        plt.plot([min(Times), max(Times)], [0, 0], '--')
        plt.ylabel('Residuals (in ppm)')
        plt.xlabel('BJD since first observation')
        plt.gca().set_ylim(1.5 * min(Residuals)/ np.max(expected) * 1.e6, 1.5 * max(Residuals)/ np.max(expected) * 1.e6)
        plt.gca().set_xlim(xlim_lo, xlim_hi)

        if self.incl_eclipse:
            M_total_F, Stellar_F, base_eclipse_F, eclipse_F = self.func(T_new, *fit_opt,
                                                                          Forward=self.Forward,
                                                                          sp_params=self.sp_params,
                                                                          orbit_times=self.orbit_times,
                                                                          satellite_time=self.satellite_time,
                                                                          Include=self.exc_array, noexcl=self.noexcl,
                                                                          exptime=self.exptime,
                                                                          Stel_puls_phase=opt[-3],
                                                                          Stel_puls_amp=opt[-2], Harm_amp=opt[-1],
                                                                          fit=False)
            M_total_F1, Stellar_F1, base_eclipse_F1, eclipse_F1 = self.func(T_new_orbit, *fit_opt,
                                                                          Forward=self.Forward,
                                                                          sp_params=self.sp_params,
                                                                          orbit_times=self.orbit_times,
                                                                          satellite_time=self.satellite_time,
                                                                          Include=self.exc_array, noexcl=self.noexcl,
                                                                          exptime=self.exptime,
                                                                          Stel_puls_phase=opt[-3],
                                                                          Stel_puls_amp=opt[-2], Harm_amp=opt[-1],
                                                                          fit=False)

            self.Forward = np.zeros_like(T_new, dtype=bool)
            M_total_R, Stellar_R, base_eclipse_R, eclipse_R = self.func(T_new, *fit_opt,
                                                                          Forward=self.Forward,
                                                                          sp_params=self.sp_params,
                                                                          orbit_times=self.orbit_times,
                                                                          satellite_time=self.satellite_time,
                                                                          Include=self.exc_array, noexcl=self.noexcl,
                                                                          exptime=self.exptime,
                                                                          Stel_puls_phase=opt[-3],
                                                                          Stel_puls_amp=opt[-2], Harm_amp=opt[-1],
                                                                          fit=False)
            M_total_R1, Stellar_R1, base_eclipse_R1, eclipse_R1 = self.func(T_new_orbit, *fit_opt,
                                                                          Forward=self.Forward,
                                                                          sp_params=self.sp_params,
                                                                          orbit_times=self.orbit_times,
                                                                          satellite_time=self.satellite_time,
                                                                          Include=self.exc_array, noexcl=self.noexcl,
                                                                          exptime=self.exptime,
                                                                          Stel_puls_phase=opt[-3],
                                                                          Stel_puls_amp=opt[-2], Harm_amp=opt[-1],
                                                                          fit=False)

            plt.subplot(413)
            plt.plot(T_new, base_eclipse_F, 'b')
            plt.plot(T_new, base_eclipse_R / np.mean(base_eclipse_R) * np.mean(base_eclipse_F), 'g')
            plt.errorbar(Times[Forward_old], (base_eclipse_exp + np.array(Residuals))[Forward_old],
                         yerr = errors[Forward_old], fmt='o', color='b', markersize=2)
            plt.errorbar(Times[~Forward_old], (base_eclipse_exp / np.mean(base_eclipse_R) * np.mean(base_eclipse_F) +
                                               np.array(Residuals))[~Forward_old],
                         yerr = errors[~Forward_old], fmt = 'o', color = 'g', markersize = 2)
            plt.gca().set_xlim(xlim_lo, xlim_hi)
            plt.gca().set_ylim(min(base_eclipse_F) - 0.1 * datadif, max(base_eclipse_F) + 0.1 * datadif)

            plt.subplot(414)

            plt.plot(T_new_orbit, Stellar_F1, 'k', alpha=0.5, label='Stellar')
            plt.plot(T_new_orbit, eclipse_F1, 'b', label='Forward planet signal')
            plt.plot(T_new_orbit, eclipse_R1, 'g', label='Reverse planet signal')
            plt.gca().set_xlim(min(T_new_orbit), max(T_new_orbit))
            plt.gca().set_ylim(1. - opt[0] * 1.5, 1. + opt[0]*1.5)
            plt.legend()
            plt.show()

    def _Plot_MCMC(self, Times, data, errors, samples, oth, ylim_lo=0.99, ylim_hi=1.003, xlim_lo=0., xlim_hi=100.,
                   no_t0_inc=False, no_abs_flux=False, same_depth=False, overridebox=False):
        """
        Plot the data with the fit

        :param Times: list: The times of the exposures
        :param data: list: The brightnesses of the exposures
        :param errors: list: The errors on the brightnesses of the exposures
        :param opt: list: list of all variables to be put in the fitted function
        :param ylim_low: float: lower limit of the y-axis in the plot
        :param ylim_hi: float: upper limit of the y-axis in the plot
        :param ylim_low: float: lower limit of the y-axis in the plot
        :param ylim_hi: float: upper limit of the y-axis in the plot
        :param overridebox: bool: Whether to override the above settings on the limits of the viewing box of the plots
        """
        inds = np.random.randint(len(samples), size=100)
        opts = []
        if no_abs_flux and same_depth:
            for ind in inds:
                sample = samples[ind]
                opts.append([sample[0], sample[0], sample[1], oth[0], oth[1],
                             sample[2], sample[3],
                             oth[2], oth[3], sample[4],
                             sample[5], oth[4], oth[5], sample[6], sample[7],
                             oth[6], oth[7], oth[8], oth[9], oth[10],
                             sample[8], oth[11], oth[12]])

            print "using median"
            OPT = np.median(samples, axis=0)
            optimal = [OPT[0], OPT[0], OPT[1], oth[0], oth[1],
                       OPT[2], OPT[3],
                       oth[2], oth[3], OPT[4],
                       OPT[5], oth[4], oth[5], OPT[6], OPT[7],
                       oth[6], oth[7], oth[8], oth[9], oth[10],
                       OPT[8], oth[11], oth[12]]
        elif not no_abs_flux and same_depth:
            for ind in inds:
                sample = samples[ind]
                opts.append([sample[0], sample[0], sample[1], oth[0], oth[1],
                             sample[2], sample[3],
                             sample[4], sample[5], sample[6],
                             sample[7], oth[2], oth[3], sample[8], sample[9],
                             oth[4], oth[5], oth[6], oth[7], oth[8],
                             sample[10], oth[9], oth[10]])

            print "using median"
            OPT = np.median(samples, axis=0)
            optimal = [OPT[0], OPT[0], OPT[1], oth[0], oth[1],
                       OPT[2], OPT[3],
                       OPT[4], OPT[5], OPT[6],
                       OPT[7], oth[2], oth[3], OPT[8], OPT[9],
                       oth[4], oth[5], oth[6], oth[7], oth[8],
                       OPT[10], oth[9], oth[10]]
        elif no_abs_flux and not same_depth:
            for ind in inds:
                sample = samples[ind]
                opts.append([sample[0], sample[1], sample[2], oth[0], oth[1],
                             sample[3], sample[4],
                             oth[2], oth[3], sample[5],
                             sample[6], oth[4], oth[5], sample[7], sample[8],
                             oth[6], oth[7], oth[8], oth[9], oth[10],
                             sample[9], oth[11], oth[12]])

            print "using median"
            OPT = np.median(samples, axis=0)
            optimal = [OPT[0], OPT[1], OPT[2], oth[0], oth[1],
                       OPT[3], OPT[4],
                       oth[2], oth[3], OPT[5],
                       OPT[6], oth[4], oth[5], OPT[7], OPT[8],
                       oth[6], oth[7], oth[8], oth[9], oth[10],
                       OPT[9], oth[11], oth[12]]
        else:
            for ind in inds:
                sample = samples[ind]
                opts.append([sample[0], sample[1], sample[2], oth[0], oth[1],
                             sample[3], sample[4],
                             sample[5], sample[6], sample[7],
                             sample[8], oth[2], oth[3], sample[9], sample[10],
                             oth[4], oth[5], oth[6], oth[7], oth[8],
                             sample[11], oth[9], oth[10]])

            print "using median"
            OPT = np.median(samples, axis=0)
            optimal = [OPT[0], OPT[1], OPT[2], oth[0], oth[1],
                       OPT[3], OPT[4],
                       OPT[5], OPT[6], OPT[7],
                       OPT[8], oth[2], oth[3], OPT[9], OPT[10],
                       oth[4], oth[5], oth[6], oth[7], oth[8],
                       OPT[11], oth[9], oth[10]]

        if not overridebox:
            if min(Times) > xlim_hi:
                xlim_hi = max(Times)
            else:
                xlim_hi = min(max(Times), xlim_hi)
            xlim_lo = max(min(Times), xlim_lo)
            ylim_hi = min(max(data + errors), ylim_hi)
            ylim_lo = max(min(data - errors), ylim_lo)

            # Make the plot box slightly larger to view it in its entirety:
            ylim_hi += 0.1 * (ylim_hi - ylim_lo)
            ylim_lo -= 0.1 * (ylim_hi - ylim_lo)
            xlim_hi += 0.1 * (xlim_hi - xlim_lo)
            xlim_lo -= 0.1 * (xlim_hi - xlim_lo)

        expected, baseline_exp, base_eclipse_exp, eclipse_exp = self.func(Times, *optimal, fit=False)
        Residuals = data - expected
        T_new = np.linspace(min(Times), max(Times), 1000)
        T_new_orbit = np.linspace(self.sp_params.t_secondary - 0.5 * self.sp_params.per,
                                  0.5 * self.sp_params.per + self.sp_params.t_secondary, 1000)
        Forward_old = self.Forward.copy()
        self.Forward = np.ones_like(T_new, dtype=bool)

        fig = plt.figure(figsize=(10, 24))
        plt.subplot(611)
        plt.scatter(Times, expected, color='k', alpha=0.3, label='Fit')
        plt.scatter(Times[Forward_old], data[Forward_old], color='b', label='Forward')
        plt.scatter(Times[~Forward_old], data[~Forward_old], color='g', label='Reverse')
        plt.legend()
        datadif = max(data) - min(data)
        plt.gca().set_ylim(min(data) - 0.1 * datadif, max(data) + 0.1 * datadif)

        plt.xlabel('BJD since first observation')
        plt.ylabel('Average electron count per second')
        plt.gca().set_xlim(xlim_lo, xlim_hi)

        fig.subplots_adjust(hspace=0)

        plt.subplot(612)
        plt.errorbar(Times[Forward_old], (Residuals[Forward_old]) / np.max(expected) * 1.e6,
                     yerr=(errors[Forward_old]) / np.max(expected) * 1.e6, color='b', fmt='o', markersize=2)
        plt.errorbar(Times[~Forward_old], (Residuals[~Forward_old]) / np.max(expected) * 1.e6,
                     yerr=(errors[~Forward_old]) / np.max(expected) * 1.e6, color='g', fmt='o', markersize=2)

        plt.plot([min(Times), max(Times)], [0, 0], '--')
        plt.ylabel('Residuals (in ppm)')
        plt.xlabel('BJD since first observation')
        plt.gca().set_ylim(1.5 * min(Residuals) / np.max(expected) * 1.e6,
                           1.5 * max(Residuals) / np.max(expected) * 1.e6)
        plt.gca().set_xlim(xlim_lo, xlim_hi)

        if self.incl_eclipse:
            plt.subplot(613)
            Stellar_F1s = []
            eclipse_F1s = []
            for opt in opts:
                M_total_F, Stellar_F, base_eclipse_F, eclipse_F = self.func(T_new, *opt, fit=False)
                M_total_F1, Stellar_F1, base_eclipse_F1, eclipse_F1 = self.func(T_new_orbit, *opt, fit=False)
                self.Forward = np.zeros_like(T_new, dtype=bool)

                #plt.plot(T_new, base_eclipse_F, 'b', alpha=0.05)
                plt.plot(T_new, M_total_F, 'b', alpha=0.05)
                Stellar_F1s.append(Stellar_F1)
                eclipse_F1s.append(eclipse_F1)
            #plt.errorbar(Times[Forward_old], (base_eclipse_exp + np.array(Residuals))[Forward_old],
            #             yerr=errors[Forward_old], fmt='o', color='b', markersize=2)
            plt.errorbar(Times[Forward_old], data[Forward_old],
                         yerr=errors[Forward_old], fmt='o', color='b', markersize=2)
            plt.gca().set_xlim(xlim_lo, xlim_hi)
            plt.gca().set_ylim(min(base_eclipse_F) - 0.1 * datadif, max(base_eclipse_F) + 0.1 * datadif)

            plt.subplot(614)
            eclipse_R1s = []
            for opt in opts:
                self.Forward = np.zeros_like(T_new, dtype=bool)
                M_total_R, Stellar_R, base_eclipse_R, eclipse_R = self.func(T_new, *opt, fit=False)
                M_total_R1, Stellar_R1, base_eclipse_R1, eclipse_R1 = self.func(T_new_orbit, *opt, fit=False)

                #plt.plot(T_new, base_eclipse_R / np.mean(base_eclipse_R) * np.mean(base_eclipse_F), 'g', alpha=0.05)
                plt.plot(T_new, M_total_R, 'g', alpha=0.05)
                eclipse_R1s.append(eclipse_R1)
            #plt.errorbar(Times[~Forward_old], (base_eclipse_exp / np.mean(base_eclipse_R) * np.mean(base_eclipse_F) +
            #                                   np.array(Residuals))[~Forward_old],
            #             yerr=errors[~Forward_old], fmt='o', color='g', markersize=2)
            plt.errorbar(Times[~Forward_old], data[~Forward_old],
                         yerr=errors[~Forward_old], fmt='o', color='g', markersize=2)
            plt.gca().set_xlim(xlim_lo, xlim_hi)
            plt.gca().set_ylim(min(base_eclipse_F) - 0.1 * datadif, max(base_eclipse_F) + 0.1 * datadif)

            plt.subplot(615)

            for i in range(len(eclipse_F1s) - 1):
                plt.plot(T_new_orbit, Stellar_F1s[i], 'k', alpha=0.025)
                plt.plot(T_new_orbit, eclipse_F1s[i], 'b', alpha=0.05)
            plt.plot(T_new_orbit, Stellar_F1s[-1], 'k', alpha=0.025)
            plt.plot(T_new_orbit, eclipse_F1s[-1], 'b', alpha=0.05)
            plt.gca().set_xlim(self.sp_params.t_secondary - 0.5 * self.sp_params.per,
                               0.5 * self.sp_params.per + self.sp_params.t_secondary)
            plt.gca().set_ylim(1. - opt[0] * 1.5, 1. + opt[0] * 1.5)

            plt.subplot(616)
            for i in range(len(eclipse_F1s) - 1):
                plt.plot(T_new_orbit, Stellar_F1s[i], 'k', alpha=0.025)
                plt.plot(T_new_orbit, eclipse_R1s[i], 'g', alpha=0.05)
            plt.plot(T_new_orbit, eclipse_R1s[-1], 'g', alpha=0.05)
            plt.gca().set_xlim(self.sp_params.t_secondary - 0.5 * self.sp_params.per,
                               0.5 * self.sp_params.per + self.sp_params.t_secondary)
            plt.gca().set_ylim(1. - opt[0] * 1.5, 1. + opt[0] * 1.5)
            plt.show()

    def _Plot_MCMC_recte(self, Times, data, errors, samples, oth, ylim_lo=0.99, ylim_hi=1.003, xlim_lo=0., xlim_hi=100.,
                   no_t0_inc=False, no_abs_flux=False, same_depth=False, overridebox=False):
        """
        Plot the data with the fit

        :param Times: list: The times of the exposures
        :param data: list: The brightnesses of the exposures
        :param errors: list: The errors on the brightnesses of the exposures
        :param opt: list: list of all variables to be put in the fitted function
        :param ylim_low: float: lower limit of the y-axis in the plot
        :param ylim_hi: float: upper limit of the y-axis in the plot
        :param ylim_low: float: lower limit of the y-axis in the plot
        :param ylim_hi: float: upper limit of the y-axis in the plot
        :param overridebox: bool: Whether to override the above settings on the limits of the viewing box of the plots
        """
        inds = np.random.randint(len(samples), size=100)
        opts = []

        if no_abs_flux and same_depth:
            print "adjust the inputs below by copying and adjusting them from 'not no abs_flux and same_depth'"
            for ind in inds:
                sample = samples[ind]
                opts.append([sample[0], sample[0], sample[1], oth[0], oth[1],
                             sample[2], sample[3],
                             oth[2], oth[3], sample[4],
                             sample[5], oth[4], oth[5], sample[6], sample[7],
                             oth[6], oth[7], oth[8], oth[9], oth[10],
                             sample[8], oth[11], oth[12]])

            print "using median"
            OPT = np.median(samples, axis=0)
            optimal = [OPT[0], OPT[0], OPT[1], oth[0], oth[1],
                       OPT[2], OPT[3],
                       oth[2], oth[3], OPT[4],
                       OPT[5], oth[4], oth[5], OPT[6], OPT[7],
                       oth[6], oth[7], oth[8], oth[9], oth[10],
                       OPT[8], oth[11], oth[12]]

        elif not no_abs_flux and same_depth:
            for ind in inds:
                sample = samples[ind]
                opts.append([sample[0], sample[0], sample[1], oth[0], oth[1],
                             sample[2], sample[3],
                             sample[4], sample[5], sample[6],
                             sample[7], sample[8], sample[9], sample[10],
                             oth[4], oth[5], oth[6], oth[7],
                             sample[11], oth[8], oth[9]])

            print "using median"
            OPT = np.median(samples, axis=0)
            optimal = [OPT[0], OPT[0], OPT[1], oth[0], oth[1],
                       OPT[2], OPT[3],
                       OPT[4], OPT[5], OPT[6],
                       OPT[7], OPT[8], OPT[9], OPT[10],
                       oth[4], oth[5], oth[6], oth[7],
                       OPT[11], oth[8], oth[9]]
        elif no_abs_flux and not same_depth:
            for ind in inds:
                sample = samples[ind]
                opts.append([sample[0], sample[1], sample[2], oth[0], oth[1],
                             sample[3], sample[4],
                             oth[2], oth[3], sample[5],
                             sample[6], oth[4], oth[5], sample[7], sample[8],
                             oth[6], oth[7], oth[8], oth[9], oth[10],
                             sample[9], oth[11], oth[12]])

            print "using median"
            OPT = np.median(samples, axis=0)
            optimal = [OPT[0], OPT[1], OPT[2], oth[0], oth[1],
                       OPT[3], OPT[4],
                       oth[2], oth[3], OPT[5],
                       OPT[6], oth[4], oth[5], OPT[7], OPT[8],
                       oth[6], oth[7], oth[8], oth[9], oth[10],
                       OPT[9], oth[11], oth[12]]
        else:
            for ind in inds:
                sample = samples[ind]
                opts.append([sample[0], sample[1], sample[2], oth[0], oth[1],
                             sample[3], sample[4],
                             sample[5], sample[6], sample[7],
                             sample[8], sample[9], sample[10], sample[11],
                             oth[2], oth[3], oth[4], oth[5],
                             sample[12], oth[6], oth[7]])

            print "using median"
            OPT = np.median(samples, axis=0)
            optimal = [OPT[0], OPT[1], OPT[2], oth[0], oth[1],
                       OPT[3], OPT[4],
                       OPT[5], OPT[6], OPT[7],
                       OPT[8], OPT[9], OPT[10], OPT[11],
                       oth[2], oth[3], oth[4], oth[5],
                       OPT[12], oth[6], oth[7]]

        if not overridebox:
            if min(Times) > xlim_hi:
                xlim_hi = max(Times)
            else:
                xlim_hi = min(max(Times), xlim_hi)
            xlim_lo = max(min(Times), xlim_lo)
            ylim_hi = min(max(data + errors), ylim_hi)
            ylim_lo = max(min(data - errors), ylim_lo)

            # Make the plot box slightly larger to view it in its entirety:
            ylim_hi += 0.1 * (ylim_hi - ylim_lo)
            ylim_lo -= 0.1 * (ylim_hi - ylim_lo)
            xlim_hi += 0.1 * (xlim_hi - xlim_lo)
            xlim_lo -= 0.1 * (xlim_hi - xlim_lo)

        expected, baseline_exp, base_eclipse_exp, eclipse_exp = self.func(Times, *optimal, fit=False)
        Residuals = data - expected
        T_new = np.linspace(min(Times), max(Times), 1000)
        T_new_orbit = np.linspace(self.sp_params.t_secondary - 0.5 * self.sp_params.per,
                                  0.5 * self.sp_params.per + self.sp_params.t_secondary, 1000)
        Forward_old = self.Forward.copy()
        self.Forward = np.ones_like(T_new, dtype=bool)

        fig = plt.figure(figsize=(10, 24))
        plt.subplot(611)
        plt.scatter(Times, expected, color='k', alpha=0.3, label='Fit')
        plt.scatter(Times[Forward_old], data[Forward_old], color='b', label='Forward')
        plt.scatter(Times[~Forward_old], data[~Forward_old], color='g', label='Reverse')
        plt.legend()
        datadif = max(data) - min(data)
        plt.gca().set_ylim(min(data) - 0.1 * datadif, max(data) + 0.1 * datadif)

        plt.xlabel('BJD since first observation')
        plt.ylabel('Average electron count per second')
        plt.gca().set_xlim(xlim_lo, xlim_hi)

        fig.subplots_adjust(hspace=0)

        plt.subplot(612)
        plt.errorbar(Times[Forward_old], (Residuals[Forward_old]) / np.max(expected) * 1.e6,
                     yerr=(errors[Forward_old]) / np.max(expected) * 1.e6, color='b', fmt='o', markersize=2)
        plt.errorbar(Times[~Forward_old], (Residuals[~Forward_old]) / np.max(expected) * 1.e6,
                     yerr=(errors[~Forward_old]) / np.max(expected) * 1.e6, color='g', fmt='o', markersize=2)

        plt.plot([min(Times), max(Times)], [0, 0], '--')
        plt.ylabel('Residuals (in ppm)')
        plt.xlabel('BJD since first observation')
        plt.gca().set_ylim(1.5 * min(Residuals) / np.max(expected) * 1.e6,
                           1.5 * max(Residuals) / np.max(expected) * 1.e6)
        plt.gca().set_xlim(xlim_lo, xlim_hi)

        if self.incl_eclipse:
            plt.subplot(613)
            Stellar_F1s = []
            eclipse_F1s = []
            for opt in opts:
                M_total_F, Stellar_F, base_eclipse_F, eclipse_F = self.func(T_new, *opt, fit=False)
                M_total_F1, Stellar_F1, base_eclipse_F1, eclipse_F1 = self.func(T_new_orbit, *opt, fit=False)
                self.Forward = np.zeros_like(T_new, dtype=bool)

                plt.plot(T_new, base_eclipse_F, 'b', alpha=0.05)
                Stellar_F1s.append(Stellar_F1)
                eclipse_F1s.append(eclipse_F1)
            plt.errorbar(Times[Forward_old], (base_eclipse_exp + np.array(Residuals))[Forward_old],
                         yerr=errors[Forward_old], fmt='o', color='b', markersize=2)
            plt.gca().set_xlim(xlim_lo, xlim_hi)
            plt.gca().set_ylim(min(base_eclipse_F) - 0.1 * datadif, max(base_eclipse_F) + 0.1 * datadif)

            plt.subplot(614)
            eclipse_R1s = []
            for opt in opts:
                self.Forward = np.zeros_like(T_new, dtype=bool)
                M_total_R, Stellar_R, base_eclipse_R, eclipse_R = self.func(T_new, *opt, fit=False)
                M_total_R1, Stellar_R1, base_eclipse_R1, eclipse_R1 = self.func(T_new_orbit, *opt, fit=False)

                plt.plot(T_new, base_eclipse_R / np.mean(base_eclipse_R) * np.mean(base_eclipse_F), 'g', alpha=0.05)
                eclipse_R1s.append(eclipse_R1)
            plt.errorbar(Times[~Forward_old], (base_eclipse_exp / np.mean(base_eclipse_R) * np.mean(base_eclipse_F) +
                                               np.array(Residuals))[~Forward_old],
                         yerr=errors[~Forward_old], fmt='o', color='g', markersize=2)
            plt.gca().set_xlim(xlim_lo, xlim_hi)
            plt.gca().set_ylim(min(base_eclipse_F) - 0.1 * datadif, max(base_eclipse_F) + 0.1 * datadif)

            plt.subplot(615)

            for i in range(len(eclipse_F1s) - 1):
                plt.plot(T_new_orbit, Stellar_F1s[i], 'k', alpha=0.025)
                plt.plot(T_new_orbit, eclipse_F1s[i], 'b', alpha=0.05)
            plt.plot(T_new_orbit, Stellar_F1s[-1], 'k', alpha=0.025)
            plt.plot(T_new_orbit, eclipse_F1s[-1], 'b', alpha=0.05)
            plt.gca().set_xlim(self.sp_params.t_secondary - 0.5 * self.sp_params.per,
                               0.5 * self.sp_params.per + self.sp_params.t_secondary)
            plt.gca().set_ylim(1. - opt[0] * 1.5, 1. + opt[0] * 1.5)

            plt.subplot(616)
            for i in range(len(eclipse_F1s) - 1):
                plt.plot(T_new_orbit, Stellar_F1s[i], 'k', alpha=0.025)
                plt.plot(T_new_orbit, eclipse_R1s[i], 'g', alpha=0.05)
            plt.plot(T_new_orbit, eclipse_R1s[-1], 'g', alpha=0.05)
            plt.gca().set_xlim(self.sp_params.t_secondary - 0.5 * self.sp_params.per,
                               0.5 * self.sp_params.per + self.sp_params.t_secondary)
            plt.gca().set_ylim(1. - opt[0] * 1.5, 1. + opt[0] * 1.5)
            plt.show()