import numpy as np
import astropy.io.fits as pyfits
import my_fns as f
import pylab as p
import data

view = data.view_frame_image
import logging
import os
from itertools import combinations_with_replacement as P_iter
from Telescope_characteristics import HST


def twoD_products_simple(x, y, m):
    """ Effectively a binomial expansion?"""
    permute_arrays = [1]
    for n in range(m)[1:]:  # order-1
        for i in range(n + 1):
            permute_arrays.append(x ** (n - i) * y ** i)
    return permute_arrays


def field_dep_coeff(m, coeffs, x, y, permute_arrays=None):
    """
    Evaluates a function at all field positions subject to the coefficients provided.
    Input the order of the field function and the coefficients.
    x and y are the size of the field in the x and y directions
    """
    # assert hasattr(coeffs, '__iter__'), 'Coeffs should be an iterable, even if it contains a single value'
    # assert m**2/2. + m/2. == len(coeffs), 'Incorrect number of coefficients for order {}'.format(m)
    ic = 0
    field_coeffs = coeffs[ic]
    for n in range(m)[1:]:  # order-1
        for i in range(n + 1):
            ic += 1
            field_coeffs += x ** (n - i) * y ** i * coeffs[ic]
    return field_coeffs


def center_of_flux(filename, x, y, size):
    # Find the peak of a psf in a small region
    # Provides correction term to SExtractor
    exp = data.load(filename)
    try:
        image = exp.SCI.data.copy()
    except AttributeError:
        image = exp.reads[0].SCI.data.copy()
    image = image[y - size:y + size, x - size:x + size]
    tot_flux = np.sum(image)
    cof_x = np.sum([row * i for i, row in enumerate(image.T)]) / tot_flux - size
    cof_y = np.sum([col * i for i, col in enumerate(image)]) / tot_flux - size
    return x + cof_x, y + cof_y


def calc_poly_order(coeffs):
    ''' m^2/2 + m/2 = len(coeffs) '''
    c = -len(coeffs)
    m = -0.5 + np.sqrt(0.25 - 2 * c)
    return int(m)


def disp_poly(conf_file, catalogue, exp_time, scan_rate, scan_direction, n='A', x_len=256, y_len=256, XOFF=0, YOFF=0,
              data_dir='/', debug=False, log=False, pix_size=HST().xscale, original_image=None, image_name='',
              disp_coef='default', object_ind=0, x=None, y=None):
    '''
    Read in configuration file and evaluate the dispersion solution at each field value.
    n is the beam  order ('A':first, 'B':zeroth, 'C':second etc...)
    x_len and y_len define size of field
    (XREF, YREF) is the position of the reference pixel
    use XOFF, YOFF to offset the source reference pixel for scan or such

    print "disp_poly args"
    print "conf_file", conf_file
    print "catalogue", catalogue
    print "exp_time", exp_time
    print "scan_rate", scan_rate
    print "scan_direction", scan_direction
    print "n", n
    print "x_len", x_len
    print "y_len", y_len
    print "XOFF", XOFF
    print "Yoff", YOFF
    print "data_dir", data_dir
    print "debug", debug
    print "log", log
    print "pix_size", pix_size
    print "original image", original_image
    print "image_name", image_name
    print "disp_coeff", disp_coef
    print "object_ind", object_ind
    print "x",x
    print "y", y
    '''
    # Read in source information, assumed brightest source is the target
    # Need to use the associated direct image catalogue
    assert os.path.isfile(catalogue), 'Catalogue file does not exist: {}'.format(catalogue)
    if x is None or y is None:
        with open(catalogue, 'r') as cat:
            lines = cat.readlines()
            objects = [line[:-1].split() for line in lines if line[0] != '#']
            objects = [[float(val) for val in obj] for obj in objects]
            objects = sorted(objects, key=lambda obj:obj[-1])
            obj = objects[object_ind]
            # obj = min(objects, key=lambda obj: obj[-1]) # take brightest mag

            image_fname = data_dir + catalogue.split('_')[-3].split('/')[-1] + '_flt.fits'
            SEx, SEy = obj[1], obj[2]
            # Correction
            x, y = SEx, SEy
            # x, y = center_of_flux(image_fname, int(SEx), int(SEy), size=10)

            if debug:
                print 'Direct image location:', x, y
                print 'Offset from SExtractor:', x - SEx, y - SEy
                print 'External offset and scan:', XOFF, YOFF
                print 'Final:', x + XOFF, y + YOFF
            XREF, YREF = x + XOFF, y + YOFF  # including offsets
    else:
        XREF, YREF = x + XOFF, y + YOFF

    with open(conf_file) as cf:
        lines = cf.readlines()
        lines = [line[:-1] for line in lines if line[0] != '#']
    DISP_COEFFS, TRACE_COEFFS = [], []
    for line in lines:
        if line.startswith('BEAM' + str(n)):
            BEAM_X1, BEAM_X2 = [int(pos) for pos in line.split()[1:]]
        elif line.startswith('DISP_ORDER_' + str(n)):
            DISP_ORDER = int(line.split(' ')[1])
        elif line.startswith('DYDX_ORDER_' + str(n)):
            TRACE_ORDER = int(line.split(' ')[1])
        elif line.startswith('DLDP_' + str(n)):
            nth_coeffs = line.split(' ')[1:]
            nth_coeffs = [float(coeff) for coeff in nth_coeffs if coeff != '']
            DISP_COEFFS.append(nth_coeffs)
        elif line.startswith('DYDX_' + str(n)):
            nth_coeffs = line.split(' ')[1:]
            nth_coeffs = [float(coeff) for coeff in nth_coeffs if coeff != '']
            TRACE_COEFFS.append(nth_coeffs)
    assert len(DISP_COEFFS) == DISP_ORDER + 1, 'Coefficients and order do not match for dispersion.'
    assert len(TRACE_COEFFS) == TRACE_ORDER + 1, 'Coefficients and order do not match for trace.'

    if disp_coef == 'wilkins':
        DISP_COEFFS[0][0] = 0.997 * DISP_COEFFS[0][0]
        DISP_COEFFS[0][1] = 0.90 * DISP_COEFFS[0][1]
        DISP_COEFFS[1][0] = 1.029 * DISP_COEFFS[1][0]
    elif disp_coef == 'custom':
        DISP_COEFFS[0][0] = 0.997 * DISP_COEFFS[0][0]
        DISP_COEFFS[0][1] = 0.90 * DISP_COEFFS[0][1]
        DISP_COEFFS[1][0] = 1.02 * DISP_COEFFS[1][0]
    elif disp_coef == 'default':
        pass

    # Make array of positions
    x = np.arange(x_len)
    y = np.arange(y_len)
    x = np.repeat(x, y_len).reshape(x_len, y_len).transpose()
    y = np.repeat(y, x_len).reshape(y_len, x_len)
    # Scale to ref pixel (inc. offsets)
    X = x - XREF  # x wrt to (0,0), X wrt to (XREF,YREF)
    Y = np.zeros_like(y)
    # All broken below, simplified with just y=0
    '''
    Y0 = y - YREF # Scan start
    view(Y0)
    # This is if the direct image didnt move, but it does, so account for it:
    BEAM_H = exp_time*scan_rate/pix_size*scan_direction # Height of beam w.r.t. to 0th direct image
    Y1 = y - YREF - BEAM_H # Scan end
    YM = np.zeros_like(Y0)
    if scan_direction>0: 
        Y = np.vstack([ Y0[:int(YREF)], YM[int(YREF):int(YREF+BEAM_H)], Y1[int(YREF+BEAM_H):] ])
    elif scan_direction<0: 
        Y = np.vstack([ Y1[:int(YREF+BEAM_H)], YM[int(YREF+BEAM_H):int(YREF)], Y0[int(YREF):] ])
    '''

    # Evaluate polynomial
    def function(x_trace, x, y, ALL_COEFFS, plot_coeffs=False, debug=False):
        # x and y are the positions X, Y in the field dep. description
        # x_trace is then the distance in final polynomial along the BEAM or TRACE
        # ALL_COEFFS is a list containing the coefficients to describe each of the field
        #   dependent coefficients in the x_trace polynomial
        total_poly = np.zeros_like(x)
        for i, coeffs in enumerate(ALL_COEFFS):
            m = calc_poly_order(coeffs)
            field_coeff = field_dep_coeff(m, coeffs, x, y)
            if plot_coeffs:
                view(field_coeff, title='Coefficient {}'.format(i))
            if debug: print '{:.4g} * X ^ {:}'.format(np.mean(field_coeff), i)
            total_poly = total_poly + np.multiply(np.power(x_trace, i), field_coeff)
        return total_poly

    if False:  # there is no need
        # look at if the position of the direct image is in the right place
        image = np.zeros((x_len, y_len))
        XSTART, YSTART = XREF + XOFF, YREF + YOFF
        image[YSTART:YSTART + 2, XSTART:XSTART + 2] = 1
        image[YREF - 1:YREF + 3, XREF - 1:XREF + 3] = 1
        p.figure()
        p.title('Offset of direct image')
        view(image, cmap='binary_r')

    TRACE = function(X, X, Y, TRACE_COEFFS)  # height of trace above YREF
    # with respect to direct image position

    # whole image at once, as if the direct image scans over all y
    # this contains a bunch of false information, the wavelengths are only valid on the beam
    # so between BEAM_X1 and BEAM_X2 and only valid at y values where the direct image has been
    # e.g. between YREF +- scan_rate/0.12*subexposure_time pixels

    # dl = ( 1 + (dy/dx)^2 )^0.5 dx
    # integrate to find length along trace at given X
    gradients = np.diff(TRACE, axis=-1)
    gradients = np.hstack([gradients[:, 0].reshape(len(gradients), 1), gradients])  # extend 1st gradient to 0th element
    integrand = np.sqrt(np.square(gradients) + 1)
    lengths = (np.roll(integrand, -1, axis=-1) + integrand) / 2
    lengths = np.hstack([np.zeros([len(TRACE), 1]), lengths[:, :-1]])  # last element is meaningless
    sum_l = np.cumsum(lengths, axis=-1)
    trace_l = sum_l - sum_l[:, int(np.round(XREF))]

    if False:
        # Plot the trace and the direct image
        for i in range(len(TRACE)):
            if abs(YREF - i) < 0.5:
                p.plot(x[0, :], YREF + TRACE[i][:], color='k', alpha=0.5)
                p.plot(x[0, int(XREF) + BEAM_X1:int(XREF) + BEAM_X2],
                       YREF + TRACE[i][int(XREF) + BEAM_X1:int(XREF) + BEAM_X2], color='k')
                p.plot(x[0, int(XREF) + BEAM_X1:int(XREF) + BEAM_X2], [YREF] * (BEAM_X2 - BEAM_X1), color='k', ls='--')
                p.plot(x[0, :], [YREF] * trace_l.shape[-1], color='k', ls='--', alpha=0.5)
                p.plot(x[0, :], [YREF + BEAM_H] * trace_l.shape[-1], color='k', ls='--', alpha=0.5)
                p.plot([XREF] * 2, [YREF, YREF + BEAM_H], marker='*', color='k')
        if original_image is None:
            view(trace_l, title='Position along trace', cmap='hot')
        else:
            view(original_image)
    if False:
        # Just plot the trace locally
        i = np.round(YREF)
        p.plot(x[0, int(XREF) + BEAM_X1:int(XREF) + BEAM_X2], YREF + TRACE[i][int(XREF) + BEAM_X1:int(XREF) + BEAM_X2],
               color='k', label='Trace')
        p.plot(x[0, int(XREF) + BEAM_X1:int(XREF) + BEAM_X2], [YREF] * (BEAM_X2 - BEAM_X1), color='k', ls='--')
        p.plot(XREF, YREF, marker='*', ls='None', color='k', label='Direct image')
        p.xlabel('Spectral pixel')
        p.ylabel('Spatial pixel')
        p.legend(loc=4)
        p.show()
    #print "trace", trace_l
    #print "X", X
    #print "Y", Y
    WAVELENGTHS = function(trace_l, X, Y, DISP_COEFFS) / 10000  # units of micrometers from Angstroms
    # view(WAVELENGTHS, title='Dispersion solution')
    #print WAVELENGTHS
    return WAVELENGTHS, TRACE


##################################################
# First method for interpolation, not vectorized #
##################################################

def interp_flux_scale(scale, wave, flux):
    '''
    Interpolate the flux to the wavelength scale defined
    by 'scale' in the same way the the wavelengths are interpolated.
    Scale should be a 1D array
    Last axis of wave/flux should be interpolated (automatic for interp1d)

    # scipy way
    extrapolator = f.extrap1d(wave,flux)
    return extrapolator(scale)
    '''
    # numpy way
    return np.interp(scale, wave, flux)


def interp_mask_scale(scale, wave, mask):
    '''
    Interpolate the mask contamination to the wavelength scale defined
    by 'scale' in the same way the the wavelengths are interpolated.
    Mask marked by 1 where bad, 0 where good pixels.
    '''
    mask_scale = interp_flux_scale(scale, wave, mask)
    # extrapolated a zero endpoint
    mask_scale[mask_scale < 0] = 0
    # extrapolated a masked endpoint
    mask_scale[mask_scale > 1] = 1
    return mask_scale


def interp_full_image(scale, waves, image, mask):
    new_flux, new_mask = [], []
    for wave, flux, row_mask in zip(waves, image, mask):
        new_flux.append(interp_flux_scale(scale, wave, flux))
        new_mask.append(interp_mask_scale(scale, wave, row_mask))
    new_flux, new_mask = np.vstack(new_flux), np.vstack(new_mask)
    return new_flux, new_mask


###############################################
# Second method for interpolation, vectorized #
###############################################

def get_corresp_ind(scale, waves):
    '''
    Define interpolation lines between waves, then find correspondence between scale
    and nearest (smaller) wave.
    Then calculation is simple.
    scale: wavelength scale to be interpolated to
    waves: wavelengths at each pixel, interpolate each row to scale
    xs: waves without last column

    sum is slightly slower than count_nonzero with list comprehension
    '''
    xs = waves[:, :-1]
    # indexes = np.array([(s > xs).sum(1) for s in scale]).T
    indexes = np.array([[np.count_nonzero(row) for row in (s > xs)] for s in scale]).T
    # indexes of the corresponding wavelengths, along rows
    return indexes


def ind_corr(array, indexes):
    '''Get corresponding values from an array using row indices'''
    rows = np.arange(indexes.shape[0])
    array = array[rows, indexes.T]
    return array.T


def interp_flux_scale2(scale, waves, fluxes, indexes):
    '''
    Same principle as version 1 but this time trying to use arrays as much as possible.
    '''
    # ys = ms * xs + cs (all arrays)
    ms = np.divide(np.diff(fluxes), np.diff(waves))
    ms = np.vstack([ms.T[0], ms.T]).T
    cs = fluxes
    xs = waves

    m = ind_corr(ms, indexes)
    c = ind_corr(cs, indexes)
    x = scale - ind_corr(xs, indexes)  # scale gets broadcast to each row
    scale_flux = m * x + c

    # plot to check
    '''
    ref = 230
    p.plot(waves[ref], fluxes[ref], marker='x')
    p.show()
    p.title('gradients')
    p.plot(ms[ref])
    p.show()
    p.title('constants')
    p.plot(cs[ref])
    p.show()
    p.title('x var')
    p.plot(xs[ref])
    p.show()
    '''
    return scale_flux


def interp_full_image2(scale, waves, image, mask):
    indexes = get_corresp_ind(scale, waves)
    scale_image = interp_flux_scale2(scale, waves, image, indexes)
    scale_mask = mask  # interp_flux_scale2(scale, waves, mask, indexes) > 0.
    return scale_image, scale_mask


#########################
# Flat field correction #
#########################

def flat_field_correct(waves, fluxes, x0=0, x1=-1, ystart=0, yend=-1,
                       flat_file='/home/jacob/hstscan/src/WFC3.G141/WFC3.IR.G141.flat.2.fits', wave_dep=True,
                       ff_min=0.5):
    '''
    Find the pixel values after a wavelength dependant flat-field correction.
    Flat-field is defined by a polynomial whos coefficients are in the conf file:
    WFC3.IR.G141.flat.2.fits
    This is by default 1014x1014, shrink to the size of the waves array.
    '''
    FFHDU = pyfits.open(flat_file, memmap=False)
    # max and min wavelength in microns
    w_max, w_min = FFHDU[0].header['WMAX'] / 10000., FFHDU[0].header['WMIN'] / 10000.
    ff_coeffs = []
    # need to extract the relevant subarray of the flat field image
    for HDU in FFHDU:
        ff_coeffs.append(HDU.data[ystart:yend, x0:x1])
    # assert ff_coeffs[0].shape == fluxes.shape, 'FF shape {}, image shape {}'.format(ff_coeffs[0].shape, fluxes.shape)
    # calculate normalized wavelengths
    if wave_dep:
        x = (waves - w_min) / (w_max - w_min)
    else:
        x = 1

    # evaluate the flat field by a polynomial and divide fluxes
    tot_ff = 0
    for i, ff_coeff in enumerate(ff_coeffs):
        tot_ff = tot_ff + np.multiply(np.power(x, i), ff_coeff)
        ff_error = ff_coeff * np.power(x, i)
        if not wave_dep: break  # only do the wavelength indep part
    FFHDU.close()

    tot_ff[tot_ff < ff_min] = 1.

    #p.plot(np.arange(tot_ff.shape[1]), np.sum(tot_ff, axis=0))
    #p.plot([92, 92], [min(np.sum(tot_ff, axis=0)), max(np.sum(tot_ff, axis=0))], alpha=0.5)
    #p.plot([134, 134], [min(np.sum(tot_ff, axis=0)), max(np.sum(tot_ff, axis=0))], alpha=0.5)
    #p.show()

    return np.divide(fluxes, tot_ff), tot_ff, ff_error
