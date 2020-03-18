# coding=utf-8
import data
import extraction_algorithm as ea
import numpy as np
import pylab as p
import my_fns as f
import batman, logging, time
import astropy.constants as cs
import astropy.io.fits as pyfits
import dispersion as disp
import calibration as cal
from scipy.optimize import leastsq
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

view = data.view_frame_image


# Analysis of data

# obj generally refers to the Data_ima object instance (an exposure)

def reduced_read(read, bg=0, replace=np.NAN, units=True, CCDgain=2.5, int_flags=[4, 32, 512]):
    """
    Reduce a read from the exposures.
    Remove bad pixels, trim reference pixels and subtract background
    if supplied.
    """
    if not read.trimmed:
        read.trim_pix(replace=replace)
    # Gain correction
    # g1, g2, g3, g4 = 2.27,	2.21, 2.20, 2.28 # e-/ADU, uncertainty of 0.02
    # for quadrants 1, 2, 3, 4. From handbook

    # g1, g2, g3, g4 = 2.34, 2.37, 2.31, 2.38
    # calibrated engineering parameters
    # gain for amplifiers A, B, C, D

    # From ISRS 2008, 50, MEB2 First ramps, no IPC corr
    # g1, g2, g3, g4 = 2.76, 2.75, 2.69, 2.77
    # MEB 2 First ramps IPC corr
    g1, g2, g3, g4 = 1.02, 1.02, 1.02, 1.02
    g1, g2, g3, g4 = 2.40, 2.39, 2.34, 2.41
    # g1, g2, g3, g4 = 2.5, 2.5, 2.5, 2.5
    gain_image = np.ones_like(read.SCI.data)
    y, x = gain_image.shape
    gain_image[y / 2:, :x / 2] *= g1
    gain_image[y / 2:, x / 2:] *= g2
    gain_image[:y / 2, x / 2:] *= g3
    gain_image[:y / 2, :x / 2] *= g4
    # gain_image = np.ones_like(read.SCI.data) / 0.87
    # read.SCI.data *= gain_image

    mask = read.remove_bad_pix(replace=replace, int_flags=int_flags)
    read.SCI.data = read.SCI.data - bg  # removes background
    if units:
        try:
            t = read.TIME.header['PIXVALUE']
        except KeyError:
            t = np.median(read.TIME.data)
        read.SCI.data = read.SCI.data * t
        read.ERR.data = read.ERR.data * t
        # convert to electrons
    return read, mask


# This finds the difference between two adjacent (in time) reads.
def create_sub_exposure(read1, read2, read_noise=20, nlincorr=False,
                        nlinfile='/home/jacob/hstscan/src/calibration/u1k1727mi_lin.fits'):
    """
    Simply subtract the two reads,
    time ordered.
    Perform non-linearity correction for saturated pixels (since they are excluded from the default calibration)
    """
    if nlincorr:  # perform non-linearity correction for saturated pixels

        # If we use the saturation flag from the DQ info, will double correct
        # sat1 = (read1.DQ.data/256 % 2).astype(bool) # 256 is the flag for saturation
        # sat2 = (read2.DQ.data/256 % 2).astype(bool)
        # Try to just get the non-corrected pixels (about >78,000 electrons)
        # http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c05_detector8.html
        sat1 = read1.SCI.data > 7.8e4
        sat2 = read2.SCI.data > 7.8e4

        nln = pyfits.open(nlinfile)
        c1 = nln[1].data;
        c2 = nln[2].data;
        c3 = nln[3].data;
        c4 = nln[4].data
        size = read1.SCI.data.shape[0]
        fullsize = c1.shape[0]
        c1, c2, c3, c4 = map(
            lambda c:c[(fullsize - size) / 2:-(fullsize - size) / 2, (fullsize - size) / 2:-(fullsize - size) / 2],
            [c1, c2, c3, c4])

        read1corr, read2corr = map(lambda f:(1 + c1 + c2 * f + c3 * f * f + c4 * f * f * f) * f,
                                   [read1.SCI.data, read2.SCI.data])

        read1.SCI.data = np.where(sat1, read1corr, read1.SCI.data)
        read2.SCI.data = np.where(sat2, read2corr, read2.SCI.data)

    # making subexposures is basically coded into the read (Single_ima) class
    if read2.SCI.header['ROUTTIME'] > read1.SCI.header['ROUTTIME']:
        sub = read2 - read1
    else:
        sub = read1 - read2
    # Create a sensible error estimate
    sub.ERR.data = np.sqrt(abs(sub.SCI.data) + read_noise ** 2)  # photon and read noise
    sub.trimmed = read1.trimmed
    assert read1.trimmed == read2.trimmed, 'Reads should both include/not include reference pixels'
    return sub


def calc_subexposure_background(subexposure, method='median', debug=False, masks=0, psf_w=200, psf_h=None, neg_masks=0,
                                xpix=None, show=True, mask_h=40, **kwargs):
    """
    Calculate the background in a subexposure by crudely masking the spectrum.
    h: height of first spectrum mask
    masks: the number of expected spectra to mask
    """
    image = subexposure.SCI.data
    DQ_mask = subexposure.DQ_mask
    CR_mask = subexposure.CR_mask
    copy = image.copy()
    copy[DQ_mask] = np.nan
    copy[CR_mask] = np.nan

    # find the spectrum and mask it
    spectrum, xpix = find_box(copy.T, h=psf_w)
    spectrum = spectrum.T.copy()
    if psf_h == None:
        copy[:, xpix - psf_w / 2:xpix + psf_w / 2] = np.nan
    else:
        spectrum, ypix = find_box(copy, h=psf_h)
        copy[ypix - psf_h:ypix + psf_h, xpix - psf_w / 2:xpix + psf_w / 2] = np.nan
        # use double the psf_h to be sure to not need more masks
    if debug and False:
        p.title('Image before')
        view(image)
        p.title('Spectrum to be removed')
        view(spectrum, **kwargs)

    # First mask reference pixels and the left edge (ff problems)
    n = 5
    mask0 = np.ones_like(copy)
    mask0[n:-n, n:-n] = 0
    if len(copy) > 256:
        mask0[:, :100] = 1
        mask0[:, -100:] = 1
    mask0 = mask0.astype(bool)
    copy[mask0] = np.nan

    all_masks = []
    # now remove the bg star spectra if there
    if masks > 0:
        mask1, pix = find_box(copy, h=mask_h)
        copy[pix - mask_h / 2:pix + mask_h / 2, :] = np.nan
        all_masks.append(image[pix - mask_h / 2:pix + mask_h / 2, :])
    if masks > 1:
        mask2, pix = find_box(copy, h=mask_h)
        copy[pix - mask_h / 2:pix + mask_h / 2, :] = np.nan
        all_masks.append(image[pix - mask_h / 2:pix + mask_h / 2, :])
    if masks > 2:
        mask3, pix = find_box(copy, h=mask_h)
        copy[pix - mask_h / 2:pix + mask_h / 2, :] = np.nan
        all_masks.append(image[pix - mask_h / 2:pix + mask_h / 2, :])
    # Lastly can remove a negative persistence area
    if neg_masks > 0:
        mask4, pix = find_box(-image.copy(), h=mask_h)
        copy[pix - mask_h / 2:pix + mask_h / 2, :] = np.nan
        all_masks.append(image[pix - mask_h / 2:pix + mask_h / 2, :])

    bgdata = copy.flatten()
    bgdata = bgdata[np.logical_not(np.isnan(bgdata))]
    # reject pixels further than 4 sigma from the median
    mu0, s0 = np.median(bgdata), np.std(bgdata)
    mu, s = mu0, s0
    storedata = bgdata.copy()
    ns = 5
    while np.any(np.logical_or(bgdata > mu + ns * s, bgdata < mu - ns * s)):
        bgdata = bgdata[np.logical_and(bgdata < mu + ns * s, bgdata > mu - ns * s)]
        mu0, s0 = np.median(bgdata), np.std(bgdata)

    bgdata = bgdata[np.logical_and(bgdata < mu0 + ns * s0, bgdata > mu0 - ns * s0)]
    mu, s = np.median(bgdata), np.std(bgdata)

    if debug:
        p.subplot(1, 2, 1)
        copy[copy > mu + ns * s] = np.nan
        copy[copy < mu - ns * s] = np.nan
        vmin, vmax = np.nanmin(copy), np.nanmax(copy)
        view(copy, show=False, vmin=vmin, vmax=vmax, units='electrons')
        view(image, title='Background pixels image', alpha=0.2, cbar=False, vmin=vmin, vmax=vmax, show=False)
        # '({:.0f}:{:.0f})'.format(mu0-ns*s0,mu0+ns*s0
        p.subplot(1, 2, 2)
        p.title('median {:.2f}, std {:.2f}'.format(np.median(bgdata), np.std(bgdata)))
        # y = a * exp( - (x-b)^2 / 2c^2 ) + d
        x = np.arange(min(bgdata), max(bgdata), 1)
        gauss = f.gauss(x, [1., np.median(bgdata), np.std(bgdata)])
        gauss = gauss / (np.sum(gauss) * (x[1] - x[0]))
        hist, bedges = np.histogram(bgdata, bins=20)
        width = bedges[1] - bedges[0]
        # p.hist(bgdata, bins=20)
        p.bar(bedges[:-1] + np.mean(np.diff(bedges)) / 2., hist, width=width)
        area = np.sum(hist) * width
        p.plot(x, gauss * np.sum(hist * width), color='k', lw=2)
        p.tight_layout()
        if show: p.show()

        # for i, mask in enumerate(all_masks):
        #    print i, (np.median(mask)-mu)/s

    if method == 'median':
        bg = np.nanmedian(bgdata)
    elif method == 'mean':
        bg = np.nanmean(bgdata)
    bg_err = np.std(bgdata)
    return float(bg), bg_err


def hist_image_bg(image, debug=False, psf_h=100, masks=2):
    if hasattr(image, 'SCI'): image = image.SCI.data
    image = image.copy()

    spectrum, pix = find_box(image, h=psf_h)
    _, xpix = find_box(image.T, h=200, refine=False)
    select = np.ones_like(image).astype(bool)
    select[pix - psf_h / 2:pix + psf_h / 2, :] = False
    select[:, xpix - 100:xpix + 100] = False
    image[np.logical_not(select)] = 0.
    if debug: print 'Spec pix', pix

    # now remove the bg star spectra if there
    # first two are clear on most images
    if masks > 0:
        mask1, pix = find_box(image, h=40)
        image[pix - 20:pix + 20, :] = 0.
        select[pix - 20:pix + 20] = False
        if debug: print 'Mask 1', pix
    if masks > 1:
        mask2, pix = find_box(image, h=40)
        image[pix - 20:pix + 20, :] = 0.
        select[pix - 20:pix + 20] = False
        if debug: print 'Mask 2', pix
    # third varies with telescope position
    if masks > 2:
        mask3, pix = find_box(image, h=40)
        image[pix - 20:pix + 20, :] = 0.
        select[pix - 20:pix + 20] = False
        if debug: print 'Mask 3', pix

    if debug: copy = image.copy(); copy[np.logical_not(select)] = np.nan; view(copy, vmin=0.)
    image = image[select].flatten()
    hist, bin_edges = np.histogram(image, bins=int(len(image) / 100.), normed=True)
    width = np.median(np.diff(bin_edges))

    def gauss(x, mu, sigma, width=1.):
        out = np.exp(-(x - mu) ** 2 / sigma ** 2 / 2.)
        return out / np.sum(out) / width

    def fit_fn(coefs):
        mu, sigma = coefs
        return (hist - gauss(bin_edges[:-1], mu, sigma, width)) ** 2

    res = leastsq(fit_fn, (8., 4.), full_output=True)
    assert res[-1] in [1, 2, 3, 4], 'Fitting failed'
    mu, sigma = res[0]

    # now trim away anything more than 10 sigma from the mean, this assumed we found the mean...
    image = image[abs(image - mu) < abs(30 * sigma)]
    hist, bin_edges = np.histogram(image, bins=int(len(image) / 10.), normed=True)
    width = np.median(np.diff(bin_edges))

    res = leastsq(fit_fn, (np.mean(image), np.std(image)), full_output=True)
    # print res
    mu, sigma = res[0]
    assert res[-1] in [1, 2, 3, 4], 'Fitting failed'
    if debug:
        ppm = abs((hist - gauss(bin_edges[:-1], mu, sigma, width)) / gauss(bin_edges[:-1], mu, sigma, width)) * 10 ** 6
        residuals = np.median(ppm)
        p.bar(bin_edges[:-1], hist, width=width, color='k')
        p.plot(bin_edges, gauss(bin_edges, mu, sigma, width), ls='--', color='r', lw=3.)
        p.title('mu: {:.2f}, sigma: {:.2f}, res: {:.4g}'.format(mu, sigma, residuals))
        p.show()
    return mu, sigma


def area_bg(source_image, row, col, psf_h, psf_w=200, pix=None, n_masks=0, debug=False):
    image = source_image.copy()
    # find the spectrum and mask it
    spectrum, xpix = find_box(image.T, h=psf_w)
    spectrum = spectrum.T.copy()
    if psf_h == None:
        image[:, xpix - psf_w / 2:xpix + psf_w / 2] = np.nan
    else:
        spectrum, ypix = find_box(image, h=psf_h)
        image[ypix - psf_h / 2:ypix + psf_h / 2, xpix - psf_w / 2:xpix + psf_w / 2] = np.nan

    image[:row[0], :col[0]] = np.nan
    image[row[1]:, :col[0]] = np.nan
    image[:row[0], col[1]:] = np.nan
    image[row[1]:, col[1]:] = np.nan
    bgdata = image.flatten()
    bgdata = bgdata[np.logical_not(np.isnan(bgdata))]

    mu0, s0 = np.median(bgdata), np.std(bgdata)
    mu, s = mu0, s0
    storedata = bgdata.copy()
    ns = 5
    while np.any(np.logical_or(bgdata > mu + ns * s, bgdata < mu - ns * s)):
        bgdata = bgdata[np.logical_and(bgdata < mu + ns * s, bgdata > mu - ns * s)]
        mu0, s0 = np.median(bgdata), np.std(bgdata)

    bgdata = bgdata[np.logical_and(bgdata < mu0 + ns * s0, bgdata > mu0 - ns * s0)]
    mu, s = np.median(bgdata), np.std(bgdata)

    if debug:
        p.subplot(1, 2, 1)
        vmin, vmax = np.nanmin(bgdata), np.nanmax(bgdata)
        view(image, title='Image', show=False, cbar=False, vmin=vmin, vmax=vmax, units='electrons')
        view(source_image, show=False, alpha=0.2, cbar=False, vmin=vmin, title='Background area', vmax=vmax)
        p.subplot(1, 2, 2)
        p.hist(bgdata, bins=20)
        p.title('Mean: {:.1f}, Std: {:.1f}'.format(mu, s))

    return mu, s


def rectangle_bg(source_image, row, col, debug=False):
    image = source_image.copy()
    bg_data = image[row[0]:row[1], col[0]:col[1]].flatten()
    bg_data = bg_data[np.logical_not(np.isnan(bg_data))]
    bg = np.median(bg_data)

    if debug:
        p.subplot(1, 2, 1)
        view(source_image, show=False, vmin=-100, vmax=100, cbar=False)
        p.title('Original Image')
        ax = p.gca()
        ax.set_autoscale_on(False)
        p.plot([row[0], row[0]], [col[0], col[1]], ls='--', color='k')
        p.plot([row[0], row[1]], [col[1], col[1]], ls='--', color='k')
        p.plot([row[1], row[1]], [col[1], col[0]], ls='--', color='k')
        p.plot([row[1], row[0]], [col[0], col[0]], ls='--', color='k')
        p.subplot(1, 2, 2)
        p.hist(bg_data, bins=50, normed=True)
        ax = p.gca()
        ax.set_autoscale_on(False)
        p.plot([bg, bg], [0, 1], ls='--', color='k')
        p.title('Bg median: {:.2f}'.format(bg))
        p.show()

    return bg


def calc_image_background(image, box_h, psf_h, debug=False, above=False):
    """
    Returns a 2D image of the background. Interpolate under the spectrum.
    Done wavelength by wavelength.
    Rewrite B = Ax, where A = [[w 1]] (spatial pixels and ones)
    B = background flux image and x = [[m1 c1], [m2 c2], ...] linear coefs at each wavelength.
    """

    # Find box
    box, pix = find_box(image, h=box_h)
    # Find spectrum in box
    spec, pix2 = find_box(box, h=psf_h)

    # Create B image and spatial pixels (remove spectrum flux)
    if above:
        # Uses only the background above the spectrum to interpolate
        B_bool = np.ones(box.shape[0]).astype(bool)  # which rows are in B
        B_bool[pix2 - box_h / 2:pix2 + psf_h / 2] = False  # dont include spectrum or bg below
        spatial_pixels = np.arange(len(box))
        B, B_pix = box[B_bool], spatial_pixels[B_bool]
    else:
        # Spectrum should always be centered on the image from box_cut
        B_bool = np.ones(box.shape[0]).astype(bool)  # which rows are in B
        B_bool[pix2 - psf_h / 2:pix2 + psf_h / 2] = False  # dont include spectrum
        spatial_pixels = np.arange(len(box))
        B, B_pix = box[B_bool], spatial_pixels[B_bool]
    view(B)
    # Construct A
    A = np.vstack([B_pix, np.ones_like(B_pix)]).T

    # Fit line
    results = np.linalg.lstsq(A, B)
    x = results[0]  # array of linear coefs [m1, c1, m2, c2, ...]
    residuals = np.sqrt(results[1] / (box_h - psf_h))
    m, c = x[0], x[1]

    # Interpolate under spec
    space = spatial_pixels.repeat(len(box.T)).reshape(box.shape)
    bg = np.ones_like(box) * c + space * m
    median = np.nanmedian(bg)

    full_bg = np.ones_like(image) * median
    full_bg[pix - box_h / 2:pix + box_h / 2] = bg

    if debug:
        p.figure()
        p.title('Background interpolations at each wavelength')
        p.xlabel('Spectral Pixel')
        p.ylabel('Background count (electrons)')
        for col in bg.T:
            p.plot(col, color='k', alpha=0.2)
        p.show()

        p.figure()
        p.plot(residuals)
        p.title('Residuals of the linear leastsq fits')
        p.xlabel('Spectral Pixel')
        p.show()

        p.figure()
        n = 30
        p.plot(space.T[n], bg.T[n])
        p.title('Median calculated bg: {}'.format(np.median(bg.T[n])))
        for x, y in zip(space.T[n], box.T[n]):
            if x < (box_h - psf_h) / 2 or x > (box_h + psf_h) / 2:
                color = 'b'
            else:
                color = 'r'
            p.plot(x, y, ls='None', marker='o', color=color)
        # p.ylim([0,100])
        p.show()
    if debug:
        view(bg, units='electrons', title='Calculated background')

    return full_bg  # array


########################################
#             Blob removal             #
########################################

'''
IR spectra have blobs of low sensitivity on the channel select mirror.
These can disperse background light in to spectra on the image.
Having rows with blobs on is therefore not only bad due to having to rescale the flux
when there are many pixels masked out but also will include a background spectrum super imposed.

These are marked by 512? in the DQ array, but cannot interpolate over them 
so probably best to just leave them and ignore the rows that are overly effected
Only if the effect seems obvious on the pixels, otherwise just ignore altogether.
'''


def blobs(mask):
    """
    Identify the blobs in a mask of bad pixels.
    Returns a second mask, that includes only blobs.
    """

    def bad_neighbours(mask):
        # mark bad pixels that have more than 2 bad neighbouring pixels
        count = np.zeros_like(mask)
        for axis in [0, 1]:
            for shift in [-1, 1]:
                count = count + np.roll(mask.astype(int), shift, axis)
        return count

    neighbours = bad_neighbours(mask)
    blob_mask = np.logical_and(neighbours > 2, mask)
    return blob_mask


def has_blobs(mask, tol=4):
    """
    Check if each spatial row in the image has
    a 'blob' feature or not. Must mask those
    that do.
    Tol sets the minimum number of blob pixels for a
    row to be masked.
    """
    blob_mask = blobs(mask)
    mask_row = np.array([np.count_nonzero(row) > tol for row in blob_mask])
    return mask_row


########################################
#          Detect Cosmic Rays          #
#             Local median             #
########################################


def spatial_median_filter(image, dq_mask, tol=5, sx=5, sy=5, replace='median', debug=False, mask_dq=False, thresh=50.):
    """
    Filter cosmic rays by using a local median of pixels
    First compute median in x and y, using sx and sy number of pixels either side
    Then check if pixel is greater than tol sigma above the median for both x and y
    If above, then replace with NaN or median

    Can input an array of pixels to ignore when checking for cosmic rays (dq_mask, bool)

    Thresh is pixel level threshold before it can be called a CR (in elecrons), not used
    """

    masks = [];
    medimages = []
    ress = [];
    stds = []
    # For each axis (y, x direction)
    for axis, shift in zip([0, 1], [sy, sx]):
        if shift != 0:
            # Find the local pixels median and std
            image_stacks = []
            for sh in range(-shift, shift + 1, 1):  # 0 spatial, 1 spectral
                if sh == 0: continue  # dont count self
                new_image = np.roll(image, sh, axis=axis)
                image_stacks.append(new_image)
            medimage = np.nanmedian(image_stacks, axis=0)
            stdimage = np.nanstd(image_stacks, axis=0)

            # check if pixels are outliers
            residuals = np.abs(image - medimage)
            mask = np.abs(residuals) > tol * stdimage
            mask = np.logical_and(mask, residuals > thresh)
            masks.append(mask);
            medimages.append(medimage)
            ress.append(residuals);
            stds.append(residuals / stdimage)
        else:
            mask = np.zeros_like(image)
            masks.append(mask)

    mask = np.logical_or(masks[0], masks[1])  # flag if either are flagged
    if not mask_dq: mask[dq_mask] = False  # dont want to count DQ pixels as CRs
    if replace == 'median':
        replace = np.mean(medimages, axis=0)
    if not replace is None:
        new_image = np.where(mask, replace, image)
    else:
        new_image = image
    cr_info = {}  # store some debugging info about CR properties
    cr_info['mask_y'] = masks[0]
    cr_info['mask_x'] = masks[1]
    cr_info['res_y'] = ress[0]
    cr_info['res_x'] = ress[1]
    cr_info['sigma_y'] = stds[0]
    cr_info['sigma_x'] = stds[1]
    cr_info['cr_vals'] = np.hstack([ress[0][masks[0]].flatten(), ress[1][masks[1]].flatten()])
    cr_info['cr_sigmas'] = np.hstack([stds[0][masks[0]].flatten(), stds[1][masks[1]].flatten()])
    return new_image, mask, cr_info


########################################
#          Fit extraction box          #
########################################

# input a 2D image space/wavelength (subexposure)
# fit a 40 pixel tall box in space over the spectrum
# maximize area within box to find spectrum and fit box

def box_cut(pix, image, h=40, horizontal=False, force_shape=True):
    if horizontal: image = image.T
    pix = int(pix)
    if pix < h / 2:
        if force_shape == True:
            box = image[:h]
        else:
            box = image[:pix + h / 2]
    elif len(image) - pix < h / 2:
        if force_shape == True:
            box = image[-h:]
        else:
            box = image[pix - h / 2:]
    else:
        box = image[pix - h / 2:pix + h / 2]
    if horizontal: box = box.T
    return box


def box_area(pix, image, h, sign):
    pix = int(pix)
    box = box_cut(pix, image, h)
    return np.nansum(box)


def find_box(source_image, h=40, sign='p', refine=False):
    """
    Find maximal flux box.
    sign - p for positive flux only
         - n for negative
         - a for all
    """
    image = source_image.copy()
    # ignore negatives/positives in finding the maximal area
    if sign == 'p':
        image[image < 0] = 0
    elif sign == 'n':
        image[image > 0] = 0
    elif sign == 'a':
        pass
    else:
        assert False, 'Choose either positive (p), negative (n) or all (a) pixels'

    # Only looks for maximal positive area (do in two steps to cut down time)
    areas = [box_area(pix, image, h, sign=sign) for pix in range(int(h / 2), len(image) - int(h / 2))]
    pix = range(int(h / 2), len(image) - int(h / 2))[areas.index(np.max(areas))]
    # first find the area maximizer, then maximize the points within that area
    # this should avoid CRs by taking a big enough area and then refining while
    # weaker to CR hits, but in a smaller space
    if refine:
        spatial_dstn = list(np.nansum(box_cut(pix, image, h), axis=-1))
        shift = spatial_dstn.index(max(spatial_dstn))  # to fine tune the pix value
        pix = pix + shift - h / 2
    else:
        pix = pix
    box = box_cut(pix, image, h)
    return box, pix


##############################################
#          Correcting spectrum shift         #
# 				  and stretch		  	     #
##############################################

from scipy.optimize import curve_fit


def spec_pix_shift2(template_x, template_y, x_new, y_new, norm=True, interp_template=True, p0=0.01, **kwargs):
    """
    Calculate optimal shift in wavelength by cross-correlation.
    Input template_x and x assumed to be in microns
    template_y and y should have been already normalised

    Returns shift between x and template_x, e.g. f(x + shift) = f2(template_x)
    """
    if norm:
        template_y = template_y / np.sum(template_y)
        y_new = y_new / np.sum(y_new)
    if interp_template:
        ref_y = y_new
        interp_y = template_y
    else:
        ref_y = template_y
        interp_y = y_new

    def func(x, shift):
        return np.interp(x, x+ shift, interp_y)

    out = curve_fit(func, template_x, ref_y, p0=(p0), **kwargs)
    shift = out[0][0]
    err = np.diag(np.sqrt(out[1]))[0]

    # assert success, 'Fitting failed'
    return shift, err

def spec_pix_shift(template_x, template_y, x_new, y_new, norm=True, interp_template=True, p0=0.01, fitpeak=False,
                   stretch=False, **kwargs):
    """
    Calculate optimal shift in wavelength by cross-correlation.
    Input template_x and x assumed to be in microns
    template_y and y should have been already normalised
    If fitpeak is enabled, this function will only fit on the peak of the spectrum. This should more accurately fit
     for the shifting of emission/transmission lines.

    Returns shift between x and template_x, e.g. f(x + shift) = f2(template_x)
    """
    if norm:
        template_y = template_y / np.sum(template_y)
        y_new = y_new / np.sum(y_new)
    if interp_template:
        ref_y = y_new
        interp_y = template_y
    else:
        ref_y = template_y
        interp_y = y_new
    if fitpeak:
        Peak = template_y > 0.7 * max(template_y)
        interp_y = interp_y[Peak]
        ref_y = ref_y[Peak]
        template_x = template_x[Peak]
        x_new = x_new[Peak]

    def func(x, shift):
        return np.interp(x, x_new + shift, interp_y)

    def func_stretch(x, shift, stretch):
        return np.interp(x, x_new * stretch + shift, interp_y)

    if stretch:
        try:
            out = curve_fit(func_stretch, template_x, ref_y, p0=(p0, 1.), **kwargs)
            Stretch = out[0][1]
        except RuntimeError:
            out = curve_fit(func, template_x, ref_y, p0=(p0), **kwargs)
            Stretch = 1.
    else:
        out = curve_fit(func, template_x, ref_y, p0=(p0), **kwargs)
    shift = out[0][0]
    err = np.sqrt(np.diag(out[1]))[0]

    # assert success, 'Fitting failed'
    if stretch:
        return shift, Stretch, err
    else:
        return shift, err




def find_xshift_di(exposure, subexposure, direct_image, options, wave_grid, cal_disp_poly_args, plot, fitpeak=False):
    """
    Given an image (with spectrum) and a stellar spectrum it computes the shift in x compared to the direct image.
    This is done by comparing the subexposure spectrum (subtracted by a stellar spectrum) to the grism response function

    The average xshift is best calculated on the average subexposure. This subexposure should have already been
     background subtracted

    :param wave_grid: numpy 1d array with a first estimate of the wavelength grid.
    :return:
    """

    Data = subexposure.SCI.data
    flux = np.nansum(Data, axis=0)
    if fitpeak:
        Peakbool = flux > 0.7 * max(flux)
    else:
        Peakbool = np.ones(len(flux), dtype=bool)

    conf_file = options['conf_file_g141']
    trans_file = options['trans_file_g141']  # Grism response function file
    stellar_file = options['stellar_spectrum']

    F = pyfits.open(trans_file)
    Sensitivity = F[1].data['SENSITIVITY']
    Sensitivity_W = F[1].data['WAVELENGTH'] / 10000.  # Converted to microns
    f_sens = interp1d(Sensitivity_W, Sensitivity, bounds_error=False, fill_value=0.)

    BEAM, DISP_COEFFS, TRACE_COEFFS = disp.get_conf_coeffs(WFC_conf_file=conf_file)
    di_image_distance = BEAM[0]

    Fitsmodel = False
    try:
        stellar_file = pyfits.open(stellar_file)[0]
        Fitsmodel = True
    except:
        print "The stellar file is not a PHOENIX model."
    if Fitsmodel:
        if "PHXVER" in stellar_file.header:
            print "using a PHOENIX stellar model"
        else:
            print "Not using a PHOENIX model but a different model."
            print "Please have a good look at the below plot and check the stellar spectrum."
            plot = True
        stellar_spec = stellar_file.data
        # The resolution of the stellar spectrum is too high, so we apply a guassian kernel of ±20 Angstrom
        if len(stellar_spec)>10000:
            stellar_spec = gaussian_filter(stellar_spec, sigma=len(stellar_spec) / 1000.)
        stellar_W = np.linspace(0.3, 2.4, len(stellar_spec))
    else:
        print "Not using a PHOENIX model but a different model."
        print "Please have a good look at the below plot and check the stellar spectrum."
        plot = True
        stellar_file = np.genfromtxt(stellar_file, skip_header=1)
        stellar_W = stellar_file.T[0] / 10000. #transform to microns.
        stellar_spec = stellar_file.T[1]
    Scale = max(stellar_spec) / max(flux)
    f_stellar = interp1d(stellar_W, stellar_spec / Scale, bounds_error=False)

    c, catalogue, subexp_time, e, scan_direction, a, L, L, XOFF, YOFF, d, b, x_di, ymid = tuple(cal_disp_poly_args)

    Bounds = ([-250., -50.], [50., 0.])
    p0 = [-120., -10.]

    #126.5 -> 127.5
    #124.6 -> 124.6
    #124.9 -> 125.4
    Object = spectrum_fit(f_sens, f_stellar, cal_disp_poly_args, Peakbool)
    opt, cov = Object.fit(wave_grid[0][Peakbool], flux[Peakbool], Bounds, p0)
    displacement, amplitude = opt
    displacement_err, amplitude_err = np.diag(cov)


    #displacement_pix =
    # amplitude = -24.5
    wave_grid_new, trace_new = cal.disp_poly(c, catalogue, subexp_time, e,
                                     scan_direction, n='A', x_len=L, y_len=L, XOFF=XOFF, YOFF=YOFF,
                                     data_dir=d, debug=False, x=x_di + displacement,
                                     y=ymid)

    Calculated = 10 ** (amplitude) * f_sens(wave_grid_new[0]) * f_stellar(wave_grid_new[0])
    """
    print ("sum sq. residuals", np.sqrt(np.sum((Calculated - flux)**2.)), "for displacement", displacement)


    displacement2 = -122.635
    # amplitude = -24.5
    wave_grid_new2, trace_new = cal.disp_poly(c, catalogue, subexp_time, e,
                                     scan_direction, n='A', x_len=L, y_len=L, XOFF=XOFF, YOFF=YOFF,
                                     data_dir=d, debug=False, x=x_di + displacement2,
                                     y=ymid)

    Calculated2 = 10 ** (amplitude) * f_sens(wave_grid_new2[0]) * f_stellar(wave_grid_new2[0])
    print ("sum sq. residuals", np.sqrt(np.sum((Calculated2 - flux)**2.)), "for displacement", displacement2)
    """

    if plot:
        fig = p.figure(figsize=(10, 10))
        p.subplot(211)
        p.plot(wave_grid_new[0], flux, label='Exposure')
        p.plot(wave_grid_new[0], Calculated, label='Fit')
        p.ylabel('Electrons')
        p.legend()
        p.title("With a shift of xshift={}".format(displacement))
        p.gca().xaxis.set_major_locator(p.NullLocator())
        p.subplot(212)
        p.plot(wave_grid_new[0], flux / max(flux), label='Exposure')
        p.plot(wave_grid_new[0], f_sens(wave_grid_new[0]) / max(f_sens(wave_grid_new[0])), label='Sensitivity')
        p.plot(wave_grid_new[0], f_stellar(wave_grid_new[0]) / max(f_stellar(wave_grid_new[0])), label='Stellar')
        fig.subplots_adjust(hspace=0)
        p.xlabel('Wavelength (microns)')
        p.gca().yaxis.set_major_locator(p.NullLocator())
        p.legend()
    return displacement


class spectrum_fit:
    def __init__(self, f_sens, f_stellar, cal_disp_poly_args, Peakbool):
        self.f_sens = f_sens
        self.f_stellar = f_stellar
        self.c = cal_disp_poly_args
        self.Peakbool = Peakbool

    def fitting_fn2(self, wave_grid, displacement, amplitude):
        Total = 10.**amplitude * self.f_sens(wave_grid + displacement) * self.f_stellar(wave_grid + displacement)
        return Total

    def fit2(self, wave_grid, flux, Bounds, P0):
        opt, cov = curve_fit(self.fitting_fn, wave_grid, flux, bounds=Bounds, p0=P0)
        return opt, cov

    def fitting_fn(self, wave_grid, displacement, amplitude):
        a, catalogue, subexp_time, b, scan_direction, d, L, L, XOFF, YOFF, e, f, x_di, ymid = self.c
        wave_grid, trace = cal.disp_poly(a, catalogue, subexp_time, b,
                                         scan_direction, n='A', x_len=L, y_len=L, XOFF=XOFF, YOFF=YOFF,
                                         data_dir=e, debug=False, x=x_di + displacement, y=ymid)
        Total = 10.**amplitude * self.f_sens(wave_grid[0][self.Peakbool]) * self.f_stellar(wave_grid[0][self.Peakbool])
        return Total**2.

    def fit(self, wave_grid, flux, Bounds, P0):
        opt, cov = curve_fit(self.fitting_fn, wave_grid, flux**2., bounds=Bounds, p0=P0)
        return opt, cov
#######################
#        Other        #
#######################

def custom_transit_params(system='GJ-1214', **kwargs):
    """
    Produces params object containing system paramaters
    for transit curve fit.
    Define any system parameters using kwargs or choose a default.
    Uses the batman package:
    http://astro.uchicago.edu/~kreidberg/batman/index.html
    """
    params = batman.TransitParams()  # object to store transit parameters

    if system == 'GJ-1214':
        # Below are rough paramaters for GJ-1214 system
        per = 1.58040464894
        params.t0 = 2454966.52488  # time of inferior conjunction (transit) in ~jd (not 56197.161)
        params.per = per  # orbital period
        params.rp = 0.11  # planet radius (in units of stellar radii) 2.678 earth radii / 1.216 Solar radii
        params.a = 15.23  # semi-major axis (in units of stellar radii)
        params.inc = 89.1  # orbital inclination (in degrees)
        params.ecc = 0.  # eccentricity
        params.w = 90.  # longitude of periastron (in degrees)
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0.28]  # limb darkening coefficients
        params.t_secondary = params.t0 + params.per / 2. * (1 + 4 * params.ecc * np.cos(params.w))
        params.fp = 500e-6  # secondary eclipse depth
    elif system == 'WASP-18':
        # Below are rough paramaters for WASP-18 system
        per = 0.94145299
        # time of inferior conjunction (transit), BJD
        # From Daniel Bayliss 2457338.630292829 -0.00016/+0.00011
        # params.t0 = 2457338.630292829
        # From Maxted paper 2455265.5525(1)  BJD_TDB
        params.t0 = 2455265.5525
        # From Triaud paper 2454664.90531 -0.00016/+0.00017  BJD
        params.per = per  # orbital period
        params.rp = 0.0958  # planet radius (in units of stellar radii) 1.106 jupiter radii /  Solar radii
        params.a = 3.57  # semi-major axis (in units of stellar radii) Triaud 2010
        params.inc = 86.  # orbital inclination (in degrees)
        params.ecc = 0.00848  # eccentricity, Triaud 2010, error ~ 0.00095
        params.w = 96.  # longitude of periastron (in degrees)
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0.33]  # stellar limb darkening coefficients
        params.t_secondary = params.t0 + params.per / 2. * (1 + 4 * params.ecc * np.cos(params.w))
        # Fixed by Spitzer secondary eclipse from L.Krediberg 2457274.142061006e-01
        params.fp = 1e-3  # secondary eclpise depth, wavelength dependent
        params.fnight = 1e-4
    elif system == 'Kepler-9':
        # Below are rough paramaters for Kepler-9 system
        per = 1.5929
        params.t0 = 2455073.43381  # time of inferior conjunction (transit), JD
        params.per = per  # orbital period
        params.rp = 0.0134  # planet radius (in units of stellar radii)
        params.a = 5.293  # semi-major axis (in units of stellar radii)
        params.inc = 90.  # orbital inclination (in degrees)
        params.ecc = 0.  # eccentricity
        params.w = 90.  # longitude of periastron (in degrees)
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0.]  # limb darkening coefficients
        params.t_secondary = params.t0 + params.per / 2. * (1 + 4 * params.ecc * np.cos(params.w))
    elif system == 'WASP-43':
        # Below are rough paramaters for WASP-43 system
        per = 0.81347753
        params.t0 = 2456601.02729  # time of inferior conjunction (transit)
        params.per = per  # orbital period
        params.rp = 0.1595  # planet radius (in units of stellar radii)
        params.a = 4.855  # semi-major axis (in units of stellar radii)
        params.inc = np.arccos(0.13727) * 360 / 2 / np.pi
        # orbital inclination (in degrees)
        params.ecc = 0.0035  # eccentricity
        params.w = 328.0  # longitude of periastron (in degrees)
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0.3]  # limb darkening coefficients
        params.t_secondary = params.t0 + params.per / 2. * (1 + 4 * params.ecc * np.cos(params.w))
        params.fp = 461e-6  # secondary eclipse depth
    elif system == 'WASP-19':
        # Below are rough paramaters for WASP-19 system
        per = 0.78884
        params.t0 = 2455168.96801  # time of inferior conjunction (transit), BJD
        params.per = per  # orbital period
        params.rp = 0.1498  # planet radius (in units of stellar radii)
        params.a = 3.827  # semi-major axis (in units of stellar radii) Triaud 2010
        params.inc = 79.4  # orbital inclination (in degrees)
        params.ecc = 0.0046  # eccentricity, Triaud 2010, error ~ 0.00095
        params.w = 90.  # longitude of periastron (in degrees)
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0.55]  # limb darkening coefficients
        params.t_secondary = params.t0 + params.per / 2. * (1 + 4 * params.ecc * np.cos(params.w))
        params.fp = params.rp / 3.  # secondary eclpise depth
    elif system == 'Kepler-78':
        # Below are rough paramaters for Kepler-78b system
        per = 0.35500744
        params.t0 = 0.
        # From Triaud paper 2454664.90531 -0.00016/+0.00017  BJD
        params.per = per  # orbital period
        params.rp = 0.0161  # Rp/Rs, mean 0.0142
        params.a = 2.0  # semi-major axis (a/Rs), mean 2.0
        params.inc = 79.  # orbital inclination (in degrees)
        params.ecc = 0.  # eccentricity
        params.w = 0.  # longitude of periastron (in degrees)
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0.55]  # stellar limb darkening coefficients
        params.t_secondary = params.t0 + params.per / 2. * (1 + 4 * params.ecc * np.cos(params.w))
        params.fp = 1e-4  # secondary eclipse depth, wave/temp dependent
    elif system == 'WASP-80':
        # Below are rough paramaters for WASP-80b system
        per = 3.0678504
        params.t0 = 2454664.90531
        # From Triaud paper 2454664.90531 -0.00016/+0.00017  BJD
        params.per = per  # orbital period
        params.rp = 0.167444  # Rp/Rs, mean 0.0142
        params.a = 12.97545  # semi-major axis (a/Rs), mean 2.0
        params.inc = 89.92  # orbital inclination (in degrees)
        params.ecc = 0.02  # eccentricity
        params.w = 94.  # longitude of periastron (in degrees)
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0.55]  # stellar limb darkening coefficients
        params.t_secondary = params.t0 + params.per / 2. * (1 + 4 * params.ecc * np.cos(params.w))
        params.fp = 5e-4  # secondary eclipse depth, wave/temp dependent
        params.Hmag = 8.513
    elif system == 'HAT-P-2':
        # Below are rough paramaters for HAT-P-2 b system
        per = 5.6334729
        params.t0 = 0.  # 2455288.84923
        params.t_secondary = 2455289.93211 - 2455288.84923
        params.t_periapse = 2455289.4721 - 2455288.84923
        params.per = per  # orbital period
        params.rp = 0.07695  # Rp/Rs, mean 0.0142
        params.a = 9.72  # semi-major axis (a/Rs), mean 2.0
        params.inc = 86.7  # orbital inclination (in degrees)
        params.ecc = 0.5171  # eccentricity
        params.w = 185.22  # longitude of periastron (in degrees)
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0.55]  # stellar limb darkening coefficients
        params.t_secondary = 55289.4734 - 55288.84988  # from https://arxiv.org/pdf/1302.5084.pdf
        params.fp = 199.5e-6  # secondary eclipse depth, wave/temp dependent
        params.Hmag = 7.652
    elif system == 'WASP-121':
        # Below are paramaters for WASP-121 b system
        per = 1.2749255
        params.t0 = 2456635.70832
        params.t_secondary = params.t0 + per / 2.
        params.per = per  # orbital period
        params.rp = 0.12454  # Rp/Rs, mean 0.0142
        params.a = 0.02544 * 1.496e11 / (1.458 * 6.957e8)  # semi-major axis (a/Rs), mean 2.0
        params.inc = 87.6  # orbital inclination (in degrees)
        params.ecc = 0.  # eccentricity
        params.w = 180.  # longitude of periastron (in degrees)
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0.55]  # stellar limb darkening coefficients
        params.fp = 199.5e-6  # secondary eclipse depth, wave/temp dependent
        params.Hmag = 0.
    elif system == 'XO2':
        # https://arxiv.org/pdf/0705.0003.pdf
        per = 2.615857
        params.t0 = 2454147.74902
        params.t_secondary = params.t0 + per / 2.
        params.per = per  # orbital period
        params.rp = 0.98 * cs.R_jup.value / (0.97 * cs.R_sun.value)  # Rp/Rs, mean 0.0142
        params.a = 0.0369 * 1.496e11 / (0.97 * 6.957e8)  # semi-major axis (a/Rs)
        params.inc = 88.9  # orbital inclination (in degrees)
        params.ecc = 0.  # eccentricity
        params.w = 180.  # longitude of periastron (in degrees)
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0]  # stellar limb darkening coefficients
        params.fp = 0  # secondary eclipse depth, wave/temp dependent
        params.Hmag = 0.
    elif system == 'WASP-103':
        per = 0.925542
        params.t0 = 2456459.59957
        params.t_secondary = params.t0 + per / 2.
        params.per = per  # orbital period
        params.rp = 1.528 * cs.R_jup.value / (1.436 * cs.R_sun.value)  # Rp/Rs, mean 0.0142
        params.a = 0.01985 * 1.496e11 / (0.97 * 6.957e8)  # semi-major axis (a/Rs)
        params.inc = 86.3  # orbital inclination (in degrees)
        params.ecc = 0.  # eccentricity
        params.w = 180.  # longitude of periastron (in degrees)
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0]  # stellar limb darkening coefficients
        params.fp = 0  # secondary eclipse depth, wave/temp dependent
        params.Hmag = 0.
    elif system == 'WASP-12':
        per = 1.0914203
        params.t0 = 2456176.66826
        params.t_secondary = params.t0 + per / 2.
        params.per = per  # orbital period
        params.rp = 1.9 * cs.R_jup.value / (1.657 * cs.R_sun.value)  # Rp/Rs, mean 0.0142
        params.a = 0.0234 * cs.au.value / (1.657 * cs.R_sun.value)  # semi-major axis (a/Rs)
        params.inc = 83.37  # orbital inclination (in degrees)
        params.ecc = 0.  # eccentricity
        params.w = 180.  # longitude of periastron (in degrees)
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0]  # stellar limb darkening coefficient
    elif system == 'K2-24b':
        per = 20.8851
        params.t0 = 2450965.7948
        params.t_secondary = params.t0 + per / 2.
        params.per = per  # orbital period
        params.rp = 0.52 * cs.R_jup.value / (1.657 * cs.R_sun.value)  # Rp/Rs, mean 0.0142
        params.a = 0.154 * cs.au.value / (1.657 * cs.R_sun.value)  # semi-major axis (a/Rs)
        params.inc = 89.25  # orbital inclination (in degrees)
        params.ecc = 0.  # eccentricity
        params.w = 180.  # longitude of periastron (in degrees)
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0.28]  # stellar limb darkening coefficient
    elif system == 'HD209458':
        per = 3.525
        params.t0 = 2456196.28934
        params.t_secondary = params.t0 + per / 2.
        params.per = per  # orbital period
        params.rp = 0.714 * cs.R_jup.value / (1.162 * cs.R_sun.value)  # Rp/Rs, mean 0.0142
        params.a = 0.04747 * cs.au.value / (1.162 * cs.R_sun.value)  # semi-major axis (a/Rs)
        params.inc = 86.59  # orbital inclination (in degrees)
        params.ecc = 0.  # eccentricity
        params.w = 83.  # longitude of periastron (in degrees)
        params.limb_dark = "linear"  # limb darkening model
        params.u = [0.28]  # stellar limb darkening coefficient
    else:
        print 'WARNING unsupported system.'
    for key in kwargs:
        # Overwrite params or define custom system
        val = kwargs[key]
        if type(val) is int:
            val = float(val)
        params.__dict__[key] = val
    return params


def find_orbit_indexes(times, tol=None):
    """Find the indexes that seperate each orbit in a light curve"""
    time_differences = np.diff(times)
    if tol is None: tol = 0.01  # days or (max(times) - min(times)) / 10 hopefully unit independent tolerance
    changes = time_differences > tol
    indexes = [i + 1 for i, change in enumerate(changes) if change]
    return indexes


def unpack_orbits(array, indexes, discard=False):
    """Return individual orbit from light curve and indices"""
    list_array = np.split(array, indexes)
    if discard:
        list_array.pop(0)  # chuck first orbit
    return list_array

