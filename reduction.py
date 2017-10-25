import data
import extraction_algorithm as ea
import numpy as np
import pylab as p
import my_fns as f
import batman, logging, time
from scipy.optimize import leastsq
view = data.view_frame_image
from matplotlib.backends.backend_pdf import PdfPages

# Analysis of data

# obj generally refers to the Data_ima object instance (an exposure)

def reduced_read(read,bg=0, replace=np.NAN, units=True, CCDgain=2.5, int_flags=[4,32,512]):
    '''
    Reduce a read from the exposures.
    Remove bad pixels, trim reference pixels and subtract background
    if supplied.
    '''
    if not read.trimmed:
        read.trim_pix(replace=replace)
    # Gain correction
    #g1, g2, g3, g4 = 2.27,	2.21, 2.20, 2.28 # e-/ADU, uncertainty of 0.02
    # for quadrants 1, 2, 3, 4. From handbook

    #g1, g2, g3, g4 = 2.34, 2.37, 2.31, 2.38
    # calibrated engineering parameters
    # gain for amplifiers A, B, C, D

    # From ISRS 2008, 50, MEB2 First ramps, no IPC corr
    #g1, g2, g3, g4 = 2.76, 2.75, 2.69, 2.77
    # MEB 2 First ramps IPC corr
    g1, g2, g3, g4 = 1.02, 1.02, 1.02, 1.02
    g1, g2, g3, g4 = 2.40, 2.39, 2.34, 2.41
    #g1, g2, g3, g4 = 2.5, 2.5, 2.5, 2.5
    gain_image = np.ones_like(read.SCI.data)
    y, x = gain_image.shape
    gain_image[y/2:,:x/2] *= g1
    gain_image[y/2:,x/2:] *= g2
    gain_image[:y/2,x/2:] *= g3
    gain_image[:y/2,:x/2] *= g4
    #gain_image = np.ones_like(read.SCI.data) / 0.87
    #read.SCI.data *= gain_image
    if False:
        view(gain_image*CCDgain, title='{}, {}, {}, {}'.format(g1,g2,g3,g4))
        view(read.SCI.data)
        p.plot(np.sum(read.SCI.data[150:400,300:400], axis=1))
        p.show()

    mask = read.remove_bad_pix(replace=replace, int_flags=int_flags)
    read.SCI.data = read.SCI.data - bg # removes background
    if units:
        try: t = read.TIME.header['PIXVALUE']
        except KeyError: t = np.median(read.TIME.data)
        read.SCI.data = read.SCI.data * t
        read.ERR.data = read.ERR.data * t
        # convert to electrons
    return read, mask

# This finds the difference between two adjacent (in time) reads.
def create_sub_exposure(read1, read2, read_noise=20):
    '''
    Simply subtract the two reads,
    time ordered.
    '''
    # this is basically coded into the read objects
    if read2.SCI.header['ROUTTIME'] > read1.SCI.header['ROUTTIME']:
        sub = read2 - read1
    else:
        sub = read1 - read2
    # Create a sensible error estimate
    sub.ERR.data = np.sqrt(abs(sub.SCI.data) + read_noise**2) # photon and read noise
    sub.trimmed = read1.trimmed
    assert read1.trimmed == read2.trimmed, 'Reads should both include/not include reference pixels'
    return sub

def calc_subexposure_background(subexposure, method='median', debug=False, masks=0, psf_w=200, psf_h=None, neg_masks=0, xpix=None, show=True, mask_h=40, **kwargs):
    '''
    Calculate the background in a subexposure by crudely masking the spectrum.
    h: height of first spectrum mask
    masks: the number of expected spectra to mask
    '''
    image = subexposure.SCI.data
    DQ_mask = subexposure.DQ_mask
    CR_mask = subexposure.CR_mask
    copy = image.copy()
    copy[DQ_mask] = np.nan
    copy[CR_mask] = np.nan

    # find the spectrum and mask it
    spectrum, xpix = find_box(copy.T, h=psf_w)
    spectrum = spectrum.T.copy()
    if psf_h == None: copy[:,xpix-psf_w/2:xpix+psf_w/2] = np.nan
    else:
        spectrum, ypix = find_box(copy, h=psf_h)
        copy[ypix-psf_h:ypix+psf_h,xpix-psf_w/2:xpix+psf_w/2] = np.nan
        # use double the psf_h to be sure to not need more masks
    if debug and False:
        p.title('Image before')
        view(image)
        p.title('Spectrum to be removed')
        view(spectrum, **kwargs)

    # First mask reference pixels and the left edge (ff problems)
    n = 5
    mask0 = np.ones_like(copy)
    mask0[n:-n,n:-n] = 0
    if len(copy) > 256:
        mask0[:,:100] = 1
        mask0[:,-100:] = 1
    mask0 = mask0.astype(bool)
    copy[mask0] = np.nan

    all_masks = []
    # now remove the bg star spectra if there
    if masks > 0:
        mask1, pix = find_box(copy, h=mask_h)
        copy[pix-mask_h/2:pix+mask_h/2,:] = np.nan
        all_masks.append(image[pix-mask_h/2:pix+mask_h/2,:])
    if masks > 1:
        mask2, pix = find_box(copy, h=mask_h)
        copy[pix-mask_h/2:pix+mask_h/2,:] = np.nan
        all_masks.append(image[pix-mask_h/2:pix+mask_h/2,:])
    if masks > 2:
        mask3, pix = find_box(copy, h=mask_h)
        copy[pix-mask_h/2:pix+mask_h/2,:] = np.nan
        all_masks.append(image[pix-mask_h/2:pix+mask_h/2,:])
    # Lastly can remove a negative persistence area
    if neg_masks > 0:
        mask4, pix = find_box(-image.copy(), h=mask_h)
        copy[pix-mask_h/2:pix+mask_h/2,:] = np.nan
        all_masks.append(image[pix-mask_h/2:pix+mask_h/2,:])

    bgdata = copy.flatten()
    bgdata = bgdata[np.logical_not(np.isnan(bgdata))]
    # reject pixels further than 4 sigma from the median
    mu0, s0 = np.median(bgdata), np.std(bgdata)
    mu, s = mu0, s0
    storedata = bgdata.copy()
    ns = 5
    while np.any(np.logical_or(bgdata>mu+ns*s,bgdata<mu-ns*s)):
        bgdata = bgdata[np.logical_and(bgdata<mu+ns*s,bgdata>mu-ns*s)]
        mu0, s0 = np.median(bgdata), np.std(bgdata)

    bgdata = bgdata[np.logical_and(bgdata<mu0+ns*s0,bgdata>mu0-ns*s0)]
    mu, s = np.median(bgdata), np.std(bgdata)

    if debug:
        p.subplot(1,2,1)
        copy[copy>mu+ns*s] = np.nan
        copy[copy<mu-ns*s] = np.nan
        vmin, vmax = np.nanmin(copy), np.nanmax(copy)
        view(copy, show=False, vmin=vmin, vmax=vmax, units='electrons')
        view(image, title='Background pixels image', alpha=0.2, cbar=False, vmin=vmin, vmax=vmax, show=False)
        # '({:.0f}:{:.0f})'.format(mu0-ns*s0,mu0+ns*s0
        p.subplot(1,2,2)
        p.title('median {:.2f}, std {:.2f}'.format(np.median(bgdata),np.std(bgdata)))
        #y = a * exp( - (x-b)^2 / 2c^2 ) + d
        x = np.arange(min(bgdata),max(bgdata),1)
        gauss = f.gauss(x, [1.,np.median(bgdata),np.std(bgdata)])
        gauss = gauss / (np.sum(gauss)*(x[1]-x[0]))
        hist, bedges = np.histogram(bgdata, bins=20)
        width = bedges[1]-bedges[0]
        #p.hist(bgdata, bins=20)
        p.bar(bedges[:-1]+np.mean(np.diff(bedges))/2., hist, width=width)
        area = np.sum(hist)*width
        p.plot(x, gauss*np.sum(hist*width), color='k', lw=2)
        p.tight_layout()
        if show: p.show()

        #for i, mask in enumerate(all_masks):
        #    print i, (np.median(mask)-mu)/s

    if method=='median':
        bg = np.nanmedian(bgdata)
    elif method=='mean':
        bg = np.nanmean(bgdata)
    bg_err = np.std(bgdata)
    return float(bg), bg_err

def hist_image_bg(image, debug=False, psf_h=100, masks=2):

    if hasattr(image, 'SCI'): image = image.SCI.data
    image = image.copy()

    spectrum, pix = find_box(image, h=psf_h)
    _, xpix = find_box(image.T, h=200, refine=False)
    select = np.ones_like(image).astype(bool)
    select[pix-psf_h/2:pix+psf_h/2,:] = False
    select[:, xpix-100:xpix+100] = False
    image[np.logical_not(select)] = 0.
    if debug: print 'Spec pix', pix

    # now remove the bg star spectra if there
    # first two are clear on most images
    if masks > 0:
        mask1, pix = find_box(image, h=40)
        image[pix-20:pix+20,:] = 0.
        select[pix-20:pix+20] = False
        if debug: print 'Mask 1', pix
    if masks > 1:
        mask2, pix = find_box(image, h=40)
        image[pix-20:pix+20,:] = 0.
        select[pix-20:pix+20] = False
        if debug: print 'Mask 2', pix
    # third varies with telescope position
    if masks > 2:
        mask3, pix = find_box(image, h=40)
        image[pix-20:pix+20,:] = 0.
        select[pix-20:pix+20] = False
        if debug: print 'Mask 3', pix

    if debug: copy = image.copy(); copy[np.logical_not(select)] = np.nan; view(copy, vmin=0.)
    image = image[select].flatten()
    hist, bin_edges = np.histogram(image, bins=int(len(image)/100.), normed=True)
    width = np.median(np.diff(bin_edges))

    def gauss(x, mu, sigma, width=1.):
        out=np.exp(-(x-mu)**2/sigma**2/2.)
        return out/np.sum(out)/width

    def fit_fn(coefs):
        mu, sigma = coefs
        return (hist - gauss(bin_edges[:-1], mu, sigma, width))**2

    res = leastsq(fit_fn, (8., 4.), full_output=True)
    assert res[-1] in [1,2,3,4], 'Fitting failed'
    mu, sigma = res[0]

    if False:
        p.title('Histogram of unmasked pixels')
        p.bar(bin_edges[:-1], hist, width=width, color='k')
        p.plot(bin_edges, gauss(bin_edges, mu, sigma, width), ls='--', color='r')
        p.plot([mu-30*sigma,mu-30*sigma],[0,max(hist)], ls='-', color='g')
        p.plot([mu+30*sigma,mu+30*sigma],[0,max(hist)], ls='-', color='g')
        p.title('mu: {:.2f}, sigma: {:.2f}'.format(mu, sigma))
        p.show()

    # now trim away anything more than 10 sigma from the mean, this assumed we found the mean...
    image = image[abs(image-mu)<abs(30*sigma)]
    hist, bin_edges = np.histogram(image, bins=int(len(image)/10.), normed=True)
    width = np.median(np.diff(bin_edges))

    res = leastsq(fit_fn, (np.mean(image), np.std(image)), full_output=True)
    #print res
    mu, sigma = res[0]
    assert res[-1] in [1,2,3,4], 'Fitting failed'
    if debug:
        ppm = abs((hist - gauss(bin_edges[:-1], mu, sigma, width))/gauss(bin_edges[:-1], mu, sigma, width))*10**6
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
    if psf_h == None: image[:,xpix-psf_w/2:xpix+psf_w/2] = np.nan
    else:
        spectrum, ypix = find_box(image, h=psf_h)
        image[ypix-psf_h/2:ypix+psf_h/2,xpix-psf_w/2:xpix+psf_w/2] = np.nan

    image[:row[0],:col[0]] = np.nan
    image[row[1]:,:col[0]] = np.nan
    image[:row[0],col[1]:] = np.nan
    image[row[1]:,col[1]:] = np.nan
    bgdata = image.flatten()
    bgdata = bgdata[np.logical_not(np.isnan(bgdata))]

    mu0, s0 = np.median(bgdata), np.std(bgdata)
    mu, s = mu0, s0
    storedata = bgdata.copy()
    ns = 5
    while np.any(np.logical_or(bgdata>mu+ns*s,bgdata<mu-ns*s)):
        bgdata = bgdata[np.logical_and(bgdata<mu+ns*s,bgdata>mu-ns*s)]
        mu0, s0 = np.median(bgdata), np.std(bgdata)

    bgdata = bgdata[np.logical_and(bgdata<mu0+ns*s0,bgdata>mu0-ns*s0)]
    mu, s = np.median(bgdata), np.std(bgdata)

    if debug:
        p.subplot(1,2,1)
        vmin, vmax = np.nanmin(bgdata), np.nanmax(bgdata)
        view(image, title='Image', show=False, cbar=False, vmin=vmin, vmax=vmax, units='electrons')
        view(source_image, show=False, alpha=0.2, cbar=False, vmin=vmin, title='Background area', vmax=vmax)
        p.subplot(1,2,2)
        p.hist(bgdata, bins=20)
        p.title('Mean: {:.1f}, Std: {:.1f}'.format(mu,s))

    return mu, s

def rectangle_bg(source_image, row, col, debug=False):

    image = source_image.copy()
    bg_data = image[row[0]:row[1],col[0]:col[1]].flatten()
    bg_data = bg_data[np.logical_not(np.isnan(bg_data))]
    bg = np.median(bg_data)

    if debug:
        p.subplot(1,2,1)
        view(source_image, show=False, vmin=-100, vmax=100, cbar=False)
        p.title('Original Image')
        ax = p.gca()
        ax.set_autoscale_on(False)
        p.plot([row[0],row[0]], [col[0],col[1]], ls='--', color='k')
        p.plot([row[0],row[1]], [col[1],col[1]], ls='--', color='k')
        p.plot([row[1],row[1]], [col[1],col[0]], ls='--', color='k')
        p.plot([row[1],row[0]], [col[0],col[0]], ls='--', color='k')
        p.subplot(1,2,2)
        p.hist(bg_data, bins=50, normed=True)
        ax = p.gca()
        ax.set_autoscale_on(False)
        p.plot([bg,bg], [0,1], ls='--', color='k')
        p.title('Bg median: {:.2f}'.format(bg))
        p.show()


    return bg


'''
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

index = [1, 3]
n_b = a.shape[0] + len(index)

not_index = np.array([k for k in range(n_b) if k not in index])

b = np.zeros((n_b, n_b), dtype=a.dtype)
b[not_index.reshape(-1,1), not_index] = a
'''

def calc_image_background(image, box_h, psf_h, debug=False, above=False):
    '''
    Returns a 2D image of the background. Interpolate under the spectrum.
    Done wavelength by wavelength.
    Rewrite B = Ax, where A = [[w 1]] (spatial pixels and ones)
    B = background flux image and x = [[m1 c1], [m2 c2], ...] linear coefs at each wavelength.
    '''

    # Find box
    box, pix = find_box(image, h=box_h)
    # Find spectrum in box
    spec, pix2 = find_box(box, h=psf_h)

    # Create B image and spatial pixels (remove spectrum flux)
    if above:
        # Uses only the background above the spectrum to interpolate
        B_bool = np.ones(box.shape[0]).astype(bool) # which rows are in B
        B_bool[pix2-box_h/2:pix2+psf_h/2] = False # dont include spectrum or bg below
        spatial_pixels = np.arange(len(box))
        B, B_pix = box[B_bool], spatial_pixels[B_bool]
    else:
        # Spectrum should always be centered on the image from box_cut
        B_bool = np.ones(box.shape[0]).astype(bool) # which rows are in B
        B_bool[pix2-psf_h/2:pix2+psf_h/2] = False # dont include spectrum
        spatial_pixels = np.arange(len(box))
        B, B_pix = box[B_bool], spatial_pixels[B_bool]
    view(B)
    # Construct A
    A = np.vstack([B_pix, np.ones_like(B_pix)]).T

    # Fit line
    results = np.linalg.lstsq(A,B)
    x = results[0] # array of linear coefs [m1, c1, m2, c2, ...]
    residuals = np.sqrt(results[1] / (box_h-psf_h))
    m, c = x[0], x[1]

    # Interpolate under spec
    space = spatial_pixels.repeat(len(box.T)).reshape(box.shape)
    bg = np.ones_like(box)*c + space*m
    median = np.nanmedian(bg)

    full_bg = np.ones_like(image)*median
    full_bg[pix-box_h/2:pix+box_h/2] = bg

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
        n=30
        p.plot(space.T[n],bg.T[n])
        p.title('Median calculated bg: {}'.format(np.median(bg.T[n])))
        for x, y in zip(space.T[n], box.T[n]):
            if x < (box_h-psf_h)/2 or x > (box_h+psf_h)/2:
                color='b'
            else:
                color='r'
            p.plot(x,y,ls='None',marker='o', color=color)
        #p.ylim([0,100])
        p.show()
    if debug:
        view(bg, units='electrons', title='Calculated background')

    return full_bg # array

########################################
#             Blob removal             #
########################################

'''
IR spectra have blobs of low sensitivity on the channel select mirror.
These can disperse background light in to spectra on the image.
Having rows with blobs on is therefore not only bad due to having to rescale the flux
when there are many pixels masked out but also will include a background spectrum super imposed.
'''

def blobs(mask):
    '''
    Identify the blobs in a mask of bad pixels.
    Returns a second mask, that includes only blobs.
    '''
    def bad_neighbours(mask):
        # mark bad pixels that have more than 2 bad neighbouring pixels
        count = np.zeros_like(mask)
        for axis in [0,1]:
            for shift in [-1,1]:
                count = count + np.roll(mask.astype(int),shift,axis)
        return count
    neighbours = bad_neighbours(mask)
    blob_mask = np.logical_and(neighbours > 2, mask)
    return blob_mask

def has_blobs(mask, tol=4):
    '''
    Check if each spatial row in the image has
    a 'blob' feature or not. Must mask those
    that do.
    Tol sets the minimum number of blob pixels for a
    row to be masked.
    '''
    blob_mask = blobs(mask)
    mask_row = np.array([np.count_nonzero(row) > tol for row in blob_mask])
    return mask_row

########################################
#          Detect Cosmic Rays          #
#             Local median             #
#            Sliding median            #
########################################

# Local median array #

def create_median_image(image, befimage=None, aftimage=None, method='space'):
    '''
    Creates a new image where each pixel is the median of the neighbouring pixels
    in the original image.
    Done by iterating through each pixel so is incredibly slow.

    If method is space, takes local median in image.
    If method is time, takes median of adjacent pixels in time
    '''
    # can parse all the 3 images in a list
    # better not have any 3x3 images
    if len(image) == 3:
        images = image
        befimage, image, aftimage = image
    else:
        images = [befimage, image, aftimage]

    # list of neighbours, really need to find a better way of doing this
    neigh_list = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]

    medimage = np.empty(image.shape)
    it = np.nditer(image, flags=['multi_index'])
    # store list of neighbouring pixels
    while not it.finished:
        found, neigh = True, []
        for step in neigh_list:
        # try to access element, if it doesn't exist pass
            try:
                if method == 'space':
                    neigh_index = f.to_tuple(np.add(it.multi_index,step))
                    neigh.append(image[neigh_index])
                elif method == 'time':
                    # more complicated, first positional value is 'height'
                    # or spatial pixel
                    # second is time, so before present or future.
                    src_image, h = images[step[0]], step[1]
                    neigh_index = f.to_tuple(np.add(it.multi_index,(h,0)))
                    neigh.append(src_image[neigh_index])
            except IndexError:
                # edge of array
                pass
        medimage[it.multi_index] = np.nanmedian(neigh)
        it.iternext()
    return medimage

def create_median_image2(image, s=2, befimage=None, aftimage=None, method='space'):
    '''
    Creates a new image where each pixel is the median of the neighbouring pixels
    in the original image.
    Done by using array manipulation over pixel iteration in an attempt to
    speed up the process.
    s = separation of neighbouring pixels
    '''
    assert len(image.shape) == 2, 'Input should be a 2D array'
    assert method=='space' and befimage==None and aftimage==None, 'Does not support time median'

    # Roll the arrays in the x and y direction, then take median.
    # How to handle edges? Repeat endpoints.
    image_stacks = []
    for spatial in range(-s,s+1,1): # 0 spatial, 1 spectral
        one_roll = np.roll(image, spatial, axis=0)
        for spectral in range(-s,s+1,1):
            if spatial == 0 and spectral == 0:
                continue # dont use the pixel in question
            new_image = np.roll(one_roll, spectral, axis=1)
            #view(new_image - image, title='{}_{}'.format(axis, shift))
            if spatial > 0:
                new_image[:s,:] = image[:s,:]
            elif spatial < 0:
                new_image[-s:,:] = image[-s:,:]
            if spectral > 0:
                new_image[:,:s] = image[:,:s]
            elif spectral < 0:
                new_image[:,-s:] = image[:,-s:]
            image_stacks.append(new_image)
    medimage = np.nanmedian(image_stacks, axis=0)
    stdimage = np.std(image_stacks, axis=0)
    return medimage, stdimage

def cr_mask(image, diff, tol, replace=np.NAN):
    '''
    Remove pixels with difference greater than tol
    Returns a bool mask.
    '''

    nans = np.empty(image.shape)
    nans[:] = replace # doesn't have to be NaNs, can replace with zeros for instance
    cr_mask = diff > tol
    view(cr_mask, cmap='binary_r', title='CR Hits')
    print('Removed {} cosmic ray pixels.'.format(np.count_nonzero(cr_mask)))
    return cr_mask

def clean_cosmic_rays(image, error, tol, replace='median', debug=False):
    '''
    Replace CR hits with NaN by taking the difference between
    pixels and the median of their surrounding pixels
    and flagging values above tol
    '''
    medimage, stdimage = create_median_image2(image)
    residual = abs(image) - abs(medimage)
    mask = abs(residual**2) > abs(stdimage**2*tol)

    inds = []
    for _ in range(100): # first 10 crs
        val = 0
        for i, row in enumerate(abs(residual**2) - abs(stdimage**2*tol)):
            for j, col in enumerate(row):
                if col > val and not (i,j) in inds:
                    val = col
                    ind = (i,j)
        inds.append(ind)
    view(image, show=False)
    ax = p.gca()
    ax.set_autoscale_on(False)
    for ind in inds:
        p.plot(ind[0],ind[1],marker='o',mec='w',mfc='None')
    p.show()

    #mask = abs(residual**2) > abs(image*tol)
    if False: # Have a look at the process
        view(residual**2/stdimage**2, vmax =tol, vmin=2)
        #view(image, title='Original Image')
        #view(error, title='Errors')
        view(residual**2/stdimage**2, title='Residual/std')
        view(stdimage, title='std at each pixel')
        #view(np.isnan(residual), cmap='binary_r', title='NaN residuals')
        #view(np.divide(residual, error), title='Criterion')
        view(mask, cmap='binary_r', title='{} CR Hits'.format(np.count_nonzero(mask)))

    if replace == 'median':
        replace = medimage.copy()
        # need to not use mask on other side
    return np.where(mask,replace,image), mask

def view_cosmic_rays(image, error, tol=20, debug=True):
    '''Plot the cosmic rays that would be identified with the given tolerance'''
    clean, mask = clean_cosmic_rays(image, error, tol=tol, debug=True)
    if debug:
        view(mask,cbar=False,cmap='binary_r')
    return mask


def spatial_median_filter(image, dq_mask, tol, sx=1, sy=1, thresh=500, replace='median', debug=False, read_noise=20, mask_dq=False):

    image_stacks = []
    for spatial in range(-sy,sy+1,1): # 0 spatial, 1 spectral
        one_roll = np.roll(image, spatial, axis=0)
        for spectral in range(-sx,sx+1,1):
            if spatial == 0 and spectral == 0:
                continue # dont use the pixel in question
            new_image = np.roll(one_roll, spectral, axis=1)
            #view(new_image - image, title='{}_{}'.format(axis, shift))
            if spatial > 0:
                new_image[:sy,:] = image[:sy,:]
            elif spatial < 0:
                new_image[-sy:,:] = image[-sy:,:]
            if spectral > 0:
                new_image[:,:sx] = image[:,:sx]
            elif spectral < 0:
                new_image[:,-sx:] = image[:,-sx:]
            image_stacks.append(new_image)

    medimage = np.nanmedian(image_stacks, axis=0)
    stdimage = np.nanstd(image_stacks, axis=0)
    # std of the pixels
    varimage = np.abs(medimage) + read_noise**2 # variance of the median pixel
    residuals = np.abs(image - medimage)

    mask2 = np.abs(residuals) > tol*np.sqrt(varimage)
    if not mask_dq: mask2[dq_mask] = False # dont want to count DQ pixels as CRs
    #largest_good = np.max(residuals[np.logical_and(np.logical_not(mask2), np.logical_not(dq_mask))])
    #largest_good = np.max(residuals[np.logical_and(np.logical_not(mask2),np.logical_not(dq_mask))])
    #mask2 = np.logical_and(mask2, residuals > largest_good)
    #mask2 = np.logical_and(mask2, residuals > 400)
    if replace == 'median':
        replace = medimage
    return np.where(mask2,replace,image), mask2

# Sliding median filter #

# input time series of data
# return CR filtered time series

def align_images(subexposures, POSTARG2=11.934, scan_rate=0.12):
    '''
    Allign all the spectra by rolling the arrays
    Need scan direction and rate if variable, the value
    of 0.12"/s is what was applied to L.Kreidberg's observations
    of GJ-1214b. For WASP-18, a scan rate of 0.30"/s seems plausible.
    '''
    if POSTARG2 >= 0:
        # backward scan
        scan_rate = -abs(scan_rate)
    elif POSTARG2 < 0:
        # forward scan
        scan_rate = abs(scan_rate)

    images, shifts = [], []
    for sub in subexposures:
        # to the nearest pixel
        shift = int(scan_rate * sub.TIME.header['PIXVALUE'] / 0.121)
        #       scan rate exp time                      pix size
        image = sub.SCI.data.copy()
        images.append(np.roll(image,shift,axis=0))
        shifts.append(shift)
    return images, shifts

def restore_images(images, shifts):
    '''Undo shifting applied to images'''
    images = [np.roll(image,-shift,axis=0) for image, shift in zip(images,shifts)]
    return images

def slide_median_filter(exposure, tol, width, thresh=500, debug=False, show=False, POSTARG2=11.934, scan_rate=0.12, logger=False):
    '''
    Apply sliding median filter to subexposures.
    Should have DQ flagged pixels already in the .mask attribute.
    Only applied to the 40 pixel tall box around spectrum in subexposure.
    '''
    subexposures = exposure.subexposures
    data, shifts = align_images(subexposures, POSTARG2, scan_rate)

    # stack time series into a n+1D image
    series = np.dstack(data[:])
    CR_masks, images = [], []
    for i in range(series.shape[-1]):
        low, up = i-width, i+width
        if low not in range(series.shape[-1]):
            low, up = 0, up-low
        elif up not in range(series.shape[-1]):
            up, low = series.shape[-1]-1, low-series.shape[-1]+up

        image = series[:,:,i]
        sigma = np.nanstd(series[:,:,low:up], axis=-1)
        median = np.nanmedian(series[:,:,low:up], axis=-1)
        sigma[np.isnan(sigma)] = 0
        lim = median + tol*sigma
        CR_mask = image > lim # highly varying pixels
        CR_mask = np.logical_and(image > thresh, CR_mask) # only include those with sufficient energy
        # i.e. > 500 electrons

        CR_mask[np.isnan(image)] = 0
        # remove the NaNs that would be marked as greater than tolsigma

        if logger:
            logger.info('Subexposure {}, {} CR hit(s)'.format(i,np.count_nonzero(CR_mask)))
            if np.count_nonzero(CR_mask) > 10:
                logger.warning('Exposure {}, subexposure {}'.format(exposure.rootname,i))
                logger.warning('Large number of CR hits found in subexposure: {}'.format(np.count_nonzero(CR_mask)))
                if np.count_nonzero(CR_mask) > 50:
                    # save an image of the CR mask
                    p.figure()
                    p.subplot(2,1,1)
                    view(CR_mask, show=False, cmap='binary_r')
                    p.subplot(2,1,2)
                    view(image, show=False)
                    png_file = '/'.join(exposure.filename.split('/')[:-1])+'/logs/'+exposure.rootname+'_{}.png'.format(i)
                    p.savefig(png_file)
                    logger.warning('Saving image to {}'.format(png_file))


            if show:
                clean = image.copy()
                clean[np.isnan(image)] = 0
                clean[CR_mask] = np.nan
                p.title('Subexposure {}'.format(i))
                view(clean)

        image = np.where(CR_mask, median, image)
        CR_masks.append(CR_mask)
        images.append(image)

    CR_masks = restore_images(CR_masks, shifts)
    images = restore_images(images, shifts)

    for image, CR_mask, subexposure in zip(images, CR_masks, subexposures):
        subexposure.SCI.data = image
        subexposure.SCI.header['CR'] = np.count_nonzero(CR_mask)
        subexposure.SCI.header.comments['CR'] = 'Number of cosmic rays found by sliding median filter'
        # don't actually want to mask out cosmic rays since replacing with median
        #subexposure.mask = np.logical_or(subexposure.mask, CR_mask)

    return subexposures

'''
# ipynb median filter code:
# mess with scan rates and CR checks
reload(data)
images, shifts = [], []
for sub in exposure.subexposures:
    # first embed in a bigger frame
    # sub.nan_to_num(0)
    frame = np.zeros((512,256))
    shift = int(0.12 *       sub.TIME.header['PIXVALUE'] / 0.121)
    #       drift rate   exp time                      pix size
    frame[256-shift:-shift-1,:] = sub.SCI.data.copy()
    frame = frame[371:412,:]
    images.append(data.Single_ima(pyfits.ImageHDU(frame)))
    shifts.append(shift)
reload(r)

cleaned = r.slide_median_filter(images,tol=2.5,width=5,thresh=500, debug=True)
'''


########################################
#          Fit extraction box          #
########################################

# input a 2D image space/wavelength (subexposure)
# fit a 40 pixel tall box in space over the spectrum
# maximize area within box to find spectrum and fit box

def box_cut(pix, image, h=40, horizontal=False):
    if horizontal: image = image.T
    pix = int(pix)
    if pix < h/2:
        box = image[0:h]
    elif len(image) - pix < h/2:
        box = image[-h:]
    else:
        box = image[pix-h/2:pix+h/2]
    if horizontal: box = box.T
    return box

def box_area(pix, image, h, sign):
    pix = int(pix)
    box = box_cut(pix,image,h)
    return np.nansum(box)

def find_box(source_image, h=40, sign='p', refine=False):
    '''
    Find maximal flux box.
    sign - p for positive flux only
         - n for negative
         - a for all
    '''
    image = source_image.copy()
    # ignore negatives/positives in finding the maximal area
    if sign == 'p': image[image<0] = 0
    elif sign == 'n': image[image>0] = 0
    elif sign == 'a': pass
    else: assert False, 'Choose either positive (p), negative (n) or all (a) pixels'

    # Only looks for maximal positive area (do in two steps to cut down time)
    #areas = [box_area(pix, image, h, sign=sign) for pix in range(int(h/2),len(image)-int(h/2),int(h/2))]
    #rpix = areas.index(np.max(areas))
    #areas = [box_area(pix, image, h, sign=sign) for pix in range(rpix-int(h/2),rpix+int(h/2))]
    areas = [box_area(pix, image, h, sign=sign) for pix in range(int(h/2),len(image)-int(h/2))]
    pix = range(int(h/2),len(image)-int(h/2))[areas.index(np.max(areas))]
    # first find the area maximizer, then maximize the points within that area
    # this should avoid CRs by taking a big enough area and then refining while
    # weaker to CR hits, but in a smaller space
    if refine:
        spatial_dstn = list(np.nansum(box_cut(pix, image, h), axis=-1))
        shift = spatial_dstn.index(max(spatial_dstn)) # to fine tune the pix value
        pix = pix + shift - h/2
    else:
        pix = pix
    box = box_cut(pix, image, h)
    return box, pix





##############################################
#          Correcting spectrum shift         #
# 				  and stretch		  	     #
##############################################

def spec_pix_shift(template_x, template_y, x, y, debug=False):
    def min_func(shift):
        shift_y = np.interp(template_x, template_x+shift, y)
        return template_y - shift_y
    results = leastsq(min_func, x0=0.01, full_output=True)
    success = results[-1] in [1,2,3,4]
    shift = results[0][0]
    assert success, 'Fitting failed'
    if debug:
        p.title('Shift: {:.6g} microns or ~{:.2f} pixels'.format(shift, shift/0.0045))
        p.plot(template_x, np.interp(template_x, template_x+shift, y), label='Shifted')
        p.plot(x, y, label='Original')
        p.legend()
        p.show()
        diff = np.max(np.interp(template_x, template_x+shift, y)-y)
        p.title('Difference {:.2g} electrons ({:.2f}%)'.format(diff, diff/max(y)*10**2))
        p.plot(template_x, np.interp(template_x, template_x+shift, y)-template_y, label='Difference from template')
        p.show()
    return shift

def fit_absorbtion(flux, p1, p2):
	'''
	Fit a gaussian to an abosrbtion line in a spectrum.
	Used to determine spectrum stretch when two absorbtion
	lines move apart/closer.
	'''
	absorbtion = flux[p1:p2]
	x = range(len(absorbtion))

	def abs_gauss(x, m, s, C, A):
		return C - A*np.exp((x-m)**2/(2*s**2))

	def fit_fn(coefs):
		m, s, C, A = coefs
		model = abs_gauss(x, m, s, C, A)
		return absorbtion - model

	results = leastsq(fit_fn, coefs0, full_output=1)
	m, s, C, A = results[0]
	model = abs_gauss(x, m, s, C, A)
	if results[-1] in [1,2,3,4]: success=True
	else: success = False

	if success and debug:
		p.figure()
		p.plot(x, absorbtion, marker='o', ls=None, label='data')
		p.plot(x, model, ls='--', color='k', label='fit')
		p.legend(loc=4)
		p.show()
	elif not success:
		print 'Fitting failed', results[-2]
	return m, s, C, A

def find_line_sep(flux):
	'''
	Find the separation between two absorbtion
	lines (in pixels) by fitting gaussians to each.
	'''
	# these are the pixel locations of the lines
	p1,p2,p3,p4 = 1,1,1,1
	m1, s1, C1, A1 = fit_absorbtion(flux, p1, p2)
	m2, s2, C2, A2 = fit_absorbtion(flux, p3, p4)
	separation = abs( (p1+m1) - (p3+m3) )
	return separation

def fit_exposures(visit_dir = '/net/glados2.science.uva.nl/api/jarcang1/GJ-1214b/'):
    '''
    Find the pixel offset that minimizes difference between exposures wl curves.
    Use first exposure as a template.
    '''
    exposures = load_all_ima(visit_dir=visit_dir)
    template = spectrum(exposures[0]) # spectrum curve of first exposure (template)

    p.figure()
    # plot shifts for each exposure
    for exposure in exposures[1:]:

        s = spectrum(exposure)

        shift, var_shift, Infodict, mesg, ler = lsq_pix_shift(template, s)
        print('Found optimal pixel shift of {} pixels'.format(shift[0]))

        if ler in [1,2,3,4]:
            colour = 'b'
        else:
            colour = 'r'
        p.plot(exposure.Primary.header['EXPSTART'], shift,mfc=colour,mec=colour,ls='None',marker='o')

    p.xlabel('Exposure time')
    p.ylabel('Optimal pixel shift')
    p.show()
    p.close()


###########################################
#          Systematic Corrections         #
###########################################

# This is a first guess at some parameters, just here for completeness
coefs0 = [ 0.11, 0.28, 0.006, 2E9, -200, 0.003, 0.002]

def custom_transit_params(system='GJ-1214', **kwargs):
    '''
    Produces params object containing system paramaters
    for transit curve fit.
    Define any system parameters using kwargs or choose a default.
    Uses the batman package:
    http://astro.uchicago.edu/~kreidberg/batman/index.html
    '''
    params = batman.TransitParams()       #object to store transit parameters

    if system == 'GJ-1214':
    # Below are rough paramaters for GJ-1214 system
        per = 1.58040464894
        params.t0 =  2454966.52488            #time of inferior conjunction (transit) in ~jd (not 56197.161)
        params.per = per                      #orbital period
        params.rp = 0.11                      #planet radius (in units of stellar radii) 2.678 earth radii / 1.216 Solar radii
        params.a = 15.23                      #semi-major axis (in units of stellar radii)
        params.inc = 89.1                     #orbital inclination (in degrees)
        params.ecc = 0.                       #eccentricity
        params.w = 90.                        #longitude of periastron (in degrees)
        params.limb_dark = "linear"           #limb darkening model
        params.u = [0.28]                     #limb darkening coefficients
        params.t_secondary = params.t0 + params.per/2.*(1 + 4*params.ecc*np.cos(params.w))
        params.fp = 500e-6           #secondary eclipse depth
    elif system == 'WASP-18':
    # Below are rough paramaters for WASP-18 system
        per = 0.94145299
        #time of inferior conjunction (transit), BJD
        # From Daniel Bayliss 2457338.630292829 -0.00016/+0.00011
        #params.t0 = 2457338.630292829
        # From Maxted paper 2455265.5525(1)  BJD_TDB
        params.t0 =  2455265.5525
        # From Triaud paper 2454664.90531 -0.00016/+0.00017  BJD
        params.per = per                      #orbital period
        params.rp = 0.219                     #planet radius (in units of stellar radii) 1.106 jupiter radii /  Solar radii
        params.a = 3.57                       #semi-major axis (in units of stellar radii) Triaud 2010
        params.inc = 86.                      #orbital inclination (in degrees)
        params.ecc = 0.00848                  #eccentricity, Triaud 2010, error ~ 0.00095
        params.w = 96.                        #longitude of periastron (in degrees)
        params.limb_dark = "linear"           #limb darkening model
        params.u = [0.33]                     #stellar limb darkening coefficients
        params.t_secondary = params.t0 + params.per/2.*(1 + 4*params.ecc*np.cos(params.w))
        # Fixed by Spitzer secondary eclipse from L.Krediberg 2457274.142061006e-01
        params.fp = 1e-3                      #secondary eclpise depth, wavelength dependent
        params.fnight = 1e-4
    elif system == 'Kepler-9':
    # Below are rough paramaters for Kepler-9 system
        per = 1.5929
        params.t0 =  2455073.43381            #time of inferior conjunction (transit), JD
        params.per = per                      #orbital period
        params.rp = 0.0134                    #planet radius (in units of stellar radii)
        params.a = 5.293                      #semi-major axis (in units of stellar radii)
        params.inc = 90.                     #orbital inclination (in degrees)
        params.ecc = 0.                   #eccentricity
        params.w = 90.                     #longitude of periastron (in degrees)
        params.limb_dark = "linear"           #limb darkening model
        params.u = [0.]                      #limb darkening coefficients
        params.t_secondary = params.t0 + params.per/2.*(1 + 4*params.ecc*np.cos(params.w))
    elif system == 'WASP-43':
    # Below are rough paramaters for WASP-43 system
        per = 0.81347753
        params.t0 = 2456601.02729             #time of inferior conjunction (transit)
        params.per = per                      #orbital period
        params.rp = 0.1595                    #planet radius (in units of stellar radii)
        params.a = 	4.855                     #semi-major axis (in units of stellar radii)
        params.inc = np.arccos(0.13727)*360/2/np.pi
        #orbital inclination (in degrees)
        params.ecc = 0.0035                   #eccentricity
        params.w = 328.0                        #longitude of periastron (in degrees)
        params.limb_dark = "linear"           #limb darkening model
        params.u = [0.3]                      #limb darkening coefficients
        params.t_secondary = params.t0 + params.per/2.*(1 + 4*params.ecc*np.cos(params.w))
        params.fp = 461e-6                    #secondary eclipse depth
    elif system == 'WASP-19':
    # Below are rough paramaters for WASP-19 system
        per = 0.78884
        params.t0 =  2455168.96801            #time of inferior conjunction (transit), BJD
        params.per = per                      #orbital period
        params.rp = 0.1498                    #planet radius (in units of stellar radii)
        params.a = 3.827                      #semi-major axis (in units of stellar radii) Triaud 2010
        params.inc = 79.4                     #orbital inclination (in degrees)
        params.ecc = 0.0046                   #eccentricity, Triaud 2010, error ~ 0.00095
        params.w = 90.                        #longitude of periastron (in degrees)
        params.limb_dark = "linear"           #limb darkening model
        params.u = [0.55]                     #limb darkening coefficients
        params.t_secondary = params.t0 + params.per/2.*(1 + 4*params.ecc*np.cos(params.w))
        params.fp = params.rp / 3.           #secondary eclpise depth
    elif system == 'Kepler-78':
    # Below are rough paramaters for Kepler-78b system
        per = 0.35500744
        params.t0 =  0.
        # From Triaud paper 2454664.90531 -0.00016/+0.00017  BJD
        params.per = per                      #orbital period
        params.rp = 0.0161                    # Rp/Rs, mean 0.0142
        params.a = 2.0                        #semi-major axis (a/Rs), mean 2.0
        params.inc = 79.                      #orbital inclination (in degrees)
        params.ecc = 0.                       #eccentricity
        params.w = 0.                         #longitude of periastron (in degrees)
        params.limb_dark = "linear"           #limb darkening model
        params.u = [0.55]                     #stellar limb darkening coefficients
        params.t_secondary = params.t0 + params.per/2.*(1 + 4*params.ecc*np.cos(params.w))
        params.fp = 1e-4                      #secondary eclipse depth, wave/temp dependent
    elif system == 'WASP-80':
    # Below are rough paramaters for WASP-80b system
        per = 3.0678504
        params.t0 =  2454664.90531
        # From Triaud paper 2454664.90531 -0.00016/+0.00017  BJD
        params.per = per                      #orbital period
        params.rp = 0.167444                  # Rp/Rs, mean 0.0142
        params.a = 12.97545                   #semi-major axis (a/Rs), mean 2.0
        params.inc = 89.92                    #orbital inclination (in degrees)
        params.ecc = 0.02                     #eccentricity
        params.w = 94.                        #longitude of periastron (in degrees)
        params.limb_dark = "linear"           #limb darkening model
        params.u = [0.55]                     #stellar limb darkening coefficients
        params.t_secondary = params.t0 + params.per/2.*(1 + 4*params.ecc*np.cos(params.w))
        params.fp = 5e-4                      #secondary eclipse depth, wave/temp dependent
        params.Hmag = 8.513
    elif system == 'HAT-P-2':
    # Below are rough paramaters for HAT-P-2 b system
        per = 5.6334729
        params.t0 =  0. #2455288.84923
        params.t_secondary=  2455289.93211 - 2455288.84923
        params.t_periapse = 2455289.4721 - 2455288.84923
        params.per = per                      #orbital period
        params.rp = 0.07695                  # Rp/Rs, mean 0.0142
        params.a = 9.72                       #semi-major axis (a/Rs), mean 2.0
        params.inc = 86.7                      #orbital inclination (in degrees)
        params.ecc = 0.5171                   #eccentricity
        params.w = 185.22                     #longitude of periastron (in degrees)
        params.limb_dark = "linear"           #limb darkening model
        params.u = [0.55]                     #stellar limb darkening coefficients
        params.t_secondary = 55289.4734 - 55288.84988    # from https://arxiv.org/pdf/1302.5084.pdf
        params.fp = 199.5e-6                      #secondary eclipse depth, wave/temp dependent
        params.Hmag = 7.652
    elif system == 'WASP-121':
    # Below are paramaters for WASP-121 b system
        per = 1.2749255
        params.t0 =  	2456635.70832
        params.t_secondary = params.t0 + per/2.
        params.per = per                      #orbital period
        params.rp = 0.12454                 # Rp/Rs, mean 0.0142
        params.a = 0.02544*1.496e11 / (1.458*6.957e8)   #semi-major axis (a/Rs), mean 2.0
        params.inc = 87.6                      #orbital inclination (in degrees)
        params.ecc = 0.                   #eccentricity
        params.w = 180.                     #longitude of periastron (in degrees)
        params.limb_dark = "linear"           #limb darkening model
        params.u = [0.55]                     #stellar limb darkening coefficients
        params.fp = 199.5e-6                      #secondary eclipse depth, wave/temp dependent
        params.Hmag = 0.
    else: print 'WARNING unsupported system.'
    for key in kwargs:
    # Overwrite params or define custom system
        val = kwargs[key]
        if type(val) is int:
            val = float(val)
        params.__dict__[key] = val
    return params

def find_orbit_indexes(times, tol=None):
    time_differences = np.diff(times)
    if tol is None: tol = 0.01 #days or (max(times) - min(times)) / 10 hopefully unit independent tolerance
    changes = time_differences > tol
    indexes = [i+1 for i, change in enumerate(changes) if change]
    return indexes

def unpack_orbits(array, indexes, discard=True):
    list_array = np.split(array, indexes)
    if discard:
        list_array.pop(0) # chuck first orbit
    return list_array
