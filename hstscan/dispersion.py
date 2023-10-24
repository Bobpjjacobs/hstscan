from my_fns import np, p, os
from matplotlib.pyplot import rcParams
from scipy.optimize import leastsq, curve_fit
from scipy.interpolate import interp1d
import calibration as cal

def get_conf_coeffs(WFC_conf_file = '/home/jacob/hstscan/src/WFC3.G141/WFC3.IR.G141.V2.5.conf', order=1):
    '''
    Read configuration file for G141
    Defines trace position and wavelengths along trace, using field-dependent coefficients
    See aXe manual for details
    '''
    if order == 0:
        n = 'B'
    elif order == 1:
        n = 'A'
    elif order == 2:
        n = 'C'
    elif order == 3:
        n = 'D'
    elif order == -1:
        n = 'E'
    with open(WFC_conf_file) as cf:
        lines = cf.readlines()
        lines = [line[:-1] for line in lines if line[0] != '#']
    DISP_COEFFS, TRACE_COEFFS = [], []
    BEAM_X1, BEAM_X2 = [], []
    for line in lines:
        if line.startswith('BEAM'+str(n)):
            BEAM_X1, BEAM_X2 = [int(pos) for pos in line.split()[1:]]
        elif line.startswith('DISP_ORDER_'+str(n)):
            DISP_ORDER = int(line.split(' ')[1])
        elif line.startswith('DYDX_ORDER_'+str(n)):
            TRACE_ORDER = int(line.split(' ')[1])
        elif line.startswith('DLDP_'+str(n)):
            nth_coeffs = line.split(' ')[1:]
            nth_coeffs = [float(coeff) for coeff in nth_coeffs if coeff != '']
            DISP_COEFFS.append(nth_coeffs)
        elif line.startswith('DYDX_'+str(n)):
            nth_coeffs = line.split(' ')[1:]
            nth_coeffs = [float(coeff) for coeff in nth_coeffs if coeff != '']
            TRACE_COEFFS.append(nth_coeffs)
    BEAM = [BEAM_X1, BEAM_X2]
    return BEAM, DISP_COEFFS, TRACE_COEFFS

def get_fielddep_coeffs(xs, ys, COEFFS, ms=None, permute_arrays=[None]*2):
    '''
    Calculate the field dependent coefficients for defining the trace and wavelength solutions
    '''
    if ms is None: 
        ms = map(cal.calc_poly_order, [COEFFS[1], COEFFS[0]])

    # at (DYDX_A_1) or aw (DLDP_A_1)
    a = cal.field_dep_coeff(ms[0], COEFFS[1], xs, ys, permute_arrays=permute_arrays[0])
    # bt (DYDX_A_0) or bw (DLDP_A_0)
    b = cal.field_dep_coeff(ms[1], COEFFS[0], xs, ys, permute_arrays=permute_arrays[1])

    return a, b

def dispersion_solution(x0, L, Dxoff, Dxref, ystart, yend, 
                        wvmin=1., wvmax=1.8, DISP_COEFFS=None, TRACE_COEFFS=None, wdpt_grid_y=20, wdpt_grid_lam=20,
                        WFC_conf_file='/home/jacob/hstscan/src/WFC3.G141/WFC3.IR.G141.V2.5.conf'):

    x1s = x0 + Dxoff + Dxref + (512-0.5*L) # relative to full array
    yss = np.linspace(ystart,yend,wdpt_grid_y) + 512-0.5*L # let y vary along scan
    lams = np.linspace(wvmin,wvmax,wdpt_grid_lam)*1e4 # in angstrom

    if not DISP_COEFFS or not TRACE_COEFFS: BEAM, DISP_COEFFS, TRACE_COEFFS = get_conf_coeffs(WFC_conf_file)
    # Pre-compute some steps for speed since need to do this at all gridpoints
    ms = map(cal.calc_poly_order, [TRACE_COEFFS[1], TRACE_COEFFS[0],DISP_COEFFS[1],DISP_COEFFS[0]])

    # Now generate a grid of (wavelength, xl, yl) used to calculate the w.d.p.t.    
    grid = []
    for lam in lams:
        for ys in yss:
            at, bt = get_fielddep_coeffs(x1s, ys, TRACE_COEFFS, ms=ms[:2])
            aw, bw = get_fielddep_coeffs(x1s, ys, DISP_COEFFS,  ms=ms[2:])
            xl = x1s - at*bt / (1+at**2) + (lam - bw)/aw*np.cos(np.arctan(at))
            yl = at*(xl-x1s) + bt + ys
            grid.append([lam, xl, yl])
    grid = np.array(grid)
    # fit the w.d.p.t. coefficients on the grid
    def wdpt_fn(xl, c1, c2, c3, s1, s2, s3):
        l = grid[:,0]
        return (c1/(c2+l)+c3) + (s1/(s2+l) + s3)*xl 
    fit = curve_fit(wdpt_fn, grid[:,1], grid[:,2], p0=[4e08, -9e03,  2e04, -8e05, -9e03, 8e-03], full_output=True)
    c1,c2,c3,s1,s2,s3 = fit[0]
    # This is the final w.d.p.t. equation for a given (xl, wavelength)
    def wdpt(xl, l):
        '''Wavelength dependent photon trajectory'''
        return (c1/(c2+l)+c3) + (s1/(s2+l) + s3)*xl 

    def get_ys(x, y, xs=x1s):
        '''
        Find position of ys whose trace goes through x,y
        trace eq: y-y* = at(x-x*)+bt
        so using the fixed x*, we can find y* (after solving the quadratic of field-dep coeffs)
        '''
        atn = TRACE_COEFFS[1]; btn = TRACE_COEFFS[0]

        # quadratic sln for y*: ay2+by+c=0, after expanding at and bt in terms of x* and y*
        a = atn[5]*(x-xs); b = atn[2]*(x-xs) + atn[4]*xs*(x-xs) + btn[2]+1
        c = (atn[0]+atn[1]*xs+atn[3]*xs**2)*(x-xs) + btn[0]+btn[1]*xs - y
        det = b*b - 4*a*c
        # return both solutions and pick physical one yourself
        '''assert np.all(det >= 0), 'Negative det, imaginary solutions only'
        if a == 0.:
            assert np.all(b != 0), 'Non-sensical equation for y*, a=b=0'
            return -c/b, -c/b
        else:
            return (-b+det**0.5)/(2*a), (-b-det**0.5)/(2*a)'''
        return np.where( a == 0, (-c/b, -c/b), ((-b+det**0.5)/(2*a), (-b-det**0.5)/(2*a)) )


    def inv_wdpt(x, y, ys=None):
        '''More usefully calculate the wavelength given a position on the detector'''
        # First find direct image position that correspons to this trace
        if ys is None:
            ys1, ys2 = get_ys(x,y,xs=x1s)
            if abs(ys1-y) < abs(ys2-y): ys = ys1
            else: ys = ys2
        aw, bw = get_fielddep_coeffs(x1s, ys, DISP_COEFFS, ms=ms[2:])

        # use DISP soln to get lambda (wavelength)
        # what is d? assume its distance from (xs,ys) along trace
        d = ((x-x1s)**2+(y-ys)**2)**0.5 # trace is straight line so distance is pythag
        lam = aw*d + bw
        return lam/1e4 # return wavelength in micron

    # Finaly compute wavelength solution
    wave_grid = np.empty((yend-ystart,200))
    x = np.arange(x0,x0+200)+(512-0.5*L); y = np.arange(ystart, yend)+(512-0.5*L)
    
    # Compute ystar (ys) positions, array-wise
    x_grid, y_grid = np.meshgrid(x, y)
    inv_yss1, inv_yss2 = get_ys(x_grid, y_grid)
    inv_yss = np.where(abs(inv_yss1-y_grid)<abs(inv_yss2-y_grid), inv_yss1, inv_yss2)

    for i in range(yend-ystart):
        for j in range(200):
            xi, yi = x[j], y[i]
            ys = inv_yss[i, j]
            lam = inv_wdpt(xi,yi,ys=ys)
            assert not np.isnan(lam), 'x{}-y{}\nNaN wavelength'.format(xi, yi)
            wave_grid[i,j] = lam

    return wave_grid

def sum_to_spectrum_linear(image, wave_grid, x0, ystart, yend, wave_bins=np.linspace(1.14,1.6,15)):
    '''
    Sum between wdpt defined by wave_bins
    Linearly interpolate edge pixels
    '''
    wave_grid_left = (np.roll(wave_grid, 1, axis=1)+wave_grid)/2.
    wave_grid_right = (np.roll(wave_grid, -1, axis=1)+wave_grid)/2.

    flux = np.empty(len(wave_bins)-1)
    for i in range(len(wave_bins)-1):
        wv1, wv2 = wave_bins[i], wave_bins[i+1]
        weights = np.logical_and(wave_grid_right<wv2, wave_grid_left>wv1).astype(float)
        assert not np.all(weights == 0), 'Wavelength bin smaller than 1 pixel'
        pixsize = wave_grid_right-wave_grid_left
        
        leftedge = np.logical_and(wave_grid_right>wv1, wave_grid_left<wv1)
        weights[leftedge] = ((wave_grid_right - wv1)/pixsize)[leftedge]
            
        rightedge = np.logical_and(wave_grid_right>wv2, wave_grid_left<wv2)
        weights[rightedge] = ((wv2 - wave_grid_left)/pixsize)[rightedge]
        #view(weights, cmap='jet', cbar=False, title='{:.2f}-{:.2f}'.format(wv1,wv2))
       
        flux[i] = np.sum(image[ystart:yend,int(x0):int(x0)+200]*weights)

    return (wave_bins[:-1]+wave_bins[1:])/2., flux

def quad_coeffs(f0,f1,f2):
    '''
    Calculate "sane" poly coefficients from pixel fluxes.
    Defined so that area of poly in pixel = pixel flux
    '''
    a = (f0-2*f1+f2)/2.
    b = (f2-f0)/2.
    c = f1 - a/12.
    return a, b, c

def int_quad(x1, x2, a, b, c):
    '''Integrate quadratic polynomical between x1 and x2.'''
    return a/3.*(x2**3-x1**3) + b/2.*(x2**2-x1**2) + c*(x2-x1)


def interp_sanity_quad(wv1, wv2, row2, plot=False, zero=1e-6):
    '''
    Interpolate spectrum (wv2,row2) to new wavelengths (wv1) using "sane" quadratic
    '''
    
    # Calculate integration limits
    wv_means1 = (wv1[:-1]+wv1[1:])/2.
    wv_lims1 = zip(wv_means1[:-1], wv_means1[1:]) # bin edges
    wv_lims1 = np.vstack([(wv1[0],wv_means1[0]), wv_lims1, (wv_means1[-1], wv1[-1])]) # end cases

    wv_means2 = (wv2[:-1]+wv2[1:])/2.
    wv_lims2 = zip(wv_means2[:-1], wv_means2[1:])
    wv_lims2 = np.vstack([(wv2[0],wv_means2[0]), wv_lims2, (wv_means2[-1], wv2[-1])])


    new_row2 = []
    for lim1 in wv_lims1: # for each reference (new) wavebin
        reflam0, reflam1 = lim1
        # Fix edgecases by essentially defining zero outside
        if (reflam0 < min(wv2) and reflam1 < min(wv2)) or (reflam0 > max(wv2) and reflam1 > max(wv2)):
            newpixvalue = zero # if it is actually zero may be a problem
        
        else:
            if reflam0 < min(wv2): relam0 = min(wv2)
            if reflam1 > max(wv2): reflam1 = max(wv2)

            # get index of respective bins on the edges
            i0, i1 = np.argmin(abs(wv2-reflam0)), np.argmin(abs(wv2-reflam1))

            # Now go through each pixel that contains some of refwave bin
            # integrate areas under "sane" 2nd order poly
            newpixvalue = 0
            for ipix in range(i0, i1+1): # for each pixel to integrate
                lim2 = wv_lims2[ipix]
                pixlam0, pixlam1 = lim2

                if reflam0 <= pixlam0 and reflam1 >= pixlam1: 
                    # both edges outside pixel, include whole pixel
                    newpixvalue += row2[ipix]
                    if plot: ax0.plot([ipix-0.5,ipix+0.5],[row2[ipix]]*2, color='r', lw=4)
                else:
                    # Partial pixel to include, need to integrate, so calculate polynomial
                    if ipix == 0:
                        # linear on endpoints, not quad, since no 3rd pixel
                        f0, f1, f2 = 0, row2[ipix], row2[ipix+1]
                        a, b, c = 0, f2-f1, f1
                    elif ipix == len(row2)-1:
                        f0, f1, f2 = row2[ipix-1], row2[ipix], 0
                        a, b, c = 0, f1-f0, f1
                    else:
                        f0, f1, f2 = row2[ipix-1], row2[ipix], row2[ipix+1]
                        a, b, c = quad_coeffs(f0,f1,f2)

                    wvarea = pixlam1-pixlam0
                    if reflam0 > pixlam0 and reflam1 < pixlam1: 
                        # both edges inside pixel
                        x0 = (reflam0-pixlam0) / wvarea - 0.5
                        x1 = (pixlam1-reflam1) / wvarea - 0.5
                    elif reflam0 > pixlam0: # and lam1 > lim2[1]: 
                        # Left edge
                        # integrate from lam0 to lim2[1]
                        x0 = ((reflam0-pixlam0) / wvarea) - 0.5
                        x1 = 0.5 # lim2[1]
                    elif reflam1 < pixlam1: # and lam0 < lim2[0]
                        # Right edge
                        # integrate from lim2[0] to lam1
                        x0 = -0.5 # lim2[0]
                        x1 = (reflam1-pixlam0) / wvarea - 0.5
                    else:
                        assert False, 'Case not accounted for: Ref {:.7f}-{:.7f}, Pix {:.7f}-{:.7f}'.format(
                                                                reflam0,reflam1,pixlam0,pixlam1)

                    area = int_quad(x0,x1,a,b,c)
                    if area < 0:
                        #print 'Area negative for lam={}, pix={}, set to 0'.format(lim1,ipix)
                        area = 0
                    newpixvalue += area

                    if plot:
                        _x = np.linspace(-.5,.5,100)
                        _xl = np.linspace(-1.5,1.5,100)
                        ax0.plot(ipix+_x, a*_x**2+b*_x+c, label='Area {:.2f}'.format(area), lw=4, color='C{}'.format(ipix-i0), zorder=10)
                        ax0.plot(ipix+_xl, a*_xl**2+b*_xl+c, label='Area {:.2f}'.format(area), ls='--', color='C{}'.format(ipix-i0))
                        ax0.plot([ipix+x0,ipix+x1],[f1]*2, color='r', lw=4)
            if plot: 
                p.legend()
                p.show()
        new_row2.append(newpixvalue)
    return new_row2

def interp_wave_grid(wave_ref, wave_grid, image, mask, tol=0.01, interp_kind='linear'):
    '''
    Interpolate image to new wavelength grid (linear interpolation or quad)   
    '''
    from data import view_frame_image as view
    assert image.shape == mask.shape, 'Image and mask should be of the same shape'
    new_image = np.empty(image.shape)
    new_mask = np.empty(mask.shape)
    for i in range(len(image)):
        wave, row, mrow = wave_grid[i], image[i], mask[i].astype(float) 
        fn = interp1d(wave, row, kind=interp_kind, bounds_error=False) # 1D kind (cubic, linear etc)
        new_row = fn(wave_ref)
        new_image[i] = new_row

        new_mrow = np.interp(wave_ref, wave, mrow)
        new_mask[i] = new_mrow
    #view(mask, cbar=False, cmap='binary_r')
    #view(new_mask, cbar=False)
    new_mask = new_mask > tol
    #view(new_mask, cbar=False, cmap='binary_r')
    return new_image, new_mask

def interp_wave_grid_sane(wave_ref, wave_grid, image, mask, tol=0.01):
    '''
    Interpolate image to new wavelength grid (linear interpolation or quad)   
    '''
    assert image.shape == mask.shape, 'Image and mask should be of the same shape'
    new_image = np.empty(image.shape)
    new_mask = np.empty(mask.shape)
    for i in range(len(image)):
        wave, row, mrow = wave_grid[i], image[i], mask[i].astype(float) 
        new_row = interp_sanity_quad(wave_ref, wave, row)
        new_image[i] = new_row

        new_mrow = interp_sanity_quad(wave_ref, wave, mrow)
        new_mask[i] = new_mrow
    new_mask = new_mask > tol
    return new_image, new_mask


# Matrix used for calculating analytic 2D quadratic over local pixels
matrix = np.array([
        [1,-1,0,13/12.,1/12.],
        [1, 0,1,1/12.,13/12.],
        [1, 0,0,1/12., 1/12.],
        [1,0,-1,1/12.,13/12.],
        [1,1, 0,13/12.,1/12.],
        ])
invmatrix = np.linalg.inv(matrix)

def oversample_image(image, n, mask=None):
    '''Create a new image, oversampled by "n", using 2D quadratic fits to local pixels.'''
    # New image (oversampled)
    overimage = np.empty((image.shape[0]*n, image.shape[1]*n))

    # Create new pixel grid (within a pixel)
    _x0 = np.linspace(-0.5,0.5,n)
    _y0 = np.linspace(-0.5,0.5,n)
    _gx0, _gy0 = np.meshgrid(_x0, _y0)

    # Unfortunately have to loop over pixels
    # for each pixel calculated 2D quadratic over adjacent pixels
    # then evaluate on new grid
    for ix in range(image.shape[1]):
        for iy in range(image.shape[0]):
            fpix =  [image[(iy+s[1])%512, (ix+s[0])%512] for s in [(-1,0),(0,1),(0,0),(0,-1),(1,0)]]
            coefs = np.matmul(invmatrix, fpix)
            def f2D(x0,y0):
                return coefs[0]+coefs[1]*x0+coefs[2]*y0+coefs[3]*x0*x0+coefs[4]*y0*y0
            overpix = f2D(_gx0,_gy0)
            overimage[n*iy:n*iy+n,n*ix:n*ix+n] = overpix

    if mask is not None:
        overmask = mask.repeat(n).reshape(mask.shape[0]*n, -1)
        overmask = overmask.T.repeat(n).reshape(mask.shape[0]*n, mask.shape[1]*n)
    else: overmask = None
    return overimage, overmask

def gaussian(x, scale, mu, sig):
    '''Simple gaussian (1D)'''
    return scale*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def extended_gaussian(x, scale, mu, sig, width):
    '''Extended gaussian, with flat peak of length width'''
    if width < 0.: # assume 0
        return gaussian(x,scale,mu,sig)
    else:
        left = gaussian(x, scale, mu-width/2., sig)
        right = gaussian(x, scale, mu+width/2., sig)
        peak = gaussian(mu, scale, mu, sig)
        return np.where(x<mu-width/2., left, np.where(x>mu+width/2., right, peak))

def two_gaussians(x, scale1, mu1, sig1, scale2, mu2, sig2):
    '''Sum of two gaussians.'''
    return gaussian(x, scale1, mu1, sig1) + gaussian(x, scale2, mu2, sig2)

def get_y0(subexposure, y_di, yguess_offsets, di_ps2, postarg2, expnr, exptime, subexptime, scan_direction, t, tel, logger):
    """
    Estimate a y0 from the direct image

    :param subexposure: an _ima subexposure instance
    :param y_di: the y-position of the star in the direct image
    :param yguess_offsets: The offsets to the guess of the y0 position: one for the forward direction, one for reverse.
    :param di_ps2: the postarg2 argument of the direct image
    :param postarg2: the postarg2 argument of the subexposure
    :param exptime: the exposure time of the exposure containing the subexposure
    :param subexptime: the exposure time of this subexposure
    :param scan_direction: the scan direction of the exposure containing the subexposure. Should be either +1 or -1
    :param t: An option dictionary
    :param tel: The telescope in use
    :param logger: A logger object.
    :return: y0 and the width of the spectrum in y-direction.
    """
    if scan_direction == -1:
        yguess_offset = yguess_offsets[1]
    elif scan_direction == 1:
        yguess_offset = yguess_offsets[0]
    y0 = y_di + yguess_offset + (scan_direction * (
            t.scan_rate * subexptime)) / tel.yscale  # guess of y, in pixels
    if t.remove_scan and ((t.default_scan == 'r' and scan_direction == +1) or (
            t.default_scan == 'f' and scan_direction == -1)):
        y0 -= (scan_direction * (
                t.scan_rate * exptime)) / tel.yscale  # undo full scan
        if expnr == 0: logger.info('default_scan=scan ({}). Undo full scan'.format(t.default_scan))
    if t.postarg_yguess:
        y0 -= (postarg2 - di_ps2) / tel.yscale
        if expnr == 0: logger.info('applying postarg offset to yguess, {:.2f} pix'.format(
            (postarg2 - di_ps2) / tel.yscale))
    if t.scanned:
        width0 = subexposure.SCI.header['DELTATIM'] * t.scan_rate / tel.yscale  # initial guess of width
    else:
        width0 = 40
    if y0 + width0  > subexposure.SCI.data.shape[1]:
        y0 = subexposure.SCI.data.shape[1] - width0 #/ 2.
    elif y0 - width0  < 0:
        y0 = width0 #/ 2.
    return y0, width0

def get_yscan(image, x0, nsig=5, debug=False, y0=None, sigma0=5, width0=30, two_scans=False):
    '''
    Find extent of scan in y-direction by fitting extended gaussian.
    
    width0 : estimate of width of extended gaussian in pixels
    '''
    row_sum = np.nansum(image[:,int(x0):int(x0)+200], axis=1)
    row_sum[np.logical_not(np.isfinite(row_sum))] = 0. 
    if y0 > len(row_sum): y0 = len(row_sum)*3/4
    elif y0 < 0: y0 = len(row_sum)/4

    if not two_scans:
        model = extended_gaussian
        p0 = (np.max(row_sum), y0, sigma0, width0)
    else:
        model = two_gaussians
        # start other gaussian in different quadrant
        if y0 > len(image)/2.: y1 = len(image)/4
        else: y1 = len(image)*3/4
        p0 = (np.max(row_sum), y0, sigma0, np.max(row_sum), y1, sigma0)

    out = curve_fit(model, np.arange(len(row_sum)), row_sum, p0=p0)
    Range = np.arange(len(row_sum))
    maxindex = np.argwhere(row_sum == np.max(row_sum))[0][0]
    f1 = interp1d(row_sum[Range < maxindex], Range[Range < maxindex])
    f2 = interp1d(row_sum[Range > maxindex], Range[Range > maxindex])
    fwhm = f2(0.5* np.max(row_sum)) - f1(0.5* np.max(row_sum))
    print "FWHM of this subexposure is ", fwhm
    args = out[0]
    arg_errs = np.sqrt(np.diag(out[1]))
    if two_scans:
        scale1, mu1, sig1, scale2, mu2, sig2 = args
        if abs(y0-mu1) < abs(y0-mu2):
            scale, mu, sig = scale1, mu1, abs(sig1)
            ymid_err = arg_errs[1]
        else:
            scale, mu, sig = scale2, mu2, abs(sig2)
            ymid_err = arg_errs[4]
        ystart, ymid, yend = int(mu-nsig*sig), mu, int(mu+nsig*sig)     
    else: 
        scale, mu, sig, width = out[0]
        ymid_err = arg_errs[1]
        sig = abs(sig) # apparently nescessary
        if width <0.: width = 0
        ystart, ymid, yend = int(mu-2*width-nsig*sig), mu, int(mu+2*width+nsig*sig)
    # Sanity corrections
    if ystart < 0: ystart = 0
    if yend >= len(row_sum): yend = len(row_sum)-1    
    

    if debug:
        p.plot(row_sum, label='Image', color='k', lw=3, alpha=0.5)
        p.plot(model(np.arange(len(row_sum)), *args), label='Best fit')
        p.plot(model(np.arange(len(row_sum)), *p0), ls='--', label='Initial Guess')        
        if not two_scans: p.title('{:.2f} +- {:.2f}, width {:.2f}'.format(mu,sig,width))
        else: p.title('{:.2f} +- {:.2f}'.format(mu,sig))

        p.gca().set_autoscale_on(False)
        ylim = p.ylim()
        p.plot([ystart]*2,[0,ylim[1]*10], color='r', ls='--', label='Bounds')
        p.plot([yend]*2,[0,ylim[1]*10], color='r', ls='--')        
        p.legend()
        p.show()

    return ystart, ymid, yend, ymid_err, fwhm
