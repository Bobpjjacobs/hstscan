from my_fns import np, p, os
from matplotlib.pyplot import rcParams
from scipy.optimize import leastsq, curve_fit
import calibration as cal

def get_conf_coeffs(conf_file = '/home/jacob/hstscan/src/WFC3.G141/WFC3.IR.G141.V2.5.conf'):
    '''
    Read configuration file for G141
    Defines trace position and wavelengths along trace, using field-dependent coefficients
    See aXe manual for details
    '''
    n = 'A'
    with open(conf_file) as cf:
        lines = cf.readlines()
        lines = [line[:-1] for line in lines if line[0] != '#']
    DISP_COEFFS, TRACE_COEFFS = [], []
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
    return DISP_COEFFS, TRACE_COEFFS

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
                        conf_file='/home/jacob/hstscan/src/WFC3.G141/WFC3.IR.G141.V2.5.conf'):

    x1s = x0 + Dxoff + Dxref + (512-0.5*L) # relative to full array
    yss = np.linspace(ystart,yend,wdpt_grid_y) + 512-0.5*L # let y vary along scan
    lams = np.linspace(wvmin,wvmax,wdpt_grid_lam)*1e4 # in angstrom

    if not DISP_COEFFS or not TRACE_COEFFS: DISP_COEFFS, TRACE_COEFFS = get_conf_coeffs(conf_file)
    # Pre-compute some steps for speed since need to do this at all gridpoints
    ms = map(cal.calc_poly_order, [TRACE_COEFFS[1], TRACE_COEFFS[0],DISP_COEFFS[1],DISP_COEFFS[0]])


    # Now generate a grid of (wavelength, xl, yl) used to calculate the w.d.p.t.    
    grid = []
    for lam in lams:
        for ys in yss:
            at, bt = get_fielddep_coeffs(x1s, ys, TRACE_COEFFS, ms=ms[:2])
            aw, bw = get_fielddep_coeffs(x1s, ys, DISP_COEFFS, ms=ms[2:])
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

    def get_ys(x,y,xs=x1s):
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
        return (-b+det**0.5)/(2*a), (-b-det**0.5)/(2*a)

    def inv_wdpt(x,y):
        '''More usefully calculate the wavelength given a position on the detector'''
        # First find direct image position that correspons to this trace
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
    for i in range(yend-ystart):
        for j in range(200):
            xi, yi = x[j], y[i]
            lam = inv_wdpt(xi,yi)
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

def interp_wave_grid(wave_ref, wave_grid, image, mask, tol=0.01):
    '''
    Interpolate image to new wavelength grid (linear interpolation)   
    '''
    from data import view_frame_image as view
    assert image.shape == mask.shape, 'Image and mask should be of the same shape'
    new_image = np.empty(image.shape)
    new_mask = np.empty(mask.shape)
    for i in range(len(image)):
        wave, row, mrow = wave_grid[i], image[i], mask[i].astype(float) 
        new_row = np.interp(wave_ref, wave, row)
        new_image[i] = new_row

        new_mrow = np.interp(wave_ref, wave, mrow)
        new_mask[i] = new_mrow
    #view(mask, cbar=False, cmap='binary_r')
    #view(new_mask, cbar=False)
    new_mask = new_mask > tol
    #view(new_mask, cbar=False, cmap='binary_r')
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
    args = out[0]
    if two_scans:
        scale1, mu1, sig1, scale2, mu2, sig2 = args
        if abs(y0-mu1) < abs(y0-mu2):
            scale, mu, sig = scale1, mu1, abs(sig1)
        else:
            scale, mu, sig = scale2, mu2, abs(sig2)
        ystart, yend = int(mu-nsig*sig), int(mu+nsig*sig)     
    else: 
        scale, mu, sig, width = out[0]
        sig = abs(sig) # apparently nescessary
        if width <0.: width = 0
        ystart, yend = int(mu-2*width-nsig*sig), int(mu+2*width+nsig*sig)
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
    
    return ystart, yend
