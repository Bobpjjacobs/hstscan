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

def get_fielddep_coeffs(xs, ys, DISP_COEFFS, TRACE_COEFFS):
    '''
    Calculate the field dependent coefficients for defining the trace and wavelength solutions
    '''
    # at (DYDX_A_1)
    m = cal.calc_poly_order(TRACE_COEFFS[1])
    at = cal.field_dep_coeff(m, TRACE_COEFFS[1], xs, ys)
    # bt (DYDX_A_0)
    m = cal.calc_poly_order(TRACE_COEFFS[0])
    bt = cal.field_dep_coeff(m, TRACE_COEFFS[0], xs, ys)
    
    #aw, bw
    # aw (DLDP_A_1)
    m = cal.calc_poly_order(DISP_COEFFS[1])
    aw = cal.field_dep_coeff(m, DISP_COEFFS[1], xs, ys)
    # bw (DLDP_A_0)
    m = cal.calc_poly_order(DISP_COEFFS[0])
    bw = cal.field_dep_coeff(m, DISP_COEFFS[0], xs, ys)

    return at, bt, aw, bw


def dispersion_solution(x0, L, Dxoff, Dxref, ystart, yend, 
                        wvmin=1.14, wvmax=1.6, 
                        conf_file='/home/jacob/hstscan/src/WFC3.G141/WFC3.IR.G141.V2.5.conf'):

    x1s = x0 + Dxoff + Dxref + (512-0.5*L) # relative to full array
    yss = np.linspace(ystart,yend,100) + 512-0.5*L # let y vary along scan
    lams = np.linspace(wvmin,wvmax,20)*1e4 # in angstrom

    DISP_COEFFS, TRACE_COEFFS = get_conf_coeffs(conf_file)

    # Now generate a grid of (wavelength, xl, yl) used to calculate the w.d.p.t.
    grid = []
    for lam in lams:
        for ys in yss:
            at, bt, aw, bw = get_fielddep_coeffs(x1s, ys, DISP_COEFFS, TRACE_COEFFS)
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
        '''Wavelegnth dependent photon trajectory'''
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
        ys = [ys1,ys2][np.argmin([abs(ys1-y),abs(ys2-y)])]
        at, bt, aw, bw = get_fielddep_coeffs(x1s, ys, DISP_COEFFS, TRACE_COEFFS)
        
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

def interp_wave_grid(wave_ref, wave_grid, image):
    '''
    Interpolate image to new wavelength grid (linear interpolation)   
    '''
    new_image = np.empty(image.shape)
    for i in range(len(image)):
        wave, row = wave_grid[i], image[i]
        new_row = np.interp(wave_ref, wave, row)
        new_image[i] = new_row
    return new_image

def gaussian(x, scale, mu, sig):
    '''Simple gaussian (1D)'''
    return scale*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_yscan(image, x0, nsig=5, debug=False):
    '''
    Find extent of scan in y-direction by fitting gaussian
    For long scans better to use a top-hat function
    '''
    row_sum = np.sum(image[:,int(x0):int(x0)+200], axis=1)
    out = curve_fit(gaussian, np.arange(len(row_sum)), row_sum, p0=(np.max(row_sum), len(row_sum)/2., 50))
    scale, mu, sig = out[0]
    sig = abs(sig) # apparently nescessary
    if debug:
        p.plot(row_sum)
        p.plot(gaussian(np.arange(len(row_sum)), scale,mu,sig))
        p.title('{:.2f} +- {:.2f}'.format(mu,sig))
        p.show()
    ystart, yend = int(mu-nsig*sig), int(np.ceil(mu+nsig*sig))
    if ystart < 0: ystart = 0
    if yend >= len(row_sum): yend = len(row_sum)-1
    return ystart, yend
