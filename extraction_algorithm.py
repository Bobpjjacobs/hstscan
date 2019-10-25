#from __future__ import print_function
# Imports of basic functions and packages
import my_fns
import my_errors as err
from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from my_fns import np, p
from data import view_frame_image as view
import logging
from matplotlib.backends.backend_pdf import PdfPages

# Notes from Algorithm paper

# All matrices are over x and l in that order
#                   space and wavelength
# e.g. D = (Dxl)
# hence there are many transposes as np by default uses the first index
# but we wish to use the second in many cases

# D is the reduced image, after removing a bias image and dividing by a balance factor
# S is the sky background
# D-S is thus the spectrum in space and wavelength space
# Sum D-S over space gives the spectrum, estimated by f
#
# Initial vairance estimate:
#                            V = V_0 + abs(D)/Q
#
# sqrt V_0 is the rms of readout noise
# Q is the effective photon no. per data number (assumed 1 if in electrons)
#
# After a spectrum estimate:
#                            V = V_0 + abs( f*P + S)/Q
#
# P probability photon with wavelength l detected at pixel (x,l)
#                - spatial profile image to fit to data
#
# Initial P estimate:
#                           P = D - S / sum( D - S ) over x
#                                       = f
#
# Then fit polynomials/gaussian to this estimate iteratively, removing bad pixels
#
# First extract a 40 pixel tall box centered in the middle of the spatial scan, otherwise
# very difficult to fit functions.



# SPECTRUM #

def estimate_spectrum(D,S,V):
    '''
    Find the estimated spectrum and variance given the image data.
    Returns spectrum and spectrum variance.
    '''
    return np.nansum(D-S, axis=0), np.nansum(V, axis=0)

def optimized_spectrum(D, S, P, V, M):
    '''Extract the spectrum using the spatial profile.'''
    denom = np.nansum(np.multiply(M,np.divide(np.square(P),V)), axis=0)
    # optimized spectrum
    f_opt = np.divide(np.nansum(np.multiply(M, np.divide(np.multiply(P,D-S),V)), axis=0), denom)
    # optimized variance
    fV_opt = np.divide(np.nansum(np.multiply(M,P), axis=0),denom)
    return f_opt, fV_opt

# VARIANCE #

def initial_variance_estimate(D, V_0, Q):
    '''Calculate first estimate of pixel the variance.'''
    return V_0 + np.absolute(D)/Q

def estimate_variance(f, P, S, V_0, Q):
    '''
    Estimate variance after a spectrum estimate.
    Provided a normalized spatial profile and a sky image.
    '''
    return V_0 + np.absolute( (np.multiply(P,f) + S) )/Q

# CLIP COSMIC RAYS #

def clip_sigma(D, f, P, S, V, s_clip):
    '''
    Mark outliers in distn by a bool array of the same size.
    '''
    # define the values to be checked
    X = np.divide(np.square(D - np.multiply(f,P) - S),V)
    if np.any(abs(X) > s_clip**2) and False:
        p.subplot(2,1,1)
        p.plot(X, marker='o')
        p.plot([0,60],[s_clip**2]*2, ls='--', color='k')
        p.subplot(2,1,2)
        p.plot(D, marker='x',color='g')
        p.show()
    #print '#', s_clip*np.std(X), '#', X
    if s_clip is None:
        return np.zeros(D.shape, dtype=bool)
    else:
        return abs(X) > s_clip**2


def clip_cosmic(D, f, P, S, V, s_clip, M):
    '''
    Remove one pixel from the array due to a cosmic ray hit.
    Done using numpy array functions, matrix operations.
    Could be faster doing one iteration loop, depends on speed of np fns.
    Returns the updated data image and if detected hit (bool).
    '''
    # define the values to be checked
    # print('D {}, f {}, P {}, S {}, V {}'.format(D.shape, f.shape, P.shape, S.shape, V.shape))
    X = np.divide(np.square(D - np.multiply(f,P) - S),V)

    val = 0
    for i, row in enumerate(X):
        for j, col in enumerate(row):
            if col > val and M[i,j]:
                val = col
                ind = (i,j)

    #ind = (flat_ind // X.shape[1], flat_ind % X.shape[1])
    # index of the maximum, have to convert from the flattened array index
    thresh = s_clip**2
    if False:
        p.title(ind[0])
        p.plot(X[:,ind[1]], color='k')
        p.plot([ind[0],ind[0]], [0,np.max(X)], color='r')
        p.plot([0,60],[s_clip**2]*2, ls='--')
        p.show()
        p.subplot(1,3,1)
        view(D, show=False, title='Image', cbar=False)
        ax = p.gca()
        ax.set_autoscale_on(False)
        p.plot(ind[1],ind[0], marker='o',mec='w',mfc='None')
        p.subplot(1,3,2)
        view(X, vmax=thresh, vmin=thresh/10.,show=False, title='thresh {:.0f}, peak {:.0f}'.format(s_clip**2, np.max(X)), cbar=False)
        ax = p.gca()
        ax.set_autoscale_on(False)
        p.plot(ind[1],ind[0], marker='o',mec='w',mfc='None')
        p.subplot(1,3,3)
        view(P, show=False, title='Profile', cbar=False)
        ax = p.gca()
        ax.set_autoscale_on(False)
        p.plot(ind[1],ind[0], marker='o',mec='w',mfc='None')
        p.tight_layout()
        p.show()
        p.title(ind)
        p.plot(P[:,ind[-1]]*f[ind[-1]],color='b',label='P')
        p.plot(V[:,ind[-1]],color='k',ls='--',label='V')
        p.plot(D[:,ind[-1]], marker='x',color='g',ls='None', label='D')
        p.plot([ind[0],ind[0]], [0,np.max(D[:,ind[-1]])], color='r')
        p.legend()
        p.show()
    if X[ind] > s_clip**2:
        M[ind] = 0
        return D, M, True
    else:
        return D, M, False

# SPATIAL PROFILE #

def initial_profile_estimate(D, S, f=None):
    '''
    Construct estimate of P.
    This is then used to fit function and retrieve spatial distn.
    If f is already calculate skips step.
    '''
    D_S = D - S
    D_S[D_S < 0] = 0 # to normalize P better
    if f is None:
        f = np.nansum(D_S,axis=0)
    f[f==0] = 1
    return np.true_divide(D_S,f)

# this dictionary contains all the functions that can be used to fit a spatial profile
# and then their first guesses for coefficients
fn_dict = {'poly':my_fns.poly_n, 'gauss':my_fns.gauss, 'heavi':my_fns.custom_hill, 'spline':my_fns.spline, 'split_spline': my_fns.split_spline, 'custom_spline': my_fns.custom_spline}
coef_dict = {'poly':[0,0,0], 'gauss':[20000.,20.,0.25,0.], 'heavi':[1, 10, 15, 20000, 0], 'spline': None, 'split_spline': None, 'custom_spline': None}
# old gauss [100,150,0.05,2]

def weight_function(coef, x, distn, weights, func_type, method):
    '''Function to be minimized.'''
    diff = distn - fn_dict[func_type](x, coef)
    weighted = diff * weights

    if method == 'lsq':
        return weighted
    else:
        # need scalar output
        return sum(np.square(weighted))


def FIT_single(x, distn, P_l, V_l, outliers, coef0, func_type, method, debug, tol, order, i, bg=None, step=None):
    '''
    Fits a polynomial/gaussian/heavisides to a spatial distribution given some set of outlier
    pixels to ignore.
    Weights are inversely proportional to the variance.
    '''
    full_debug = False # for excessive information

    # If not signal, return fail
    if np.mean(distn) < np.sqrt(np.mean(V_l)):
        return False, 0.

    weights = np.ones_like(P_l)
    #weights[np.logical_or(np.isnan(weights), np.isinf(weights))]= 0
    weights[ outliers ] = 0 # ignore outlier pixels
    weights = weights / np.sum(weights)
    assert np.count_nonzero(V_l<0) == 0, 'Pixels exist with negative variances {}'.format(V_l)
    assert np.count_nonzero(weights<0) == 0, 'Pixels exist with negative weights {}'.format(weights)

    if func_type == 'spline':
        if not tol is None: tol = np.mean(distn)*tol # set the tolerance to n% of the mean flux?
        if tol <= 0 or tol is None: tol=len(distn); logger.info('Tolerance set to default in spline fit for column {}'.format(i))

        if debug and full_debug: print 'Tolerance set to:', tol
        if not order: order = 2
        if np.all(weights == np.zeros_like(weights)): print 'panic'

        results = UnivariateSpline(x=x,y=distn,w=weights, s=tol, k=order, check_finite=True)

        #if type(results) is None: logger.warning('Fitting has failed to produce a spline')
        fit = my_fns.spline(x, results)
        residual = results.get_residual()
        success = residual < tol*1.1
        j = 1
        while not success:
            # Try iterating a few more times
            results.set_smoothing_factor(tol)
            fit = my_fns.spline(x, results)
            residual = results.get_residual()
            success = residual < tol*1.1
            if np.any(np.isnan(fit)):
                #logger.info('Fitting failed, NaN outputs... interrupted column {}.'.format(i))
                success = False
                break
            elif j == 5 and not success:
                #logger.info('Over 100 iterations attempted... interrupted column {}.'.format(i))
                success = False
                break
            j += 1
        if np.any(np.isnan(fit)): success = False
        #logger.warning('{} iterations, tolerance {}'.format(j, tol))

        if debug and False:
            print 'Tolerance set to:', tol
            print 'Initial P between {:.2g} and {:.2g}'.format(min(P_l), max(P_l))
            print 'Variance between {:.2g} and {:.2g} electrons'.format(min(V_l), max(V_l))
            print 'Weights between {:.2g} and {:.2g}'.format(min(weights), max(weights))
            fit = my_fns.spline(x, results)
            p.plot(distn, marker='o', ls='None')
            p.plot(fit, color='g')
            p.show()
            p.title('RMS: {:.4f}'.format(np.sqrt(np.mean((distn-fit)**2))))
            p.plot(distn-fit, marker='o', ls='None')
            p.show()
        #res, knots = results.get_residual(), len(results.get_knots())
        results = [success, results]

    elif func_type == 'custom_spline':
        if not tol is None: tol = 40*len(distn)*tol # set the tolerance to n% of the mean flux?
        if tol <= 0 or tol is None: tol=len(distn); logger.info('Tolerance set to default in spline fit for column {}'.format(i))
        if not order: order = 2

        n_knots = 2 # start
        res = tol + 1
        while res > tol:
            # initial guesses
            poly_coefs = np.repeat(np.zeros(order+1),(n_knots-1)).reshape(n_knots-1, order+1)
            nodes = np.linspace(0,len(distn), n_knots)
            def fit_fn(coefs):
                nodes, poly_coefs = coefs[:n_knots], coefs[n_knots:].reshape(n_knots-1, order+1)
                y = my_fns.custom_spline(x, (nodes, poly_coefs))
                y = y.astype(float)
                res = np.abs(distn-y)
                return np.sum(res)
            def check_cts(coefs):
                nodes, poly_coefs = coefs[:n_knots], coefs[n_knots:].reshape(n_knots-1, order+1)
                temp1 = [np.sum([ poly_coefs[ind-1][n]*(nodes[ind]**n) for n in range(len(poly_coefs[ind])) ]) for ind in range(len(nodes))[1:-1]]
                temp2 = [np.sum([ poly_coefs[ind][n]*(nodes[ind]**n) for n in range(len(poly_coefs[ind])) ]) for ind in range(len(nodes))[1:-1]]
                return np.sum(np.abs(np.array(temp1)-np.array(temp2)))

            coefs0 = np.hstack((nodes, poly_coefs.flatten()))
            #results = optimize.leastsq(fit_fn, coefs0, full_output=1, ftol=tol)
            results = optimize.fmin_slsqp(fit_fn, coefs0, f_eqcons=check_cts, acc=1e-4, iter=200, full_output=1, iprint=0)
            coefs, success = results[0], results[-2] == 0
            coefs = (coefs[:n_knots], coefs[n_knots:].reshape(n_knots-1, order+1))
            fit = my_fns.custom_spline(x, coefs)
            fit_nodes = my_fns.custom_spline(nodes, coefs)
            res = np.sum(np.abs(distn - fit))
            if res < tol:
                if success: p.title('Good Result')
                else: p.title('Failed fit')
                print 'Knots:', n_knots, '\nCurrent residuals', res, 'tolerance', tol
                print results[-2], results[-1]
                p.plot(x, distn, marker='x', ls='None', color='g')
                p.plot(x, fit, color='b')
                p.plot(nodes, fit_nodes, marker='o', color='b', ls='None')
                p.show()
            if res < tol: pass
            else: n_knots += 1
            assert n_knots <= len(distn), 'Too many nodes'
        results = coefs, success

    elif func_type == 'split_spline':
        if not order: order = 2

        # find the size of the remaining background
        # ASSUMES extraction box is 130 pixels
        bg_pixels = np.hstack([distn[:40],distn[70:]])
        bg, bg_err = np.median(bg_pixels), np.std(bg_pixels)
        #if debug: print 'Bg found to be {:.1f}, std {:.1f}'.format(bg, bg_err)

        # find start and end indices of the spline fit
        for i, fl in enumerate(distn):
            if fl > bg + bg_err:
                start = i
                break
            else:
                start = 0
        for i, fl in enumerate(reversed(distn)):
            if fl > bg + bg_err:
                end = len(distn) - i - 1
                break
            else:
                end = 0
        start = 10
        end = 50

        x_l1, y_l1, w_l1 = x[:start], distn[:start], weights[:start]
        x_l2, y_l2, w_l2 = x[end:], distn[end:], weights[end:]
        x_sp, y_sp, w_sp = x[start:end], distn[start:end], weights[start:end]
        if len(x_sp) < 4:
            x_l1, y_l1, w_l1 = x, distn, weights
            x_l2, y_l2, w_l2 = [], [], []
            spline_coefs = None
        else:
            # Fit spline
            tol = np.mean(y_sp)*tol # set the tolerance to n% of the mean flux?
            if tol <= 0: tol=None
            if debug and full_debug: print 'Tolerance set to:', tol
            w_sp = np.ones_like(w_sp) / float(len(w_sp))
            results = UnivariateSpline(x=x_sp,y=y_sp,w=w_sp, s=tol, k=order)
            res, knots = results.get_residual(), len(results.get_knots())
            spline_coefs = results

        # Fit lines, Ax = B, x = [m, c]
        if len(x_l1) > 1:
            A1 = np.vstack([x_l1, np.ones_like(x_l1)]).T
            B1 = y_l1
            l1_fit = np.linalg.lstsq(A1,B1)
            m1, c1 = l1_fit[0]
        elif len(x_l1) == 1:
            m1, c1 = 0, y_l1[0]
        else:
            m1, c1 = None, None
        if len(x_l2) > 1:
            A2 = np.vstack([x_l2, np.ones_like(x_l2)]).T
            B2 = y_l2
            l2_fit = np.linalg.lstsq(A2,B2)
            m2, c2 = l2_fit[0]
        elif len(x_l2) == 1:
            m2, c2 = 0, y_l2[0]
        else:
            m2, c2 = None, None

        x_split = [x_l1, x_sp, x_l2]

        return spline_coefs, m1, c1, m2, c2, x_split

    elif method == 'lsq':
        if not tol:
            tol = 1.49012e-08 # fix None problem
        if not step:
            step = np.nanmean(distn)*np.sqrt(np.finfo(float).eps) #https://mail.scipy.org/pipermail/scipy-user/2010-July/025975.html
        if debug: print 'step size:', step
        results = optimize.leastsq(weight_function, coef0, args=(x, distn, weights, func_type, method), full_output=1, ftol=tol, epsfcn=step)
        if debug:
            print(results[-2])
    else:
        results = optimize.minimize(weight_function, coef0, args=(x, distn, weights, func_type, method), method=method)
    return results

def FIT(D, V_0, Q, f, fV, P, S, V, s_clip, func_type, method, debug, tol, step, order, origima, M=None, pdf=None, bg_array=None, M_DQ=None, M_CR=None):
    '''
    Fit a spatial profile to an image,
    pixels with low variance are weighted higher in the fit to
    optimize extraction of the spectrum.
    '''
    n_fits, n_success = D.shape[1], 0
    if debug or pdf:
        n_figs = 0 # number of figures stored in pdf
        p.figure() # get started

    if type(bg_array) != type(None):
        col_bg = np.median(bg_array, axis=0)
    else:
        col_bg = [None]*D.shape[1]

    # initial guesses
    coef0 = coef_dict[func_type]
    if func_type == 'gauss':
        coef0 = [np.mean(D), D.shape[0]/2, 2, 0]

    fail_list = []
    fail_type = []
    for i in range(D.shape[1]):
        # iterate along columns (wavelengths) of the IMAGE

        #print('#################\n'+'Spectral pixel {}'.format(i)+'\n#################')
        count, loop = 0, True

        # fix the wavelength to i
        distn, P_l, V_l, S_l, f_l, fV_l, bg, or_l, dq_l, cr_l = D[:,i], P[:,i], V[:,i], S[:,i], f[i], fV[i], col_bg[i], origima[:,i], M_DQ[:,i], M_CR[:,i]

        # x values, the spatial pixels
        x = np.arange(len(distn))
        # start with no outliers, or start with outliers in the mask
        if M is None:
            outliers = np.zeros(distn.shape, dtype=bool)
            M_l = np.ones_like(distn)
        else:
            # outliers has True where there is a bad pixel
            # M has 0 where there is a bad pixel, otherwise 1
            M_l = M[:,i]
            outliers = np.logical_not(M_l)

        while loop:
        # loop over sigma clipping procedure until can ignore all outlier pixels
            count += 1
            #print('Iteration {}'.format(count))
            # find weights, low variance are favouredlogg
            if bg: # check that not background dominated
                flux_pix = distn > 2*bg
                # if just bg, skip fitting
                if np.count_nonzero(flux_pix) < 0.01*len(flux_pix):
                    P_l, success = np.ones_like(distn), True
                    results = success, None
                    fit = False
                else: fit = True
            else: fit = True

            if fit:

                #if i == 65: debug = True
                results = FIT_single(x=x, distn=distn, P_l=P_l, V_l=V_l, outliers=outliers, coef0=coef0, func_type=func_type, method=method, debug=debug, tol=tol, step=step, order=order, bg=bg, i=i)
                #if i == 65: debug=False
                weights = np.divide(np.square(P_l), V_l)
                weights = weights / np.sum(weights)

                if func_type == 'spline':
                    success, coef = results # this is actually a spline object
                elif func_type == 'custom_spline':
                    coef, success = results
                elif func_type == 'split_spline':
                    coef  = results[:-1]
                    x = results[-1] # split into 3 parts
                    success = True
                    spline_coefs, m1, c1, m2, c2, x_split = results
                elif method == 'lsq':
                    coef = results[0]
                    success = results[-1] in [1,2,3,4]
                    #print success
                else:
                    coef = results['x']
                    success = results['success']
                    #print success
                if not success:
                    if func_type != 'spline':
                        #logger.warning('Optimal extraction fitting failed (col {}), using straight line'.format(i))
                        fail_list.append(i); fail_type('LINE')
                        P_l = np.ones_like(distn)
                    else:
                        try:
                            # Optimal extraction fitting failed, using previous column fit
                            fail_list.append(i)
                            results = old_results
                            success, coef = old_results
                            assert success and coef != None, 'Previous fits also failed'
                            P_l = my_fns.spline(x, coef)
                            fail_type.append('PREVIOUS')
                        except (UnboundLocalError, AssertionError): # when this is the first column, use straight line
                            #logger.info('Optimal extraction fitting failed (col {}), using straight line'.format(i))
                            resuls = (False, None)
                            P_l = np.ones_like(distn)
                            fail_type.append('LINE')
                else:
                    # this is the updated spatial fit at this wavelength
                    n_success += 1
                    P_l = fn_dict[func_type](x, coef)

            # enforce positivity
            P_l[P_l < 0] = 0
            P_l_unscaled = P_l.copy()

            # If the fit doesn't succeed weight all pixels equally
            # normalization at each wavelength, replace all zeros with equal weighting on good pix
            if np.nansum(P_l) == 0 or not success:
                P_l = np.ones_like(P_l)
                P_l_unscaled = P_l.copy()
            # considered masking bad pixels here, or apply after as a condition on P
            P_l = np.divide(P_l, float(np.nansum(P_l)))

            # update variance
            V_l = estimate_variance(f=f_l, P=P_l, S=S_l, V_0=V_0, Q=Q)

            if s_clip:
                new_outliers = clip_sigma(distn, f_l, P_l, S_l, V_l, s_clip)
                if M is not None:
                    new_outliers = np.logical_or(new_outliers, np.logical_not(M_l))
            else:
                new_outliers = outliers

            # update spectrum
            f_l, fV_l = optimized_spectrum(D=distn, S=S_l, P=P_l, V=V_l, M=M_l)

            # Plots, if wanted
            if np.array_equal(new_outliers, outliers) or count > 30:
                if (debug and i % 1 == 0) or (pdf and i % 2 == 0): # just puts limits on the numbers (and max(distn) > 1000)
                    n_figs += 1
                    p.subplot(2,2,n_figs)

                    p.title('Spectral pixel {}, bg {:.1f}'.format(i, bg))
                    p.xlabel('Spatial pixel')
                    p.ylabel('Electrons')
                    p.plot(or_l, alpha=0.5, color='k', label='Original', ls='--')
                    for j, point in enumerate(distn):
                        if new_outliers[j] and M_l[j] == 1:
                            # sigma clipped pixel
                            colour ='r'
                            mec = colour
                            marker = 'o'
                            mfc = 'None'
                        elif dq_l[j] == 0:
                            # DQ masked pixel
                            colour = 'k'
                            mec = colour
                            marker = 'o'
                            mfc = colour
                        elif cr_l[j] == 0:
                            # CR masked pixel
                            colour = 'y'
                            mec = colour
                            marker = 'o'
                            mfc = colour
                        else:
                            colour = 'g'
                            mec = colour
                            marker='x'
                            mfc = colour
                        p.plot(j,point,marker=marker,mfc=mfc,mec=mec, ls='None',color=colour)
                    p.plot(j, point, marker='x', color='g', label='Smoothed', ls='None')
                    ax = p.gca()
                    ax.set_autoscale_on(False)
                    if np.any(dq_l == 0):
                        p.plot(-1,0,marker='o', mec='k', mfc='k', ls='None', label='DQ pixels')
                    if np.any(cr_l == 0):
                        p.plot(-1,0,marker='o', mec='y', mfc='y', ls='None', label='CR pixels')
                    if np.any(np.logical_not(M_l) != new_outliers):
                        print np.count_nonzero(np.logical_not(M_l) != new_outliers), ' pixels clipped'
                        p.plot(-1,0,marker='o', mec='r', mfc='None', ls='None', label='Sigma Clipped')
                    if func_type == 'split_spline':
                        ind1 = len(x_split[0])
                        ind2 = ind1+len(x_split[1])
                        ind3 = ind2+len(x_split[2])
                        p.plot(range(ind1), P_l_unscaled[:ind1], color='c', label='Line fit')
                        p.plot(np.array(range(ind2-ind1))+ind1, P_l_unscaled[ind1:ind2], color='b', label='Spline Fit' )
                        p.plot(np.array(range(ind3-ind2))+ind2, P_l_unscaled[ind2:], color='c')
                        ax = p.gca()
                        ax.set_autoscale_on(False)
                        p.plot([ind1,ind1], [-10**6,10**6], color='b', ls='--')
                        p.plot([ind2,ind2], [-10**6,10**6], color='b', ls='--')
                    else: p.plot(P_l_unscaled, label='Fit', color='b')
                    p.legend(loc=0, fontsize='small')

                    # Plot the residuals

                    p.subplot(2,2,n_figs+2)
                    residuals = (distn - P_l_unscaled)
                    rms = np.sqrt(np.mean(residuals**2))
                    p.title('Spectral pixel {}, rms {:.1f}'.format(i, rms))
                    p.xlabel('Spatial pixel')
                    p.ylabel('Difference (electrons)')
                    p.plot(residuals, marker='o', ls='None', color='b')
                    ax = p.gca()
                    ax.set_autoscale_on(False)
                    p.plot([0,500], [0,0], ls='--', color='k', alpha=0.5)

                    # Plot weights
                    #p.subplot(3,2,n_figs+4)
                    #p.title('Weights')
                    #p.plot(weights, marker='o', ls='None', color='k')

                    if n_figs == 2:
                        p.tight_layout()
                        if pdf is None or pdf is False:
                            p.show()
                        else:
                            pdf.savefig() # save the page
                            #pdf.close()
                        p.close()
                        fig = p.figure(1)
                        fig.set_canvas(p.gcf().canvas)
                        n_figs=0
                    # coef0 = coef
                    # could keep old fit, risk of negative feedback

                loop = False
                # done!
            else:
                outliers = new_outliers
                # and loop
        if debug:
            logger.info('Stored {} figures to pdf file'.format(n_figs))

        old_results = results

        # store results in the original arrays
        P[:,i] = P_l.copy()
        V[:,i] = V_l.copy()
        f[i] = f_l
        fV[i] = fV_l
    if n_success/float(n_fits) >= 0.9: logger.info('{} successes out of {} total fits'.format(n_success, n_fits))
    else: logger.warning('Warning: {} column fits failed out of {} total fits'.format(n_fits-n_success, n_fits))

    logger.warning('Optimal extraction fitting failed for columns: {}'.format(fail_list))

    return P, V, f, fV


'''
Median filter take from:
https://gist.github.com/bhawkins/3535131
All credit to bhawkins, miss you brother.
'''
def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    if type(k) == float and int(k) == k: k = int(k)
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.nanmedian (y, axis=1)

# RUN ALGORITHM #

def extract_spectrum(D, S, V_0, Q, V=None, s_clip=16, s_cosmic=25, func_type='spline', method='lsq', debug=False, tol=None, step=None, order=2, M_DQ=None, M_CR=None, k_col=None, k_row=None, pdf_file=None, skip_fit=False, bg=None, logger=None):
    '''
    Extract spectrum using either a poly or gauss fit.
    Toggle cosmic ray removal by setting s_cosmic to None or
    the sigma threshhold.

    V: initial variance estimate, otherwise calculated
    s_clip: size of sigma clipping in spatial fit (optimal extraction)
    s_cosmic: size of sigma clipping in cosmic ray removal (oe)
    func_type: function to be fit to spatial dstn
    method: lsq for least squares, CG etc. for other (in scipy.optimize.minimize)
    M: mask of bad pixels. 0 where bad, else 1.
    k: length of median filter

    Output is the spectrum, variances as well as the spatial profile and variance (P, V)
    in case the input was not the exact spectrum (such as one where masked pixels have been meaned).

    debug: plot a few fits to the spectra
    pdf_file: where to store them

    skip_fit: to test whether simply using the flux as a profile is better
    '''
    if logger is None: logger = logging.getLogger()
    globals()['logger'] = logger

    if pdf_file:
        assert pdf_file, 'If debugging need to specify an output file'
        #print('Fitting {} to spatial distribution with optimization method {}.'.format(func_type, method))
        #print('Cosmic ray removal is set to: {}'.format(bool(s_cosmic)))
        pdf = PdfPages(pdf_file)
    else:
        pdf = None

    # Interpolate over bad pixels
    D[np.logical_not(np.isfinite(D))] = 0
    V[np.logical_not(np.isfinite(V))] = 0
    origima = D.copy()

    if M_DQ is None:
        M_DQ = np.ones_like(D)
        if M_CR is None: M = None; M_CR = np.ones_like(D)
        else: M = M_CR
    else: M = np.logical_and(M_DQ, M_CR)

 
    if False:
        if M is not None:
            # interpolate over bad pixels in D marked by M==0 along dispersion direction
            mask = M.astype(bool) # False where bad
            interp_spec = []
            # for each row
            for spec, spec_mask in zip(D, mask):
                if not spec_mask.any():
                    # all bad
                    interp_spec.append(spec)
                else:
                    ind = np.arange(len(spec))
                    t_ind = ind[spec_mask] # good indices
                    t_spec = spec[spec_mask]
                    # use an interpolation based on the good pixels
                    spec = np.interp(ind, t_ind, t_spec)
                    interp_spec.append(spec)
            D = np.vstack(interp_spec)
        else:
            M = np.ones_like(D)
    else:
        D[np.logical_not(np.isfinite(D))] = 0


    if not k_row is None:
        # Do a median smooth along each row
        smooth_spec = []
        for row in D:
            row = medfilt(row, k_row)
            smooth_spec.append(row)
        D = np.vstack(smooth_spec)

    if not k_col is None:
        # Do a median smooth along each column
        smooth_spec = []
        for col in D.T:
            col = medfilt(col, k_col)
            smooth_spec.append(col)
        D = np.vstack(smooth_spec).T

    if type(S) in [int, float]:
        S = np.ones_like(D) * S
        # lets you input a number for a median bg
    if bg is None:
        bg = S
    elif type(bg) in [int, float]:
        bg = np.ones_like(D) * bg

    if V is None:
        V = initial_variance_estimate(D=D, V_0=V_0, Q=Q)
    f, fV = estimate_spectrum(D, S, V)

    P = initial_profile_estimate(D, S, f)  #D-S/f, hardly need a function call

    # find spatial profile and variance estimatesTrue
    # can not use M mask for this fit, have already smoothed and interpolated bad pixels
    P, V, f, fV = FIT(D, V_0, Q, f, fV, P, S, V, s_clip, func_type, method, debug, tol, step, order, origima=origima, M=M, pdf=pdf, bg_array=bg, M_DQ=M_DQ, M_CR=M_CR)
    #V = estimate_variance(f=f, P=P, S=S, V_0=V_0, Q=Q)

    n, loop = 0, True
    # D has been median smoothed so unlikely to find CR hits this way, do externally
    # or maybe apply to original image
    while s_cosmic and loop:

        # remove cosmic ray hit
        f, fV = optimized_spectrum(origima, S, P, V, M)
        D, M, loop = clip_cosmic(D=origima, f=f, P=P, S=S, V=V, s_clip=s_cosmic, M=M)

        '''
        P, V, f, fV = FIT(D, V_0, Q, f, fV, P, S, V, s_clip, func_type, method, debug, tol, step, order, origima=origima, M=M, pdf=pdf, bg_array=bg)
        # Smooth the splines
        if not k is None:
            # Now do a median smooth along each row
            smooth_spec = []
            for row in P:
                row = medfilt(row, k)
                smooth_spec.append(row)
            P = np.vstack(smooth_spec)
        '''
        # and loop if found any cosmic hits
        #if loop: print('Cosmic ray loop {}...'.format(n+1))
        if loop: n += 1
        if n == 30:
            logger.warning('Terminated CR clipping at 30 iterations')
            loop=False
    logger.info('Clipped {} cosmic rays in OE'.format(n))
    # end

    if skip_fit:
        V = initial_variance_estimate(D=D, V_0=V_0, Q=Q)
        f, fV = estimate_spectrum(D, S, V)
        P = initial_profile_estimate(D, S, f)

    # Now that we have the spatial profile, apply it to the original image
    # so as to not actually include bad pixels in the final results use M mask
    f, fV = optimized_spectrum(origima, S, P, V, M)

    temp_M = np.ones_like(M)
    temp_M[np.logical_not(np.isfinite(origima))] = 0
    f2, fV2 = optimized_spectrum(origima, S, P, V, temp_M)

    #p.title('OE Spectrum')
    #p.plot(f)
    #p.plot(f2)
    #p.show()
    #view(M, cmap='binary', cbar=False)

    if debug:
        p.subplot(2,1,1)
        p.plot(f)
        p.ylabel('Flux')
        p.title('Extracted spectrum')
        p.subplot(2,1,2)
        p.plot(fV)
        p.xlabel('Spectral Pixel')
        p.ylabel('Variance')
        p.title('Extracted spectrum variance')
        p.tight_layout()
    if pdf is None: p.show()
    else: pdf.savefig()

    if hasattr(pdf, 'close'):
        pdf.close()
        del pdf


    return f, fV, P, V
