from __future__ import division
#from future.utils import raise_with_traceback
#from future.utils import bind_method
import os
import astropy.io.fits as pyfits
import subprocess
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
import types
import matplotlib.pylab as p
import operator
from scipy.optimize import leastsq

class EmptyLogger():
            def info(self, string): pass
            def warning(self, string): pass

def to_tuple(a):
    '''Convert an object to a tuple or a tuple of tuples'''
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a

def silentremove(filename):
    '''Remove a file if it exists, otherwise pass'''
    try:
        os.remove(filename)
    except OSError:
        pass

def bb(l, T):
    '''
    Calculate black-body emission for given temperature at given wavelength
    '''
    h = 6.62607004E-34
    c = 2.99792458E8
    k = 1.38064852E-23
    return 2*h*c**2 / l**5 / (np.exp(h*c/l/k/T)-1)

class Bunch(object):
    '''
    Stick all the key, values in a dictionary
    into an object.
    Thus can be accesed by Bunch(dict).key
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)

    def __iter__(self):
        attributes = [a for a in dir(self) if not a.startswith('__')]
        for attr in attributes:
            yield attr

    def __str__(self):
        return ', '.join(self)

    def __getitem__(self, key):
        return self.__dict__[key]
    def __setitem__(self, key, value):
        self.__dict__.__setitem__(key, value)

    def __len__(self):
        attributes = [a for a in dir(self) if not a.startswith('__')]
        return len(attributes)

def bunch_kwargs(toggles={}, toggle=None, kwargs={}, name='<object>', verbose=False, logger=None):
    if type(toggle) is bool:
        # apply global toggle switch
        update_toggles = dict([(key, toggle) for key in toggles if type(toggles[key]) is bool])
        toggles.update(update_toggles)
    t = Bunch(toggles)

    # stick on extra kwargs
    for key in kwargs:
        #print key, t[key], kwargs[key]
        if key not in t:
            if verbose:
                print('{} not a valid input to {}, ignored.'.format(key, name))
            else:
                pass
        else:
            t.__dict__.update({key: kwargs[key]})
    return t

def find_closest(search,iterable):
    '''Returns the index of the closest element'''
    (index, value) = min(enumerate(iterable), key=lambda i: abs(iterable[i[0]]-search))
    return index

def interpolate(scale, x, y):
    '''
    Interpolate values 'y' at points 'x' to a scale 'scale'.
    Use linear extrapolation at end points.
    '''
    def pointwise(xi):
        i1 = find_closest(xi,x)
        # assume x and y were sorted
        if xi > x[i1]:
            i2 = i1 + 1
            mark = 'a' # after
        else:
            i2 = i1 - 1
            mark = 'b' # before

        try:
            m = (y[i2] - y[i1])/(x[i2] - x[i1])
        except IndexError:
            # try extrapolating either way
            if mark == 'a':
                m = (y[i1] - y[i1-1])/(x[i1] - x[i1-1])
            elif mark == 'b':
                m = (y[i1+1] - y[i1])/(x[i1+1] - x[i1])
        return y[i1] + m*(xi-x[i1])
    return np.array(map(pointwise, scale))

def extrap1d(x,y,**kwargs):
    '''
    Use linear extrapolation around scipy interp1d.
    Source: http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range
    '''
    interpolator = interp1d(x,y,**kwargs)

    def extrapolator(xi):
        if xi < x[0]:
            return y[0]+(xi-x[0])*(y[1]-y[0])/(x[1]-x[0])
        elif xi > x[-1]:
            return y[-1]+(xi-x[-1])*(y[-1]-y[-2])/(x[-1]-x[-2])
        else:
            return interpolator(xi)
    def ufunclike(xis):
        return np.array(map(extrapolator, np.array(xis)))
    return ufunclike

# Different functional forms, used for psf shape fitting
def poly_n(x,coef):
    '''
    Returns a polynomial function with coefficients defined by coef.
    Evaluated at point or points (arraylike) x.
    coef[0]*x**(N-1) + coef[1]*x**(N-2) + ... + coef[N-2]*x + coef[N-1]

    '''
    return np.polyval(coef,x)

def gauss(x, coef):
    '''
    Coefficients are, in order, a,b,c,(d) defined by:

        y = a * exp( - (x-b)^2 / 2c^2 ) + d
    Input x can be array or single value.
    '''
    a, b, c = coef[0], coef[1], coef[2]
    try:
        d = coef[3] #optional adjust baseline
    except IndexError:
        d = 0
    return a*np.exp(-np.square(x-b) / (2.*c*c)) + d

def spline(x, coef):
    '''
    Hack to implement spline results from UnivariateSpline
    in format of other function calls.
    Here coef is the spline object
    '''
    return coef(x)

def custom_spline(x, coefs):
    '''
    coefs should be in the format [nodes, poly coefs]
    '''
    nodes, coefs = coefs

    def spline_fn(xi):
        if xi <= nodes[0]:
            ind = 0
        elif xi >= nodes[-1]:
            ind = len(nodes)-2
        else:
            for i, nd in enumerate(nodes):
                if xi < nd:
                    break
                else:
                    ind = i
        return np.sum([ coefs[ind][n]*(xi**n) for n in range(len(coefs[ind])) ])
    return np.array(map(spline_fn, x))

def split_spline(x, coef):
    '''
    Use spline with linear fit on either side (for wings).
    '''
    spline_coefs, m1, c1, m2, c2 = coef
    x1, xs, x2 = x
    out = []
    if m1 != None:
        y1 = m1*x1 + c1
        out.append(y1)
    if spline_coefs != None:
        ys = spline(xs, spline_coefs)
        out.append(ys)
    if m2 != None:
        y2 = m2*x2 + c2
        out.append(y2)
    out = np.hstack(out)
    if len(out) == 1:
        out = out[0]
    return out

def smooth_Heaviside(x, k, h):
    '''
    Coeff is k, controls the smoothing.
    As k -> inf, function -> heaviside step.

        y = h / ( 1 + exp(-2kx ) )
    '''
    return h*np.reciprocal(1 + np.exp(-2*k*x))

def custom_hill(x, coef):
    '''
    Returns a custom 'hill' function made of
    two smoothed heavisides.

    coef = [k , w , s , h , (, y)]

    Coefficients are k, w and s the smoothing
    parameter, width of the hill, start of increase and
    overall amplitude scale.
    Can additionally parse a scaling factor y that changes
    the baseline.

        y = _/---  + --\___
              up      down
    '''
    k, w, s, h = coef[0], coef[1], coef[2], coef[3]
    try:
        y = coef[4]
    except IndexError:
        y = 0

    if s < 0:
        s = 0
    if k < 0:
        k = 0
    if w < 0:
        w = 0
    up = smooth_Heaviside(x-s,k,h)
    down = -smooth_Heaviside(x-s-w,k,h)
    return up + down + y

class Spectrum():
    '''
    Basic spectrum object.
    Calling the spectrum at a given x gives you a y by linear interpolation.
    '''
    def __init__(self, x, y, x_unit=None, y_unit=None, fname=None):
        #should include a zip sort
        if len(x) != len(y):
            raise InputError('Input data should be of the same length')
        self.x = np.array(x)
        self.y = np.array(y)
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.fname = fname

    def __call__(self,x):
        '''
        Linearly interpolate points not defined.
        At endpoints simply extend last defined point.
        '''
        x = np.array(x)
        try:
            return np.array([self.__call__(xi) for xi in x])
        except TypeError:
            pass

        if x in self.x:
            index = find_closest(x,self.x)
            return self.y[index]
        # outside range returns end point, disabled
        #elif x < min(self.x):
        #    return min(self.x)
        #elif x > max(self.x):
        #    return max(self.x)

        i1 = find_closest(x,self.x)
        # assume x and y were sorted
        if x > self.x[i1]:
            i2 = i1 + 1
            mark = 'a' # after
        else:
            i2 = i1 - 1
            mark = 'b' # before

        try:
            m = (self.y[i2] - self.y[i1])/(self.x[i2] - self.x[i1])
        except IndexError:
            # try extrapolating either way
            if mark == 'a':
                m = (self.y[i1] - self.y[i1-1])/(self.x[i1] - self.x[i1-1])
            elif mark == 'b':
                m = (self.y[i1+1] - self.y[i1])/(self.x[i1+1] - self.x[i1])
        return self.y[i1] + m*(x-self.x[i1])

    def bin_to_res(self,to_res=70, pix_width=5):
        '''Re-bin the spectrum to a new resolution.'''
        # pix_width = self.R // to_res ?
        nx, ny = [], []
        for i in range(0, len(self.x), pix_width):
            nx.append(self.x[i])
            ny.append(sum(self.y[i:i+pix_width]))
        self.x = np.array(nx)
        self.y = np.array(ny)
        self.R = to_res

    def plot(self, show=False, **kwargs):
        '''Plot the spectrum'''
        p.plot(self.x,self.y,**kwargs)
        if self.x_unit:
            p.xlabel(str(self.x_unit))
        if self.y_unit:
            p.ylabel(str(self.y_unit))
        if self.fname:
            p.title(str(self.fname))
        if show:
            p.show()
            p.close()

    def __add__(s1,s2):
        '''
        Addition of two spectral objects
        Uses linear interpolation if the x values don't match
        '''
        if np.array_equal(s1.x, s2.x):
            # simply add the two y values if over the same range
            y = s1.y + s2.y
        else:
            # otherwise interpolate the second spectrum on the domain of the first
            y = s1.y + s2(s1.x)
        return Spectrum(s1.x,y,s1.x_unit,s1.y_unit,s1.name)

    def __sub__(s1,s2):
        '''
        Subtraction of two spectral objects
        Uses linear interpolation if the x values don't match
        '''
        if np.array_equal(s1.x, s2.x):
            # simply sub the two y values if over the same range
            y = s1.y - s2.y
        else:
            # otherwise interpolate the second spectrum on the domain of the first
            y = s1.y - s2(s1.x)
        return Spectrum(s1.x,y,s1.x_unit,s1.y_unit,s1.name)

    def __mul__(self,val):
        return Spectrum(self.x, self.y*val, self.x_unit,self.y_unit,self.name)
    def __rmul__(self,val):
        return Spectrum(self.x, val*self.y, self.x_unit,self.y_unit,self.name)
