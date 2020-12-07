"""
This is the place for all your telescope characterstics.

Currently we only have HST WFC3 characteristics. We'll add JWST later
"""

class HST:
    def __init__(self):
        self.name   = "HST"
        self.xscale = 0.1355    #arcsec/pixel
        self.yscale = 0.1211    #arcsec/pixel
        self.w_ref1 = 1.12      #micron  #start of reliable spectrum
        self.w_ref2 = 1.63      #micron  #end of reliable spectrum

        # Data quality file flags and descriptions
        self.dq_info = {
            0: ('GOODPIXEL', 'OK'),
            1: ('SOFTERR', 'Reed-Solomon decoding error'),
            2: ('DATALOST', 'data replaced by fill value'),
            4: ('DETECTORPROB', 'bad detector pixel'),
            8: ('BADZERO', 'unstable IR zero-read pixel'),
            16: ('HOTPIX', 'hot pixel'),
            32: ('UNSTABLE', 'IR unstable pixel'),
            64: ('WARMPIX', 'unused'),
            128: ('BADBIAS', 'bad reference pixel value'),
            256: ('SATPIXEL', 'full-well or a-to-d saturated pixel'),
            512: ('BADFLAT', 'bad flat-field value'),
            1024: ('SPIKE', 'CR spike detected during ramp fitting'),
            2048: ('ZEROSIG', 'IR zero-read signal correction'),
            4096: ('TBD', 'cosmic ray detected by Astrodrizzle'),
            8192: ('DATAREJECT', 'rejected during up-the-ramp fitting'),
            16384: ('HIGH_CURVATURE', 'not used'),
            32768: ('RESERVED2', 'cant use')
        }

        # Used below
        # FILTER, dx, dxerr, dy, dyerr, shift, shifterr
        self.filt_data = [['F098M', 0.150, 0.026, 0.268, 0.030, 0.309, 0.034],
                     ['F140W', 0.083, 0.020, 0.077, 0.022, 0.113, 0.030],
                     ['F153M', 0.146, 0.022, -0.106, 0.029, 0.186, 0.036],
                     ['F139M', 0.110, 0.022, 0.029, 0.028, 0.114, 0.036],
                     ['F127M', 0.131, 0.023, -0.055, 0.024, 0.143, 0.034],
                     ['F128N', 0.026, 0.022, -0.093, 0.021, 0.095, 0.030],
                     ['F130N', 0.033, 0.014, 0.004, 0.019, 0.030, 0.024],
                     ['F132N', 0.039, 0.018, 0.154, 0.022, 0.155, 0.028],
                     ['F126N', 0.264, 0.018, 0.287, 0.025, 0.389, 0.031],
                     ['F167N', 0.196, 0.012, -0.005, 0.013, 0.200, 0.018],
                     ['F164N', 0.169, 0.022, -0.125, 0.024, 0.214, 0.032],
                     ['F160W', 0.136, 0.013, 0.046, 0.016, 0.149, 0.021],
                     ['F125W', 0.046, 0.022, 0.195, 0.023, 0.206, 0.032],
                     ['F110W', -0.037, 0.023, 0.209, 0.029, 0.214, 0.037],
                     ['F105W', 0.015, 0.023, 0.027, 0.030, 0.036, 0.038]]

    def get_wfc3_filter_offs(self, filt):
        # Different filters have different inherent direct image offsets, refer to:
        # http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2010-12.pdf
        # copied into table above
        # this is because the conf file was made using the F140W filter
        filts = [fd[0] for fd in self.filt_data]
        try:
            i = filts.index(filt)
            dx, dy = self.filt_data[i][1], self.filt_data[i][3]
            return dx, dy
        except:  # filter missing
            return None, None
