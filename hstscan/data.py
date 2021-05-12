#provides all imports
import my_fns as f
from my_fns import pyfits, np, types, p, subprocess, os
from scipy import integrate
import timecorr, scipy

def read_conf_file(fname):
    '''
    Read in configuration options for pipeline.
    Return a dictionary with values.
    File should be tab delimited with # for comment lines.
    '''
    kwargs = {}
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('#') and line.strip() != '':
                key, val = line.split()
                key, val = key.strip(), val.strip()
                key = key.lower()
                if val.lower() == 'none':
                    val = None
                elif val.lower() == 'nan':
                    val = np.nan
                elif val.lower() == 'true':
                    val = True
                elif val.lower() == 'false':
                    val = False
                elif key in ['n_masks','cr_tol','s','v_0','q','s_clip','s_cosmic','fit_tol']:
                    val = float(val)
                elif key in ['skip_start', 'skip_end', 'psf_h', 'cr_x', 'cr_y', 'object_ind', 'dq_mean_width', 'drift_width']:
                    val = int(val)
                elif key == 'dq_flags':
                    # 4: Bad detector pixel, 32: Unstable photometric response, 256: Saturated, 512: Bad flat field
                    if val == 'None': 
                        val = None
                    else:
                        val = val.split(',')
                        val = [ int(v) for v in val ]
                elif key == 'cr_replace' or key=='dq_replace':
                    if val.lower() == 'nan': val = np.nan
                    elif val.lower() in ['mean', 'median']: pass
                    else:
                        try: val = float(val)
                        except ValueError: pass # string
                else:
                # unexpected config parameter
                    try: val = float(val)
                    except ValueError: pass
                kwargs[key] = val
    return kwargs

class Single_ima():
    '''
    One subexposure from the ima fits file.
    '''
    def __init__(self, SCI=pyfits.ImageHDU(0), ERR=pyfits.ImageHDU(0), DQ=pyfits.ImageHDU(0), SAMP=pyfits.ImageHDU(0), TIME=pyfits.ImageHDU(0), nobj=1):

        # do some weird things to make sure the image data is read into the memory
        junk = [SCI.data, ERR.data, DQ.data, SAMP.data, TIME.data]

        self.SCI = SCI
        self.ERR = ERR
        self.DQ = DQ
        self.SAMP = SAMP
        self.TIME = TIME

        #this is for reference when composing Single_ima objects
        self.nobj = nobj

        #keep track of reductions done
        self.trimmed = False

    def copy(self):
        '''
        Generates a copy of the object.
        '''
        return Single_ima(self.SCI.copy(),self.ERR.copy(),self.DQ.copy(),self.SAMP.copy(),self.TIME.copy())

    def __add__(self,exposure_2):
        '''
        Adding Single_ima objects add pixel values and stores rms of errors.
        '''
        nobj = self.nobj + exposure_2.nobj
        SCI = pyfits.PrimaryHDU(self.SCI.data + exposure_2.SCI.data,self.SCI.header)
        ERR_data = np.sqrt(np.square(self.ERR.data)*self.nobj + np.square(exposure_2.ERR.data)*exposure_2.nobj)
        ERR = pyfits.PrimaryHDU(ERR_data,self.ERR.header)
        DQ = pyfits.PrimaryHDU(np.zeros_like(self.SCI.data)) #empty object, can't realistically track these
        SAMP = self.SAMP
        TIME = self.TIME
        return Single_ima(SCI,ERR,DQ,SAMP,TIME,nobj=nobj)

    def __sub__(self,exposure_2):
        '''
        Subtracting Single_ima objects takes the diff in pixel values and stores rms of errors.
        '''
        nobj = self.nobj + exposure_2.nobj
        SCI = pyfits.PrimaryHDU(self.SCI.data - exposure_2.SCI.data,self.SCI.header)
        ERR_data = np.sqrt(np.square(self.ERR.data)*self.nobj + np.square(exposure_2.ERR.data)*exposure_2.nobj)
        ERR = pyfits.PrimaryHDU(ERR_data,self.ERR.header)
        DQ = pyfits.PrimaryHDU(np.zeros_like(self.SCI.data)) #empty object, can't realistically track these
        SAMP = self.SAMP
        TIME = self.TIME
        return Single_ima(SCI,ERR,DQ,SAMP,TIME,nobj=nobj)

    def remove_bad_pix(self, int_flags=[4,32,512], replace=np.NAN, width=1):
        '''
        Removes pixels that have been flagged as bad according to calwf3.
        Replaces pixels with "replace" (default NaN).
        Input flags should be pure powers of 2 integers, according to WFC3 Handbook guidelines:
        http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/appendixE2.html
        or (since that link seems broken)
        http://www.stsci.edu/hst/wfc3/documents/handbooks/currentDHB/Chapter3_calibration4.html#547338

        4: bad detector pixel
        32: IR unstable pixel
        (256: Saturated pixel)
        512: bad flat-field value
        '''
        if int_flags is None: return np.zeros_like(self.SCI.data).astype(bool)

        if self.DQ.data is None:
            try:
                self.SCI.data[self.mask] = replace
            except AttributeError:
                raise Exception('The file has no DQ data or mask.')

        mask = np.sum([ self.DQ.data/flag % 2 == 1 for flag in int_flags ], axis=0).astype(bool)

        self.mask = mask
        if replace == 'mean' or replace == 'median':
            self.SCI.data[mask] = np.nan
            self.mean_nan_pix(replace=replace, width=width)
        elif replace is not None:
            self.SCI.data[mask] = replace

        return mask

    def mean_nan_pix(self, width=1, replace='mean'):
        '''
        New mean nan pix
        Remove all NaNs or infs, replace with local median of pixels
        within width (in pixels)
        '''
        image = self.SCI.data
        nans = np.logical_not(np.isfinite(self.SCI.data))
        images = []
        for shiftx in np.arange(-width,width+1):
            for shifty in np.arange(-width,width+1):
                if shiftx == 0 and shifty == 0: continue
                _image = np.roll(image, shifty, axis=0)
                _image = np.roll(_image, shiftx, axis=1)
                images.append(_image)
        if replace == 'mean': local_mean = np.nanmean(images, axis=0)
        elif replace == 'median': local_mean = np.nanmedian(images, axis=0)
        clean_image = np.where(nans, local_mean, image)
        self.SCI.data = clean_image

    def mean_mask_pix(self, mask, width=1, replace='mean'):
        '''
        Based off new mean nan pix
        Remove all masked pix, replace with local median of pixels
        within width (in pixels)
        '''
        image = self.SCI.data
        images = []
        for shiftx in np.arange(-width,width+1):
            for shifty in np.arange(-width,width+1):
                if shiftx == 0 and shifty == 0: continue
                _image = np.roll(image, shifty, axis=0)
                _image = np.roll(_image, shiftx, axis=1)
                images.append(_image)
        if replace == 'mean': local_mean = np.nanmean(images, axis=0)
        elif replace == 'median': local_mean = np.nanmedian(images, axis=0)
        clean_image = np.where(mask, local_mean, image)
        self.SCI.data = clean_image

    def trim_pix(self, n=5):
        '''
        Remove n pixels from each edge.
        There is a 5 reference pixel border around each image.
        '''
        self.SCI.data = self.SCI.data[n:,n:]
        self.SCI.data = self.SCI.data[:-n,:-n]
        self.ERR.data = self.ERR.data[n:,n:]
        self.ERR.data = self.ERR.data[:-n,:-n]
        if type(self.DQ.data) != 'NoneType':
            self.DQ.data = self.DQ.data[n:,n:]
            self.DQ.data = self.DQ.data[:-n,:-n]
        else:
            pass #no DQ data to trim
        self.trimmed = True

class Data_ima():
    '''``
    Creates object containing all the reads from the exposure _ima file.
    '''

    def __init__(self,filename, hst_file='/home/jacob/Project_1/js41_hst.vec',
                 tai_file='/home/jacob/Project_1/tai_utc.txt', bjd=True):
        self.filename = filename
        self.rootname = filename.split('/')[-1].split('_')[0]

        file_type = filename.split('_')
        assert file_type[-1] == 'ima.fits', '{} has the wrong file type; it should be _ima.fits'.format(filename)

        with pyfits.open(filename, memmap=False) as HDUList:
            # memmap forces you to keep every fits file open which is not possible when
            # analysing >70 files per visit.
            self.Primary = HDUList[0]
            mjd_ut = np.mean([self.Primary.header['EXPSTART'],self.Primary.header['EXPEND']])
            if mjd_ut < 2400000.5:
                jd_utc = mjd_ut + 2400000.5 # undo mjd correction
            else: # it must have been in JD
                jd_utc = mjd_ut
            RA, DEC = self.Primary.header['RA_TARG'], self.Primary.header['DEC_TARG']

            # Now do the timing corrections
            if bjd:
                # jd -> bjd
                bjd_dt = timecorr.suntimecorr(RA / 180. * np.pi, DEC / 180. * np.pi, np.array(jd_utc), hst_file)

                # utc -> tdb
                tdb_dt = timecorr.jdutc2jdtdb(jd_utc, taifile=tai_file)
                dt = bjd_dt + tdb_dt
                self.t = jd_utc + dt/60./60./24. # in days
                self.dt = dt # timing offset in seconds
                self.t_units = 'BJD_TT'
            else:
                self.t = jd_utc
                self.dt = 0
                self.t_units = 'JD_UTC'

            # Store these in the header
            self.Primary.header['t'] = (self.t, 'calculated time')
            self.Primary.header['t_units'] = (self.t_units, 'units of calculated time')

            #place each exposure in its own object
            reads, i = [], 1
            while i < len(HDUList):
                [SCI, ERR, DQ, SAMP, TIME] = HDUList[i:i+5]
                read = Single_ima(SCI,ERR,DQ,SAMP,TIME)
                reads.append(read)
                i += 5
            self.reads = reads

            # store scan direction
            self.POSTARG2 = HDUList[0].header['POSTARG2']

    def close(self):
        del self

class Single_red(Single_ima):
    def __init__(self,SCI,ERR,DQ,SAMP,TIME,nobj=1):
        Single_ima.__init__(self,SCI,ERR,DQ,SAMP,TIME,nobj)
        self.mask = DQ.data.astype(bool)
        self.bg = SAMP.data

class Data_red(Data_ima):

    def __init__(self,filename, bjd=None):
        self.filename = filename
        self.rootname = filename.split('/')[-1].split('_')[0]
        file_type = filename.split('_')
        if not file_type[-1] == 'red.fits':
            raise_with_traceback(InputError('Wrong file type.'))

        with pyfits.open(filename, memmap=False) as fits_file:
            self.fits_file = fits_file
            self.Primary = fits_file[0]

            #place each exposure in its own object
            subexposures, i = [], 1
            while i < len(fits_file):
                [SCI, ERR, MASK, BG, TIME] = fits_file[i:i+5]
                subexposure = Single_red(SCI,ERR,MASK,BG,TIME)
                subexposures.append(subexposure)
                i += 5
            self.subexposures = subexposures

class Data_flt():
    '''Break up an flt file into separate extensions and add methods'''

    def __init__(self,filename, hst_file='/home/jacob/Project_1/js41_hst.vec',
                 tai_file='/home/jacob/Project_1/tai_utc.txt', bjd=True):
        self.filename = filename
        self.rootname = filename.split('/')[-1].split('_')[0]
        file_type = filename.split('_')
        if not file_type[-1] == 'flt.fits':
            raise_with_traceback(InputError('Wrong file type.'))

        #flt type fits files have information in the primary and the image data in the next HDU
        with pyfits.open(filename, memmap=False) as fits_file:
            self.fits_file = fits_file

            mjd_ut = np.mean([fits_file[0].header['EXPSTART'],fits_file[0].header['EXPEND']])
            if mjd_ut < 2400000.5:
                jd_utc = mjd_ut + 2400000.5 # undo mjd correction
            else: # it must have been in JD
                jd_utc = mjd_ut
            RA, DEC = fits_file[0].header['RA_TARG'], fits_file[0].header['DEC_TARG']

            if bjd:
                # Now do the timing corrections
                # jd -> bjd
                bjd_dt = timecorr.suntimecorr(RA / 180. * np.pi, DEC / 180. * np.pi, np.array(jd_utc), hst_file)
                # 'js41_hst.vec' is the horizons ephemeris file for HST covering observation range
                # utc -> tdb
                tdb_dt = timecorr.jdutc2jdtdb(jd_utc, taifile=tai_file)
                dt = bjd_dt + tdb_dt
                self.t = jd_utc + dt/60./60./24. # in days
                self.dt = dt # timing offset in seconds
                self.t_units = 'BJD_TT'

            else:
                self.t = jd_utc
                self.dt = 0
                self.t_units = 'JD_UTC'

            # do some weird things to make sure the image data is read into the memory
            junk = map(lambda x: x.data, fits_file)

            #store all the extensions
            self.Primary = fits_file[0]
            self.SCI = fits_file[1]
            self.ERR = fits_file[2]
            self.DQ = fits_file[3]
            self.SAMP = fits_file[4]
            self.TIME = fits_file[5]
            #self.WCSCORR = fits_file[6]

            #place this in a read
            [SCI, ERR, DQ, SAMP, TIME] = fits_file[1:6]

            self.reads = [Single_ima(SCI,ERR,DQ,SAMP,TIME)]


class Data_drz():
    '''Break up a drz file into separate extensions and add methods'''

    def __init__(self,filename, hst_file='/home/jacob/Project_1/js41_hst.vec',
                 tai_file='/home/jacob/Project_1/tai_utc.txt', bjd=True):
        self.filename = filename
        file_type = filename.split('_')
        if not file_type[-1] == 'drz.fits':
            raise_with_traceback(InputError('Wrong file type.'))

        with pyfits.open(filename, memmap=False) as fits_file:
            self.fits_file = fits_file

            #store all the extensions
            self.Primary = fits_file[0]
            self.SCI = fits_file[1]
            self.WHT = fits_file[2]
            self.CTX = fits_file[3]
            self.HDRTAB = fits_file[4]
            self.SCI.data # otherwise the file is closed? since the data are never accessed

            mjd_ut = np.mean([self.Primary.header['EXPSTART'],self.Primary.header['EXPEND']])
            if mjd_ut < 2400000.5:
                jd_utc = mjd_ut + 2400000.5 # undo mjd correction
            else: # it must have been in JD
                jd_utc = mjd_ut
            RA, DEC = self.Primary.header['RA_TARG'], self.Primary.header['DEC_TARG']

            # Now do the timing corrections
            if bjd:
                # jd -> bjd
                bjd_dt = timecorr.suntimecorr(RA / 180. * np.pi, DEC / 180. * np.pi, np.array(jd_utc), hst_file)

                # utc -> tdb
                tdb_dt = timecorr.jdutc2jdtdb(jd_utc, taifile=tai_file)
                dt = bjd_dt + tdb_dt
                self.t = jd_utc + dt/60./60./24. # in days
                self.dt = dt # timing offset in seconds
                self.t_units = 'BJD_TT'
            else:
                self.t = jd_utc
                self.dt = 0
                self.t_units = 'JD_UTC'
            



########################################
#             Load in data             #
#            and properties            #
########################################


def which_class(filename):
    '''Return the appropriate Class with which to instantiate a file.'''
    class_dict = {'drz':Data_drz, 'flt':Data_flt, 'ima':Data_ima, 'red':Data_red}
    split_name = filename.split('_')
    if split_name[-1][:-5] in class_dict:
        return class_dict[split_name[-1][:-5]]
    else:
        raise 'Unsupported file type '+ split_name[-1][:-5]

def load(filename, **kwargs):
    '''Load in data as appropriate class instance.'''
    if not filename.split('_')[-1].endswith('.fits'):
        return Data_ima(filename+'_ima.fits', **kwargs)
    Class = which_class(filename)
    return Class(filename , **kwargs)

def load_all_ima(system = 'GJ-1214', source_file='input_image.lis', data_dir='/home/jacob/hst_data/', visits=False,
                 direction='a'):
    '''
    Load in all the data as objects.

    Returns a generator of loaded orbits in order of time
    direction: a for all, f for forward scans, r for reverse scans only
    '''
    source_dir = data_dir + system + '/'
    with open(source_dir+source_file,'r') as g:
        lines = g.readlines()
    lines = [line.split('\t') for line in lines if not line.startswith('#')]
    if direction != 'a':
        lines = [line for line in lines if line[-1].startswith(direction)]
    lines = [line[0] for line in lines if line[1].startswith('G')]
    for fname in lines:
	file = source_dir+fname+'_ima.fits'
        yield load(file)

def load_all_red(system = 'GJ-1214', source_file='input_image.lis', data_dir='/home/jacob/hst_data/', visits=False,
                 direction='a'):
    '''
    Load in all the data as objects.

    Returns a generator of loaded orbits in order of time
    direction: a for all, f for forward scans, r for reverse scans only
    '''
    source_dir = data_dir + system + '/'
    with open(source_dir+source_file,'r') as g:
        lines = g.readlines()
    lines = [line.split('\t') for line in lines if not line.startswith('#')]
    if direction != 'a':
        lines = [line for line in lines if line[-1].startswith(direction)]
    lines = [line[0] for line in lines if line[1].startswith('G')]
    for fname in lines:
        yield load(source_dir+fname+'_red.fits')

def read_visit_names(source_file='input_image.lis', data_dir = '/net/glados2.science.uva.nl/api/jarcang1/GJ-1214/', append=''):
    '''
    Read in the rootnames of each exposure, seperated into visits by the 5th and 6th character of the rootname.
    Include in output the time of observation and the scan direction so that the files don't need to be re-opened.
    '''
    with open(data_dir+source_file,'r') as g:
        lines = g.readlines()
        lines = [ line for line in lines if not line.startswith('#') ]
        indexes = np.array([i for i, line in enumerate(lines) if line.split('\t')[1].startswith('F')])
        visits = [lines[indexes[i]+1:indexes[i+1]] for i in range(len(indexes)-1)]
        # index by where a new direct image is taken
        visits = [visit for visit in visits if len(visit) > 1]
        times = [ [float(line.split('\t')[2]) for line in visit if line.split('\t')[1].startswith('G')] for visit in visits]
        directions = [ [line.split('\t')[3] for line in visit if line.split('\t')[1].startswith('G')] for visit in visits]
        visits = [ [line.split('\t')[0]+append for line in visit if line.split('\t')[1].startswith('G')] for visit in visits]
        # just the names
        i, sets, time_sets, dir_sets = 0, [], [], []
        while i < len(visits):
            visit, time, direction = visits[i], times[i], directions[i]
            vis_id = visit[-1][4:6]
            n = 1 # number of lists combined, start with 1
            for j, next in enumerate(visits[i+1:]):
                if next[0][4:6] == vis_id:
                    visit = visit + next
                    time = time + times[i+1+j]
                    direction = direction + directions[i+1+j]
                    vis_id = next[-1][4:6]
                    n += 1
                    continue
                break
            sets.append(visit)
            time_sets.append(time)
            dir_sets.append(direction)
            i += 1 + n
    return sets, time_sets, dir_sets

def read_scan_direction(exposure, t, logger):
    """
    Reads the scan direction and scan rate

    :param exposure: A .fits exposure
    :param t: The option paramters with which the reduction is run
    :return: scan direction and the scan rate.
    """
    # Find scan direction from positional offset
    if exposure.Primary.header['POSTARG2'] >= 0.:
        scan_direction = +1;
        logger.info('Forward scan')
    else:
        scan_direction = -1;
        logger.info('Reverse scan')
    if t.scanned:
        if not 'scan_rate' in dir(t):
            scan_rate = exposure.Primary.header['SCAN_RAT']
            if scan_rate == 0.:
                logger.warning('SCANNED=True while exposure scan rate is zero')
        else:
            if 'SCAN_RAT' in exposure.Primary.header:
                assert abs(t.scan_rate - exposure.Primary.header['SCAN_RAT']) / t.scan_rate < 0.01, \
                    'Scan rates do not match (input {}, fits {})'.format(t.scan_rate,
                                                                         exposure.Primary.header['SCAN_RAT'])
            scan_rate = t.scan_rate
    else:
        scan_rate = 0.
    return scan_direction, scan_rate

def find_star_in_catalogue(catalogue, di_name, t, logger):
    """
    Finds the central coordinates in a catalogue file

    :param catalogue: The name of the catalogue file
    :param di_name: name of the direct image file
    :param t: The options
    :return: The x-coordinate, y-coordinate of the direct image plus the postarg1, postarg2 and PA_V3 arguments of the
             direct image.
    """
    print "catalogue", catalogue
    try:
        if os.path.exists(t.source_dir + di_name + '_ima.fits'):
            direct_image = Data_ima(t.source_dir + di_name + '_ima.fits', hst_file=t.hst_eph_file,
                                    tai_file=t.tai_utc_file, bjd=False)
            logger.warning('Catalogue file: {}'.format(di_name + '_ima.fits'))
        else:
            direct_image = Data_flt(t.source_dir + di_name + '_flt.fits', hst_file=t.hst_eph_file,
                                    tai_file=t.tai_utc_file, bjd=False)
            logger.warning('Catalogue file: {}'.format(di_name + '_flt.fits'))
    except IOError:  # no _ima or _flt for _drz files of multiple direct images
        direct_image = Data_drz(t.source_dir + di_name + '_drz.fits', bjd=False)
        logger.warning('No catalogue file found for {}'.format(di_name + '_drz.fits'))
    di_size = direct_image.reads[0].SCI.shape[0]
    if np.log2(di_size) % 1 != 0:
        # Remove reference pixles from direct image
        di_size -= 10  # ref pix

    di_ps1, di_ps2, di_pav3 = direct_image.Primary.header['POSTARG1'], direct_image.Primary.header['POSTARG2'], \
                              direct_image.Primary.header['PA_V3']
    with open(catalogue, 'r') as cat:
        lines = cat.readlines()
        objects = [line[:-1].split() for line in lines if line[0] != '#']
        objects = [[float(val) for val in obj] for obj in objects]
        objects = sorted(objects, key=lambda obj:obj[0])
        print("########################")
        obj = objects[t.object_ind]

        SEx, SEy = obj[1], obj[2]
        # SEx, SEy = cal.center_of_flux(t.source_dir+di_name, SEx, SEy, size=10)
        # Location of the direct image
        x_di, y_di = SEx, SEy  # w.r.t to reference pixel of direct image exposure
        logger.debug('Direct image location of ({},{})'.format(x_di, y_di))
    return x_di, y_di, di_ps1, di_ps2, di_pav3, direct_image

def read_spec(fname, wmin=-np.inf, wmax=np.inf):
    with open(fname,'r') as g:
        g.readline() # header# header
        g.readline() # second header line
        lines = g.readlines()
        lines = [line.strip('\n') for line in lines]
    lines = [[float(p) for p in line.split('\t')] for line in lines]
    lines = [line for line in lines if float(line[0]) > wmin and float(line[0]) < wmax]

    if len(lines[0]) == 3:
        waves, fluxes, errors = zip(*lines)
        return np.array(waves), np.array(fluxes), np.array(errors)
    elif len(lines[0]) == 2:
        waves, fluxes = zip(*lines)
        return np.array(waves), np.array(fluxes), np.zeros_like(fluxes)
    else:
        return zip(*lines)

def broadband_fluxes(files=None, system='GJ-1214',source_dir='/home/jacob/hst_data/', wmin=-np.inf, wmax=np.inf, plot=False, direction='a', all_plot=False, save_extension='_spec.txt', shift=False, peak=False, sane=None, shift_file=None, **kwargs):
    
    from dispersion import interp_sanity_quad
    import reduction as r
    reload(r)
    
    with open(source_dir+files) as g:
        lines = g.readlines()
    lines = [line.split('\t') for line in lines if not line.startswith('#') and not line.strip()=='']
    lines = [line for line in lines if line[1].startswith('G')]
    if direction[0] in ['f', 'r']:
        lines = [line for line in lines if line[3].startswith(direction)]
    rootnames = [line[0] for line in lines]
    times = [float(line[2]) for line in lines]
    directions = [line[3][0] for line in lines]

    if shift_file:
        with open(shift_file, 'r') as g:
            sf_lines = g.readlines()
        sf_rootnames, sf_shifts, sf_errs = zip(*[sf_line.split() for sf_line in sf_lines])
        sf_shifts, sf_errs = np.array(sf_shifts).astype(float), np.array(sf_errs).astype(float)
        file_shifts = dict(zip(sf_rootnames, sf_shifts))

    all_flux, all_waves, all_times, all_errors = [], [], [], []
    for rootname, time in zip(rootnames, times):
        # look for the spec file
        for file_ in os.listdir(source_dir):
            if file_.endswith(save_extension) and file_.startswith(rootname) and not file_.endswith('_subs'+save_extension):
                # got em!
                waves, fluxes, errors = read_spec(source_dir+file_, wmin=-np.inf, wmax=np.inf)
                # dont use wave limits here as spectrum may have drifted
                all_flux.append(fluxes)
                all_waves.append(waves)
                all_times.append(time)
                all_errors.append(errors)
                break
            else: 
                pass

    # Interpolate to first spectrum in the visit/orbit
    template_x, template_y = all_waves[-1], np.median(all_flux, axis=0)
    # median doesnt work for direction='a'
    interp_spectra, interp_errors, shifts = [], [], []
    for waves, fluxes, err, rootname in zip(all_waves, all_flux, all_errors, rootnames):
        if not shift:
            shift_y = np.interp(template_x, waves, fluxes)
            shift_err = err; shift = 0
        else:
            # Correct shift between exposures
            if shift_file:
                shift = file_shifts[rootname]
            else:
                if not peak:
                    shift, _ = r.spec_pix_shift(template_x, template_y, waves, fluxes, norm=True)
                else:
                    i0, i1 = np.argmin(abs(template_x-1.14)), np.argmin(abs(template_x-1.6))
                    shift, _ = r.spec_pix_shift(template_x[i0:i1], template_y[i0:i1], waves[i0:i1], fluxes[i0:i1], norm=True)
            shift_y = np.interp(template_x, template_x-shift, fluxes)
            shift_err = np.interp(template_x, template_x-shift, err)
        interp_spectra.append(shift_y)
        interp_errors.append(shift_err)
        shifts.append(shift)
    if all_plot:
        for waves, fluxes, interp, shift in zip(all_waves, all_flux, interp_spectra, shifts):
            p.plot(template_x, fluxes, color='g')
            p.plot(template_x, interp, color='b')
            p.plot(template_x, template_y, color='k', alpha=0.5, ls='--')
            p.title('Shift: {:.6g} microns or ~{:.2f} pixels'.format(shift, shift/0.0045))
            p.show()
        p.title(len(interp_spectra))
        for spec in interp_spectra:
            p.plot(template_x, spec)
        p.show()

    # Find the amount of each pixel used, for error calculation and simple integral
    in_bin = []
    bin_sizes = np.hstack([np.diff(template_x), template_x[-1]-template_x[-2]])
    wv_left = template_x - bin_sizes/2.; wv_right = template_x + bin_sizes/2.
    for left, wv, right in zip(wv_left, template_x, wv_right):
        
        if left >= wmax:
            # out
            in_bin.append(0)
        elif left >= wmin:
            # left within new bin
            if right <= wmax:
                # Fully in
                in_bin.append(1.)
            else:
                # right > wvmax, right edge
                in_bin.append( (wmax-left) / (right-left))
        elif right > wmin:
            # left < wvmin, left edge
            in_bin.append( (right-wmin) / (right-left))
        else:
            # Fully out
            in_bin.append(0)
    in_bin = np.array(in_bin)

    broad_time = all_times
    broad_flux = [ np.sum(fluxes*in_bin) for fluxes in interp_spectra ]
    broad_errors = [ np.sqrt(np.sum((errors**2*in_bin))) for errors in interp_errors ]

    if plot:
        for t, fl, direction in zip(broad_time, broad_flux, directions):
            if direction == 'r':
                color = 'r'
            elif direction == 'f':
                color = 'b'
            p.plot(t, fl, ls='None', marker='o', color=color, **kwargs)
    return np.array(broad_time), np.array(broad_flux), np.array(broad_errors), np.array(directions), np.array(shifts)

def get_sub_times(files, source_dir):

    fstub = files.split('.')[0] 
    # Creat sub_times file
    if not os.path.isfile(source_dir+fstub+'_sub_times.dat'):
        with open(source_dir+files) as g:
            lines = g.readlines()
        lines = [line.split('\t') for line in lines if not line.startswith('#') and not line.strip()=='']
        lines = [line for line in lines if line[1].startswith('G')]
        rootnames = [line[0] for line in lines]

        times = []
        for rootname in rootnames:
            ima = rootname+'_ima.fits'
            try:
                ima = load(source_dir+ima)
            except IOError:
                _source_dir = '/'.join(source_dir.split('/')[:-2])+'/'
                ima = load(_source_dir+ima)
            times.append([str(read.SCI.header['ROUTTIME']) for read in ima.reads[:-2]])

        try:
            with open(source_dir+fstub+'_sub_times.dat', 'w') as g:
                for rootname, time in zip(rootnames, times):
                    g.write(rootname+'\t'+'\t'.join(time)+'\n') 
        except:
            f.silentremove(source_dir+fstub+'_sub_times.dat')
            raise

    # Read in sub times
    with open(source_dir+fstub+'_sub_times.dat', 'r') as g:
        lines = g.readlines()
        lines = [ line.split() for line in lines ]
    sub_times = {}
    for line in lines:
        sub_times[line[0]] = line[1:]
    return sub_times

def broadband_sub_fluxes(files=None, system='GJ-1214',source_dir='/home/jacob/hst_data/', wmin=-np.inf, wmax=np.inf, direction='a', save_extension='_spec.txt'):
    from reduction import spec_pix_shift
    with open(source_dir+files) as g:
        lines = g.readlines()
    lines = [line.split('\t') for line in lines if not line.startswith('#') and not line.strip()=='']
    lines = [line for line in lines if line[1].startswith('G')]
    if direction[0] in ['f', 'r']:
        lines = [line for line in lines if line[3].startswith(direction)]
    rootnames = [line[0] for line in lines]
    times = [float(line[2]) for line in lines]
    directions = [line[3][0] for line in lines]

    # load in sub times:
    sub_times = get_sub_times(files, source_dir)

    all_flux, all_waves, all_times, all_errors = [], [], [], []
    for rootname, time in zip(rootnames, times):
        # look for the spec file
        for file in os.listdir(source_dir):
            if file.endswith(save_extension) and file.startswith(rootname+'_subs'):
                # got em!
                data = np.loadtxt(source_dir+file, skiprows=2)
                waves = data[:,0]
                errors = data[:,2::2]
                fluxes = data[:,1::2]
                times = [float(t) for t in sub_times[rootname]]

                all_flux.append(fluxes)
                all_waves.append(waves)
                all_times.append(times)
                all_errors.append(errors)
                break

    template_x, template_y = all_waves[-1], np.median(np.array(all_flux)[-1], axis=-1) # median subs of last exposure
    #template_x, template_y = all_waves[-1], np.median(all_flux[-8:], axis=0)
    # median doesnt work for direction='a'
    all_interp_spectra, all_interp_errors, all_shifts = [], [], []
    for waves, _fluxes, _errors in zip(all_waves, all_flux, all_errors):
        interp_spectra, interp_errors, shifts = [], [], []
        for fluxes, errors in zip(np.array(_fluxes).T,np.array(_errors).T):        
            shift = spec_pix_shift(template_x, template_y, waves, fluxes, debug=False)
            shift_y = np.interp(template_x, template_x+shift, fluxes)
            shift_err = np.interp(template_x, template_x+shift, errors)
            interp_spectra.append(shift_y)
            interp_errors.append(shift_err)
            shifts.append(shift)
        all_interp_spectra.append(interp_spectra); all_interp_errors.append(interp_errors); all_shifts.append(shifts)

    broad_flux = [ [ np.sum([ _f for w, _f in zip(wvs, fl) if w > wmin and w < wmax ]) for fl in fls ] for wvs, fls in zip(all_waves, all_interp_spectra)]
    broad_time = all_times
    broad_errors = [ [ np.sqrt(np.sum(np.square([ e for w, e in zip(wvs, er) if w > wmin and w < wmax ]))) for er in errs ] for wvs, errs in zip(all_waves, all_interp_errors)]

    return np.array(broad_time), np.array(broad_flux), np.array(broad_errors), np.array(directions)

#################
# Viewing tools #
#################

def view_frame_image(count, units='', direct_image=False, cbar=True, show=True, title='', xlabel='Spectral Pixel', ylabel='Spatial Pixel',
                     origin='lower', Return=False, bg_mask=None,**kwargs):
    if direct_image or bg_mask is not None:
        if bg_mask is not None:
            fig, ax = p.subplots(figsize=(10,10))
            im = ax.imshow(count, origin=origin, **kwargs)
            bg_alphas = 0.3 * np.ones_like(bg_mask)
            ax.imshow(np.array([bg_mask.T, bg_mask.T, bg_mask.T, bg_alphas.T]).T, origin=origin, cmap='binary_r')
        else:
            fig, ax = p.subplots()
            im = ax.imshow(count, origin=origin,**kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if cbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            Cbar = p.colorbar(im, cax=cax)
            Cbar.set_label(units)
        if show:
            p.show()
        elif Return:
            return ax
    else:
        im = p.imshow(count, origin=origin,**kwargs)
        p.xlabel(xlabel)
        p.ylabel(ylabel)
        p.title(title)
        if cbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(p.gca())
            cax = divider.append_axes("right", size="5%", pad=0.05)
            Cbar = p.colorbar(im, cax=cax)
            Cbar.set_label(units)
        if show:
            p.show()
        elif Return:
            return im


def view_3d_image(image, xlabel='Spectral Pixel', ylabel='Spatial Pixel', zlabel='electrons/s', title='', cmap='jet',
                  show=True):

    from mpl_toolkits.mplot3d import Axes3D
    fig = p.figure()
    ax = fig.add_subplot(111, projection='3d')

    Z = image
    (x_len, y_len) = image.shape
    x, y = np.arange(x_len), np.arange(y_len)
    Y = np.repeat(y,x_len).reshape(y_len,x_len).transpose()
    X = np.repeat(x,y_len).reshape(x_len, y_len)
    ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0)
    #ax.w_xaxis.gridlines.set_lw(0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    p.title(title)
    if show:
        p.show()

def save_frame_image(count, filename, **kwargs):
        '''
        Plot coloured image of spatially scanned data frame.
        Pixel at (spatial,spectral) has value 'count'.
        '''
        #fig, ax = p.imshow(count)
        silentremove('/scratch/jarcang1/Documents/plots/'+filename)
        p.imsave('/scratch/jarcang1/Documents/plots/'+filename, count, vmin=np.min(count),vmax=np.max(count),**kwargs)

        return None

        # this was because i thought matplotlib couldnt do images
        from scipy import misc as smp

        #data[512,512] = [254,0,0]       # Makes the middle pixel red
        #data[512,513] = [0,0,255]       # Makes the next pixel blue

        img = smp.toimage( count )       # Create a PIL image
        img.save(filename)

        return None

        # this garbage isn't any use, matplotlib does not work with pixels
        ax = p.gcf().gca()
        colours = list(p.cm.rainbow(count.flatten()))
        cmap = p.cm.get_cmap('rainbow')
        y = np.repeat(np.arange(spatial),spatial)
        x = np.tile(np.arange(spectral), spectral)
        s = ax.scatter(x,y,c=count.flatten(),cmap=cmap,marker='s',vmin=-10,vmax=500,s=40,**kwargs)
        cb = p.colorbar(s)
        if colorbarlabel:
                cb.set_label(colorbarlabel)
        p.ylim([0,spatial])
        p.xlim([0,spectral])
        if show:
            p.show()

def plot_data(x, y, title='', ylabel='', xlabel='', **kwargs):
    """
    Plots data

    :param x: The x-data
    :param y: The y-data
    :param title: The title
    :param ylabel: The label on the y-axis
    :param xlabel: The label on the x-axis
    :param kwargs: Extra arguments
    :return:
    """
    if not type(x[0]) in [int, float]:
        for i,X in enumerate(x):
            kwargsnew = {key: kwargs[key][i] for key in kwargs.keys()}
            p.plot(X, y[i], **kwargsnew)
    else:
        p.plot(x, y, **kwargs)
    p.title(title)
    p.xlabel(xlabel)
    p.ylabel(ylabel)




def hist_image(image=np.zeros(1), show=True, ymax=None, xlabel=None, ylabel=None, title=None, filename=None, hist_data=None, **kwargs):
    image = image.copy()
    image[np.isnan(image)] = 0
    pix_vals = image.flatten()
    if hist_data:
        hist, bin_edges = hist_data
    else:
        hist, bin_edges = np.histogram(a=pix_vals, **kwargs)
    h = [bin_edges[i+1] - bin_edges[i] for i in range(len(bin_edges)-1)]
    center = [bin_edge + h for bin_edge in bin_edges[:-1]]
    left = bin_edges[:-1]
    if show:
        fig, ax = p.subplots()
        ax.bar(left, hist, align='center', width=h)
        if ymax:
            p.ylim([0,ymax])
        if xlabel:
            p.xlabel(xlabel)
        if ylabel:
            p.ylabel(ylabel)
        if title:
            p.title(title)
        if filename:
            fig.savefig(filename)
        p.show()


def plot_backgrounds(source_file=None, data_dir = '/net/glados2.science.uva.nl/api/jarcang1/GJ-1214/', HST=False, direction='a', show=True):
    bgs, bg_errs, times, colors = [], [], [], []
    sub_bgs ,sub_errs, sub_times, sub_colors = [], [], [], []
    if source_file:
        with open(data_dir+source_file, 'r') as g:
            lines = g.readlines()
        lines = [line.split('\t') for line in lines if not line.startswith('#')]
        lines = [line for line in lines if line[1].startswith('G')]
        if direction[0] in ['f', 'r']:
            lines = [line for line in lines if line[3].startswith(direction)]
        rootnames = [line[0] for line in lines]
        fnames = [ data_dir + rootname + '_red.fits' for rootname in rootnames ]
        for fname in fnames:
                exp = load(fname)
                exp_bgs ,exp_errs, exp_times, exp_colors = [], [], [], []
                for sub in exp.subexposures:
                    bg = sub.SCI.header['BG']
                    bg_err = sub.SCI.header['BG_ERR']
                    exp_bgs.append(bg)

                    exp_errs.append(bg_err)

                    t = sub.SCI.header['ROUTTIME']
                    if HST:
                        t = np.mod(t, 96./24/60)*60*24/96. - 0.4
                        if t < 0.: t += 1.
                    exp_times.append(t)

                    POSTARG2 = exp.Primary.header['POSTARG2']
                    if POSTARG2 < 0:
                        color = 'r'
                    else:
                        color = 'b'
                    exp_colors.append(color)

                bgs.append(np.nanmean(exp_bgs))
                bg_errs.append(np.sqrt(np.mean(np.square(bg_err))))
                times.append(exp_times[0])
                colors.append(exp_colors[0])
                sub_bgs.append(exp_bgs)
                sub_times.append(exp_times)
                sub_colors.append(exp_colors)
                sub_errs.append(exp_errs)

        p.title('Background for each exposure')
        if HST: p.xlabel('HST phase')
        else: p.xlabel('Time (days)')
        p.ylabel('Electrons')
        for t, bg, bg_err, color in zip(times, bgs, bg_errs, colors):
            p.plot(t, bg, color=color, marker='o')
        if show: p.show()
        return times, bgs, bg_errs, colors

    else:
        print 'Warning loading a lot of files'
        for file in os.listdir(data_dir):
            if file.endswith('_red.fits'):
                exp = load(data_dir+file)
                for sub in exp.subexposures:
                    try:
                        bg = sub.SCI.header['BG']
                    except KeyError:
                        bg = np.median(sub.SAMP.data)
                    bgs.append(bg)
                    t = sub.SCI.header['ROUTTIME']
                    if HST: t = np.mod(t, 96./24/60)*60*24/96.
                    times.append(t)
                    POSTARG2 = exp.Primary.header['POSTARG2']
                    if POSTARG2 < 0:
                        color = 'r'
                    else:
                        color = 'b'
                    colors.append(color)
                p.plot(t, bg, color=color, marker='o')
    p.show()
    return times, bgs, colors

############################################
#             Save reduced data            #
############################################

def create_sub_history(t):
    history = ['~~~~~~~~~~~~~~~~~~~~~~~~~~~','CUSTOM REDUCTION PIPELINE','~~~~~~~~~~~~~~~~~~~~~~~~~~~']
    history.append('flat field correction was set to {}'.format(t.flat_field))
    history.append('background removal set to {}'.format(t.bg))
    history.append('dispersion solution correction set to {}'.format(t.dispersion))
    history.append('local cosmic ray removal set to {}'.format(t.cr_local))
    if t.cr_local:
        history.append('            - with tolerances {}, {} replacing with {}'.format(t.cr_tolx, t.cr_toly,t.cr_replace))
    history.append('DQ flagged pixels replaced with {}'.format(t.dq_replace))
    return history

def write_reduced_fits(subexposures, Primary, t, dest_dir=None):

    # add history to primary header
    Primary = Primary.copy()

    history = create_sub_history(t)
    for line in history:
        Primary.header['HISTORY'] = line

    Primary.header['xpix'] = subexposures[0].xpix

    hdu = pyfits.HDUList(Primary)

    for subexposure in subexposures:
        data = subexposure.SCI.data
        mask = subexposure.mask
        zeros = np.zeros_like(data)

        bg = np.multiply(np.ones_like(data), subexposure.bg)

        SCI = pyfits.ImageHDU(data=data, header=subexposure.SCI.header, name='SCI')
        hdu.append(SCI)
        ERR = pyfits.ImageHDU(data=subexposure.ERR.data, header=subexposure.ERR.header, name='ERR')
        hdu.append(ERR)
        MASK = pyfits.ImageHDU(data=mask.astype(float), header=None, name='MASK')
        hdu.append(MASK)
        BG = pyfits.ImageHDU(data=bg, header=None, name='BG')
        hdu.append(BG)
        TIME = pyfits.ImageHDU(data=subexposure.TIME.data, header=subexposure.TIME.header, name='TIME')
        hdu.append(TIME)


    fname = dest_dir + Primary.header['ROOTNAME'] + '_red.fits'
    f.silentremove(fname)
    print('Saving reduced exposure to {}...'.format(fname))
    hdu.writeto(fname)
    hdu.close()



############################################
#            aXe data directory            #
############################################

def clean_axe_dir(axe_dir = '/net/glados2.science.uva.nl/api/jarcang1/aXe/'):
    '''
    Remove temporary file copies from aXe directory.
    '''
    dirs = ['CONF',  'DATA',  'DRIZZLE',  'IMDRIZZLE',  'OUTPUT',  'save']

    for file in os.listdir(axe_dir+'DATA'):
        if not file.startswith('default'):
            os.remove(axe_dir+'DATA/'+file)
    for file in os.listdir(axe_dir+'save'):
        os.remove(axe_dir+'save/'+file)

def clean_grism_drz(system='WASP-18', data_dir='/home/jacob/hst_data/'):
    number = 0
    for file in os.listdir(data_dir+system):
        if file.endswith('_drz.fits'):
            with pyfits.open(data_dir+system+'/'+file, memmap=False) as HDU:
                if HDU[0].header['FILTER'].startswith('G'):
                    number += 1
                    os.remove(data_dir+system+'/'+file)
    print('{} files removed'.format(number))

def make_driz_list(data_dir='/home/jacob/hst_data/'):
    '''
    Collect a list of all the drizzle combined images from visits.
    Identify visists as those with more than one exposure used in drizzle.
    '''
    f.silentremove(data_dir+'/visit_driz.lis')
    cnt = 0
    with open(data_dir+'/visit_driz.lis','w') as g:
        for file in os.listdir(data_dir):
            if file.endswith('_drz.fits'):
                with pyfits.open(data_dir + file, memmap=False) as HDU:
                    if HDU[0].header['FILTER'].startswith('F'):
                        g.write(file+'\n')
                        cnt += 1
            elif file.endswith('_flt.fits'):
                drzfile = file.replace('_flt.fits', '_drz.fits')
                imafile = file.replace('_flt.fits', '_ima.fits')
                if not os.path.isfile(data_dir + drzfile):
                    with pyfits.open(data_dir + file, memmap=False) as HDU:
                        if HDU[0].header['FILTER'].startswith('F'):
                            g.write(file + '\n')
                            cnt += 1
                    if os.path.isfile(data_dir + imafile):
                        if not os.path.isdir(data_dir + 'not-used_direct_images'):
                            os.mkdir(data_dir + 'not-used_direct_images')
                        os.rename(data_dir + imafile, data_dir + 'not-used_direct_images/' + imafile)

    assert cnt > 0, 'No driz combined images found for data in {}'.format(data_dir)

def make_input_image_list(data_dir='/home/jacob/hst_data/'):
    os.nice(20)
    all_lines = []
    for file in sorted(os.listdir(data_dir)):
        if file.startswith('visit') and file.endswith('.lis') and not file.startswith('visit_driz'):
            with open(data_dir+file, 'r') as g:
                lines = g.readlines()
            lines = [line for line in lines if not line.startswith('#')]
            for line in lines:
                all_lines.append(line)
    all_lines.sort()
    output = data_dir+'/input_image.lis'
    f.silentremove(output)
    with open(output, 'w') as g:
        for line in all_lines:
            g.write(line)

def make_input_image_lists(input_file=None, data_dir='/home/jacob/hst_data/WASP-18/', prop_str='iccz',
                           conf_file='/Users/bob/Documents/PhD/KELT-9b/transmission_reduction.conf'):
    '''
    List all exposures, sorted into orbits with corresponding direct images.
    '''
    os.nice(20)
    #for no in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10','11', '12', '13', '14', '15']:
    nos = []
    conf_kwargs = read_conf_file(conf_file)
    if input_file is None:
        for file in os.listdir(data_dir):
            if file.startswith(prop_str) and file.endswith('_ima.fits'):
                nos.append(file[4:6])

    else:
        with open(data_dir+input_file, 'r') as g:
            lines = g.readlines()
            lines = [ line.strip() for line in lines ]
            for file in lines:
                nos.append(file[4:6])

    nos = sorted(list(set(nos)))
    print nos
    for no in nos:
        print 'Starting visit', no
        j = 0
        line_dat = []
        for file in os.listdir(data_dir):
            if file.startswith(prop_str+no) and (file.endswith('_ima.fits') or file.endswith('_drz.fits')):
                if file.endswith('_drz.fits'):
                    imafile = file.replace('_drz.fits', '_ima.fits')
                    if imafile in os.listdir(data_dir):
                        #There's an _ima equivalent to this file, so ignore this _drz file
                        continue
                print file
                j += 1
                exp = load(data_dir + file, bjd=True, hst_file=conf_kwargs['hst_eph_file'],
                            tai_file=conf_kwargs['tai_utc_file'])
                time = exp.t
                filt = exp.Primary.header['FILTER']
                rootname = exp.Primary.header['ROOTNAME']
                if filt.startswith('F'):
                    direction = 'Direct'
                else:
                    POSTARG2 = exp.Primary.header['POSTARG2']
                    if POSTARG2 >= 0:
                        direction = 'reverse'
                    else:
                        direction = 'forward'
                line_dat.append([str(rootname), str(filt), str(time), direction])
                print 'Completed file', j, 'for visit', no
        line_dat.sort()
        lines = '\n'.join(['\t'.join(dat) for dat in line_dat])
        f.silentremove(data_dir+'/visit_'+no+'.lis')
        with open(data_dir+'/visit_'+no+'.lis', 'w') as g:
            g.write(lines)
            g.write('\n')
        print '#####################\nVisit', no, 'completed.\n#####################'

def find_catalogue(rootname, logger, data_dir='/home/jacob/hst_data/'):
    '''
    Find direct image (orbit) catalogue corresponding to a given
    grism exposure.
    '''
    cat_rootname = None
    with open(data_dir+'input_image.lis','r') as g:
        lines = g.readlines()
    lines = [line for line in lines if not line.startswith('#') and not line.strip()=='']

    cat = None
    for line in lines:
        l_rootname, l_filter, l_expstart, l_scan = line.split('\t')
        if l_filter.startswith('F'):
            # Direct image filter
            cat = data_dir + l_rootname + '_flt_1.cat'
            cat_rootname = l_rootname
        if rootname in l_rootname:
            if cat is None: continue
            if not os.path.isfile(cat):
                logger.warning('No catalogue file found for {}'.format(cat))
                # then catalogue may be for visit drizzled file
                catalogue_split = cat.split('_')
                catalogue_split[-3] = catalogue_split[-3][:-3] + '011'
                cat = '_'.join(catalogue_split)
                if not os.path.isfile(cat):
                    catalogue_split[-3] = catalogue_split[-3][:-3] + '010'
                    cat = '_'.join(catalogue_split)
                logger.warning('Replaced catalogue with {}'.format(cat))
            return cat, cat_rootname
    assert cat_rootname is not None, "No direct image found for {}".format(rootname)
    return cat, cat_rootname

def find_reference_exp(rootname, data_dir='/home/jacob/hst_data/WASP-18/'):
    '''
    Find exposure taken directly after
    direct image (calibration image)
    To use a reference exposures
    '''
    cat_rootname = None
    with open(data_dir+'input_image.lis','r') as g:
        lines = g.readlines()
    lines = [line for line in lines if not line.startswith('#') and not line.strip()=='']

    store_next = False
    for line in lines:
        l_rootname, l_filter, l_expstart, l_scan = line.split('\t')
        if store_next:
            ref_rootname = l_rootname
        if l_filter.startswith('F'):
            # Direct image filter
            cat = data_dir + l_rootname + '_flt_1.cat'
            cat_rootname = l_rootname
            store_next = True
        else:
            store_next = False
        if rootname in l_rootname:
            return ref_rootname



#####################################
## Tools for reading emcee results ##
#####################################
import emcee

def save_chain(chain, fname, nwalkers, nsteps, ndim, coefs_names):
    h = open(fname, 'w')
    h.write(','.join(coefs_names))
    h.write('\n')
    h.write('{},{},{}'.format(nwalkers, nsteps, ndim))
    h.write('\n')
    for i in range(nsteps):
        position = chain[:,i,:]
        for k in range(position.shape[0]): # for each walker
            h.write("{0:4d} {1:s}\n".format(k, " ".join((str(val) for val in position[k]))))
    h.close()

# So the reader has to take each line as a walkers position and when it has read all walkers
# it moves onto next step in chain.
def emcee_chain_reader(fname):
    with open(fname, 'r') as g:
        step = 0
        coefs = g.readline()[:-1]
        coefs_names = coefs.split(',')
        line = g.readline()[:-1]
        nwalkers, nsteps, ndim = [ int(val) for val in line.split(',')]
        print 'Chain fit of: {}'.format(', '.join(coefs_names))
        print '{} walkers, {} steps, {} dim'.format(nwalkers, nsteps, ndim)
        chain = np.empty([nwalkers, nsteps, ndim])
        for i, line in enumerate(g):
            line = line[:-2] # trim /n
            line = line.split(' ')
            line = [ item for item in line if item != '' ]
            j = int(line[0]) # walker no.
            assert j < nwalkers, 'Incorrect nwalkers for given file.'

            position = [float(val) for val in line[1:]]
            assert len(position) == ndim, 'Incorrect ndim for given file.'
            chain[j,step,:] = position
            if j == nwalkers-1: step += 1
    return {'chain':chain, 'coefs_names':coefs_names, 'nwalkers':nwalkers, 'ndim':ndim, 'nsteps':nsteps}



##########################
####### Orbit cats #######
##########################

def create_orbit_cats_gauss(target='GJ-1214', source_dir='/home/jacob/hst_data/'):
    '''
    Fit a gaussian to direct image to create visit catalogue.
    Use same format as SExtractor for catalogue files.
    '''
    from lmfit import minimize, Parameters
    import astropy
    from astropy import modeling

    data_dir = source_dir
    if not os.path.exists(data_dir+'input_image.lis'):
        make_input_image_list(data_dir=source_dir)
    if not os.path.exists(data_dir+'visit_driz.lis'):
        make_driz_list(data_dir=source_dir)

    Gaussian2D = astropy.modeling.functional_models.Gaussian2D

    with open(data_dir+'visit_driz.lis', 'r') as driz_list:
        for line in driz_list:
            fname = line[:-1]
            dest_file = data_dir + fname.split('_')[0]+'_flt_1.cat'
            flt_fname  = data_dir + fname.split('_')[0]+'_flt.fits'

            di = data.load(flt_fname)
            image = di.SCI.data.copy()

            params = Parameters()
            params.add_many(('amplitude', np.max(image), True, 0.), \
                            ('x_mean', image.shape[1]/2, True, 0, image.shape[1]), \
                            ('y_mean', image.shape[0]/2, True, 0, image.shape[1]), \
                            ('x_stddev', 10, True, 0), ('y_stddev', 10, True, 0))

            size = image.shape[0]
            x = np.repeat(np.arange(0,size),size).reshape(size,size).T
            y = np.repeat(np.arange(0,size),size).reshape(size,size)

            def residuals(params, image, x, y):
                model = Gaussian2D.evaluate(x, y, amplitude=params['amplitude'], x_mean=params['x_mean'], y_mean=params['y_mean'], \
                                            x_stddev=params['x_stddev'], y_stddev=params['y_stddev'], theta=0)
                return (image - model).flatten()

            out = minimize(residuals, params, args=(image, x, y))
            params = out.params
            fit_params = params

            params.pretty_print()
            view(image, show=False, cmap='binary_r')
            p.title(fname)
            ax = p.gca()
            ax.set_autoscale_on(False)
            p.plot(params['x_mean'].value, params['y_mean'].value, marker='x', color='r')
            p.xlim([params['x_mean'].value-25, params['x_mean'].value+25])
            p.ylim([params['y_mean'].value-25, params['y_mean'].value+25])
            p.show()

            x_image, y_image = fit_params['x_mean'].value, fit_params['y_mean'].value
            x_std, y_std = fit_params['x_stddev'].value, fit_params['y_stddev'].value

            line = '\t'.join(['1', str(x_image), str(y_image), '0', '0', \
                        str(x_std), str(y_std), '0.0', '0', '0', '0', '0.0']) + '\n'

            with open(dest_file, 'w') as g:
                g.write('# 1  NUMBER  Running object number\n# 2  X_IMAGE  Object position along x  [pixel]\n')
                g.write('# 3  Y_IMAGE  Object position along y  [pixel]\n# 4  X_WORLD  Barycenter position along world x axis  [deg]\n')
                g.write('# 5  Y_WORLD  Barycenter position along world y axis  [deg]\n# 6  A_IMAGE  Profile RMS along major axis  [pixel]\n')
                g.write('# 7  B_IMAGE  Profile RMS along minor axis  [pixel]\n# 8  THETA_IMAGE  Position angle (CCW/x)  [deg]\n')
                g.write('# 9  A_WORLD  Profile RMS along major axis (world units)  [deg]\n# 10 B_WORLD  Profile RMS along minor axis (world units)  [deg]\n')
                g.write('# 11 THETA_WORLD  Position angle (CCW/world-x)  [deg]\n# 12 MAG_F1384  Kron-like elliptical aperture magnitude  [mag]\n')

                g.write(line)
            # catalogue create for direct image