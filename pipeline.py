import reduction as r
import my_fns as f
import my_errors as myerr
import data, systematics
import extraction_algorithm as ea
import calibration as cal
from my_fns import np, p, os
import shutil, logging, time, multiprocessing, sys
from multiprocessing.pool import Pool
from matplotlib.pyplot import rcParams
from scipy.optimize import leastsq
from matplotlib.backends.backend_pdf import PdfPages

import pyfits
view = data.view_frame_image

reload(data)

'''
For masking of bad pixels have just attached an array to the subexposure objects.
It's possible there are better ways to do this such as np.ma masked array.
'''

def reduce_files(source_file, system='WASP-18', data_dir='/home/jacob/hst_data/', direction='f', conf_file='/home/jacob/hst_data/WASP-18/WASP-18_red.conf', cores=None):

    source_dir = data_dir + system + '/'
    with open(source_dir+source_file,'r') as g:
        lines = g.readlines()
    lines = [line.split('\t') for line in lines if not line.startswith('#')]
    if direction != 'a':
        lines = [line for line in lines if line[-1].startswith(direction)]
    lines = [line[0] for line in lines if line[1].startswith('G')]
    files = [ source_dir+fname+'_ima.fits' for fname in lines ]


    def func(fname):
        exp = data.load(fname)
        return reduce_exposure(exp, conf_file=conf_file)

    sys.stdout = os.devnull
    sys.stderr = os.devnull

    pool = Pool(cores)
    r = pool.map_async(func, files)
    r.get()

# need to define wavelength bins and the corresponding calculated limb darkening coefficients for GJ-1214
wavelength_bins=np.array([1.135,1.158,1.181,1.204,1.228,1.251,1.274,1.297,1.320,1.343,1.366,1.389,1.412,1.435,1.458,1.481,1.504,1.527,1.550,1.573,1.596,1.619,1.642])
limb_dark_dw=np.array([0.27,0.26,0.25,0.26,0.26,0.26,0.23,0.23,0.26,0.30,0.28,0.28,0.29,0.29,0.32,0.28,0.27,0.27,0.28,0.26,0.26,0.22])


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
                elif key in ['n_masks','cr_tol','cr_sigma','cr_width','cr_thresh','dq_replace','s','v_0','q','s_clip','s_cosmic','fit_tol']:
                    val = float(val)
                elif key in ['skip_start', 'skip_end', 'psf_h', 'box_h', 'cr_x', 'cr_y', 'object_ind']:
                    val = int(val)
                elif key == 'dq_flags':
                    # 4: Bad detector pixel, 32: Unstable photometric response, 512: Bad flat field
                    val = val[1:-1].split(',')
                    val = [ int(v) for v in val ]
                elif key == 'cr_replace':
                    try: val = float(val)
                    except ValueError: pass # string
                else:
                # unexpected config parameter
                    try: val = float(val)
                    except ValueError: pass
                kwargs[key] = val
    return kwargs

def add_handlers(logger, log_file, warnings_file, level):
        fh = logging.FileHandler(log_file, mode='w') # file handling, remove exisiting log
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)

        # console output and file warnings
        wh = logging.FileHandler(warnings_file)
        wh.setLevel(logging.WARNING)
        wh.setFormatter(logging.Formatter('%(name)s - %(message)s'))
        logger.addHandler(wh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        sh.setFormatter(logging.Formatter('%(name)s - %(message)s'))
        logger.addHandler(sh)
        return logger

def reduce_exposure(exposure, toggle=None, conf_file=None, **kwargs):
    '''
    Input a data.Data_ima instance of an exposure file.
    Returns a reduced exposure.
    Reductions involves:
        - forming subexposures
        - remove cosmic rays with local (slow) or sliding median filter
        - calculate dispersion solution
        - interpolate image to common wavelength scale (slow)
        - apply flat-field correction
        - calculate and remove background
        - save to new fits file (_red.fits)

    toggle: sets the default of a process toggle to True or False
            overriden by individual toggles
    kwargs: custom toggles, assigned by keyword arguments
            possibilities are:

            BOOLS
            units: unit conversion from electrons/s to electrons
            CR_local: cosmic ray removal by taking local median
            CR_slide: CR removal using a sliding median filter, should not be used for drift scans
            flat_field: wavelength dep flat field correction
            bg: calculate and remove background using interpolation
            dispersion: interpolate to wavelength scale using dispersion solution
            debug: True to enable full debugging, otherwise just logging info
            logger: True then logs progress, else ignored

            VALUES
            data_dir: where the source fits file is stored
            object_ind: object index (brightest first) to extract from image
            CR_tol: change tolerance for local CR hits
            CR_sigma: change the sigma limit in sliding median for a CR detection
            CR_thresh: lower limit for a cosmic ray dump, default 500 electrons
            CR_replace: change the value that replaces a CR pixel
            CR_width: number of adjacent subexposures used for sliding median filter
            DQ_replace: value replacing DQ flagged bad pixels
            save_dir: destination for reduced fits file
            n_masks: number of bg star spectra expected to be masked in bg estimation
            neg_masks: number of negative persistence masks used in bg estimates
            contam_thresh: contamination threshold, pixels with more than this contamination are masked
            bg_box: height of the box around the spectrum from which to calcualate the bg
            psf_h: width of the spectrum on the spatial axis to be ignored in bg calculation
    '''
    t0 = time.time()
    # store all the possible bool toggles in an object 't'
    # default them to 'toggle' value
    # Probably also going to bung the other value options in 't'
    toggles = {'toggle':toggle, 'system': 'WASP-18', 'source_dir': '/home/jacob/hst_data/', 'read_noise': 20, 'flat_field': True, 'bg': True, \
                'n_masks': 3, 'dispersion': True, 'units': True, 'log': True, 'cr_local': True, 'cr_slide': False, 'cr_tol': 30, 'cr_sigma': 3, \
                'cr_width': 5, 'cr_thresh': 500,  'cr_replace': 'median', 'dq_replace': None, 'save_dir': None, 'contam_thresh': 0.01, \
                'debug': False, 'bg_box':200, 'bg_plot': False, 'psf_h':130, 'logger': True, 'force_no_offset': False, 'neg_masks': 1, \
                'shift_spectra': True, 'cr_plot': False, 'cr_x': 1, 'cr_y': 1, 'bg_pdf':False, 'mask_h':40, 'dq_flags': [4,32,512], 'psf_w':220, \
                'spatial_plot':False, 'disp_coef':'default', 'scanned':True, 'object_ind': 0, 'check_sextractor':False, 'fit_image':False}
    if conf_file:
        conf_kwargs = read_conf_file(conf_file)
        for kw in kwargs:
            conf_kwargs[kw] = kwargs[kw]
        full_kwargs = conf_kwargs
    else:
        full_kwargs = kwargs
    t = f.bunch_kwargs(kwargs=full_kwargs, toggle=toggle, toggles=toggles, name='reduction', verbose=True)

    # open up an exposure if you input a filename
    if type(exposure) == str:
        exposure = data.load(t.source_dir+t.system+'/'+exposure)

    # Set up logging
    if t.logger:
        log_file = t.source_dir+t.system+'/logs/'+exposure.rootname+'_red.log'
        f.silentremove(log_file)
        warnings_file = t.source_dir+t.system+'/logs/red_warnings.log'
        f.silentremove(warnings_file)
        logger = logging.getLogger(exposure.rootname+'_red')
        logger.setLevel(logging.DEBUG)
        if len(logger.handlers) == 0:
            if t.debug:
                level = logging.DEBUG
            else:
                level = logging.INFO
            logger = add_handlers(logger, log_file, warnings_file, level)
    else:
        logger = f.EmptyLogger()
    logger.info('########################################')
    logger.info('###########Starting Reduction###########')
    logger.info('########################################')
    logger.info('Data reduction pipeline performed on exposure {}'.format(exposure.filename))
    logger.info('For configuration, see pipeline conf file: {}'.format(conf_file))

    # Find scan direction from positional offset
    POSTARG1 = exposure.Primary.header['POSTARG1']
    POSTARG2 = exposure.Primary.header['POSTARG2']
    PA_V3 = exposure.Primary.header['PA_V3']
    if t.force_no_offset: POSTARG2 = 0
    if POSTARG2 >= 0:
        scan_direction = +1 # forward
        logger.info('Forward scan')
    else:
        scan_direction = -1 # reverse
        logger.info('Reverse scan')

    CCDgain = exposure.Primary.header['CCDGAIN'] # commanded gain of CCD
    masks, reads = [], []
    for read in exposure.reads:
        # Reduce the read
        read, mask = r.reduced_read(read, replace=t.dq_replace, units=t.units, CCDgain=CCDgain, int_flags=t.dq_flags)
        if t.units and read.SCI.header['BUNIT'] == 'ELECTRONS/S':
            read.SCI.header['BUNIT'] = 'ELECTRONS'
            read.ERR.header['BUNIT'] = 'ELECTRONS'
        masks.append(mask)
        reads.append(read)
            # Remove DQ flagged pixels, replace with zeros
            # Trim reference pixels
    logger.info('DQ flagged pixels replaced by {} in each read, Unit correction set to {}'.format(t.dq_replace,t.units))
    exposure.reads = reads
    exposure.masks = masks
    assert len(reads) != 0, 'No reads in file'
    if len(reads) == 1: logger.warning('Only 1 read')

    # direct image catalogue for orbit
    catalogue, di_name = data.find_catalogue(exposure.rootname, data_dir=t.source_dir+t.system+'/')
    direct_image = data.load(t.source_dir+t.system+'/'+di_name)
    di_size = direct_image.reads[0].SCI.shape[0] - 10 # ref pix
    logger.info('Direct image catalogue used: {}'.format(catalogue))
    if t.check_sextractor: print 'Direct image catalogue used: {}'.format(catalogue.split('/')[-1])
    di_ps1, di_ps2 = direct_image.Primary.header['POSTARG1'], direct_image.Primary.header['POSTARG2']
    di_pav3 = direct_image.Primary.header['PA_V3']
    with open(catalogue,'r') as cat:
        lines = cat.readlines()
        objects = [line[:-1].split() for line in lines if line[0] != '#']
        objects = [[float(val) for val in obj] for obj in objects ]
        objects = sorted(objects, key=lambda obj: obj[-1])
        obj = objects[t.object_ind]

        image_fname = t.source_dir+t.system+'/'+catalogue.split('_')[-3].split('/')[-1]+'_flt.fits'
        SEx, SEy = obj[1], obj[2]
        SEx, SEy = cal.center_of_flux(t.source_dir+t.system+'/'+di_name, SEx, SEy, size=10)
        # Location of the direct image
        x_di, y_di = SEx, SEy # w.r.t to reference pixel of direct image exposure
        #x_di, y_di = SEx + (POSTARG1- di_ps1) / 0.135, SEy + (POSTARG2 - di_ps2) / 0.121
        # correct for telescope offset between direct image and exposures using POSTARGs
        # 1st order (BEAMA) is from 15-196 pix w.r.t. direct image
        #x_order = x_di + 15
        # use center of beam for reduction.box_cut
        #x_box = x_order + int((196-15)/2.)
        # for y pixel, width of beam is not defined but y_di should be the center (or end of scan)

    if t.cr_local:
        logger.info('Removing local CRs using local mean array')
        logger.info('Tolerance set to {} sigma'.format(t.cr_tol))

    subexposures = []
    if exposure.filename.endswith('_flt.fits'): _n = 0 # only 1 read!
    else: _n = 3 # 1 for zeroth, 1 for fast read, 1 for difference

    for i in range(len(exposure.reads)-_n):

            if exposure.filename.endswith('_flt.fits'):
                subexposure = exposure.reads[i]
                DQ_mask = subexposure.remove_bad_pix(replace=t.dq_replace, int_flags=t.dq_flags)
                subexposure.DQ_mask = DQ_mask
            else:
                read2, read1 = exposure.reads[i], exposure.reads[i+1]
                DQ_mask = np.logical_or(masks[i], masks[i+1])

                # Form subexposure
                subexposure = r.create_sub_exposure(read1, read2, read_noise=t.read_noise)
                subexposure.DQ_mask = DQ_mask
                if read1.trimmed: subexposure.trimmed = True
                else: subexposure.trimmed = False

            if t.dispersion: # currently cannot be toggled off
                # Calculate offset
                if t.scanned:
                    #exp_time, scan_rate, scan_time = subexposure.SCI.header['DELTATIM'], data.get_scan_rate(exposure), subexposure.TIME.header['PIXVALUE']
                    exp_time, deltatime, scan_rate = subexposure.SCI.header['SAMPTIME'], subexposure.SCI.header['DELTATIM'], data.get_scan_rate(exposure)
                    scan_time = exp_time
                else:
                    exp_time, deltatime, scan_rate, scan_time = subexposure.SCI.header['SAMPTIME'], subexposure.SCI.header['DELTATIM'], 0., 0.

                y_len, x_len = subexposure.SCI.data.shape
                conf_file_G141 = t.source_dir+'/aXe/CONF/WFC3.IR.G141.V2.5.conf'
                flat_file_G141 = t.source_dir+'aXe/CONF/WFC3.IR.G141.flat.2.fits'
                # n = 'A' specifies first order beam

                if not t.fit_image:
                    scan_x = 0.
                    offset_x = (POSTARG1- di_ps1) / 0.135
                    # moving the telescope right moves the target right on the image as it is facing the other way
                    scan_y = (scan_direction * (scan_rate * scan_time)) / 0.121
                    offset_y = -(POSTARG2 - di_ps2) / 0.121 # in pixels
                    # moving the telescope down moves the target up on the image

                    # (V2,V3) axes are offset by 45deg to (Xpos,Ypos)
                    # so that +V3 -> +x,+y and +V2 -> -x,+y
                    # hence x_offset, y_offset = (V3 + V2)/root(2), (V3 - V2)/root(2)
                    dV3 = (PA_V3 - di_pav3) * 3600. # in arcseconds from degrees
                    dV3 = 0.
                    pa_offset_x, pa_offset_y = dV3/np.sqrt(2) / 0.135, dV3/np.sqrt(2) / 0.121 # in pix
                    offset_x += pa_offset_x
                    offset_y += pa_offset_y

                    if t.check_sextractor: print 'PA_V3 offset of {:.2f} arcseconds, ({:.1f},{:.1f}) in pix'.format(dV3, pa_offset_x, pa_offset_y)
                    #logger.debug('exposure time of {} with a scan rate of {}\"/s'.format(scan_time,scan_rate))

                    if t.shift_spectra:
                        if i == 0:
                            template = np.sum(subexposure.SCI.data, axis=0)
                        x, y = np.arange(len(template)), np.sum(subexposure.SCI.data, axis=0)
                        shift = r.spec_pix_shift(x, template, x, y, debug=False)
                        shift_y = np.interp(shift+x, x, y)
                        if False:
                            p.title(shift)
                            p.plot(x,template, color='g',label='Template')
                            p.plot(x,y,color='r', label='Original')
                            p.plot(shift+x,shift_y,color='b', ls='--', label='Shifted')
                            p.legend()
                            p.show()
                    else: shift = 0

                    # Different filters have different inherent direct image offsets, refer to:
                    #http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2010-12.pdf
                    # this is because the conf file was made using the F140W filter
                    # For WFC3/IR F130N: dx = 0.033 +- 0.014, dy = 0.004 +- 0.019
                    # For WFC3/IR F126N: dx = 0.264 +- 0.018, dy = 0.287 +- 0.025
                    # For WFC3/IR F139M: dx = 0.11 +- 0.022, dy = 0.029 +- 0.028
                    target = exposure.Primary.header['TARGNAME']
                    if target == 'GJ-1214': XOFF, YOFF = 0.033, 0.004
                    elif target == 'WASP-18': XOFF, YOFF = 0.264, 0.287
                    elif target == 'WASP-43': XOFF, YOFF = 0.264, 0.287
                    elif target in ['WASP-19CEN', 'Kepler-9', 'WASP-80'] or target.startswith('2MASS-J19021775+3824032'): XOFF, YOFF = 0.11, 0.09
                    XOFF += shift + scan_x + offset_x
                    YOFF += scan_y + offset_y
                    if subexposure.trimmed:
                        # SExtractor includes reference pixels
                        XOFF -= 5
                        YOFF -= 5

                    # Check images were on the same subarray size:
                    im_size = subexposure.SCI.data.shape[0]
                    if im_size == di_size:
                        diff = 0.
                    else:
                        diff = (di_size - im_size)/2
                        XOFF -= diff; YOFF -= diff



                    if i == 0: debug = True
                    else: debug = False
                    logger.debug('Offsets of XOFF: {} and YOFF: {} of direct image'.format(XOFF, YOFF))
                    x_order, y_order = x_di + XOFF, y_di + YOFF
                    # reference pix for 1st order
                    subexposure.x_order, subexposure.y_order = x_order, y_order
                    # center of 1st order, correct for drift in y so that ypix is in the center of the scan not edge
                    subexposure.xpix = x_order + int((192-15)/2.)
                    subexposure.ypix = y_order + (scan_direction * (scan_rate * deltatime)) / 0.121 /2.

                    if t.check_sextractor:
                        print 'Direct image is at ({}, {}) on a {} array'.format(x_di, y_di, di_size)
                        print 'direct image is {}x{}, exposure is {}x{} so image moves by ({},{})'.format(di_size,di_size,im_size,im_size,diff,diff)
                        if t.shift_spectra: print 'calculated shift of {} in the x-direction'.format(shift)
                        print 'between DI and EXP, telesope moves by ({}, {})'.format(offset_x, offset_y)
                        print 'During EXP, telescope has scanned by {} pixels in the y ({:.1f} seconds)'.format(scan_y, scan_time)
                        view(subexposure.SCI.data, vmin=0., show=False, title='{:.0f}, {:.0f}'.format(subexposure.xpix, subexposure.ypix))
                        ax = p.gca()
                        ax.set_autoscale_on(False)
                        p.plot(subexposure.xpix, subexposure.ypix, marker='o',color='w')
                        p.plot(x_order, y_order, marker='o',mec='w',mfc='None')
                        p.plot(x_di, y_di, marker='x',mec='r', mfc='None')
                        p.show()
                        view(direct_image.reads[0].SCI.data, vmin=0., vmax=100, show=False, alpha=0.5, title=(x_di, y_di))
                        ax = p.gca()
                        ax.set_autoscale_on(False)
                        p.plot(x_di, y_di, marker='o',mec='k', mfc='None')
                        p.show()
                    fix_x, fix_y = None, None

                else: # fit_image = True
                    rows, subexposure.ypix  = r.find_box(subexposure.SCI.data, t.psf_h, refine=False)
                    spectrum, subexposure.xpix = r.find_box(rows.T, 192-15, refine=False)
                    fix_x, fix_y = subexposure.xpix - (192-15)/2, subexposure.ypix
                    XOFF, YOFF = 0, 0
                    if t.check_sextractor:
                        view(subexposure.SCI.data, show=False)
                        ax = p.gca()
                        ax.set_autoscale_on(False)
                        p.plot(subexposure.xpix, subexposure.ypix, marker='o',color='w')
                        p.plot(fix_x, fix_y, marker='o',mec='w',mfc='None')
                        p.show()

                dispersion_solution, beam = cal.disp_poly(conf_file_G141, catalogue, deltatime, scan_rate, -scan_direction, n='A', x_len=x_len, y_len=y_len, XOFF=XOFF, YOFF=YOFF, data_dir=t.source_dir+t.system+'/', debug=False, log=t.logger, original_image=subexposure.SCI.data, disp_coef=t.disp_coef, object_ind=t.object_ind, x=fix_x, y=fix_y)

                subexposure.waves = dispersion_solution

                before = subexposure.SCI.data.copy()
                if t.flat_field:
                    before = subexposure.SCI.data.copy()
                    subexposure.SCI.data, ff, ff_error = cal.flat_field_correct(subexposure.waves, subexposure.SCI.data, flat_file_G141)

                    # Pixels before the start of the first order dont have a defined wavelength
                    # and therefore dont have a defined flat-field

                    # zero order (BEAM B) *******************
                    #BEAMB -207 -177
                    # First order (BEAM A) *******************
                    #BEAMA 15 196
                    # Diff: 192 + 15

                    # Set areas without wavelengths to 1.
                    x1 = x_order+10
                    x2 = x_order+197 # 192-15 is the length of the first order BEAM
                    ff[:,:x1+1] = 1.
                    ff[:,x2:] = 1.
                    subexposure.SCI.data = before/ff


                    logger.info('Flat-field correction performed')
                    # Wavelength dependent flat field correction
                    # some NaNs/inf creep in due to dead pixels in ff, change these to zeros
                    bad_pixels = np.logical_or(np.isinf(subexposure.SCI.data),np.isnan(subexposure.SCI.data))
                    subexposure.SCI.data[bad_pixels] = 0
                    # Turns out this isnt enough, there are some very small values in ff that can cause spikes in flux
                    # Hence just run a second round of CR removal on this image for bad pixels in flat-field

            else: # if not calcing dispersion just find loc of subexposure
                rows, subexposure.ypix  = r.find_box(subexposure.SCI.data, t.psf_h, refine=False)
                spectrum, subexposure.xpix = r.find_box(rows.T, t.psf_w, refine=False)
                spectrum = spectrum.T
            if t.cr_local:

                #CR_clean, CR_mask = r.clean_cosmic_rays(subexposure.SCI.data, np.sqrt(subexposure.SCI.data), tol=t.cr_tol, replace=t.cr_replace, debug=False)
                CR_clean, CR_mask = r.spatial_median_filter(subexposure.SCI.data.copy(), subexposure.DQ_mask, tol=t.cr_tol, replace=t.cr_replace, debug=False, thresh=t.cr_thresh, read_noise=t.read_noise, sx=t.cr_x, sy=t.cr_y)
                assert np.count_nonzero(np.logical_and(CR_mask, DQ_mask)) == 0, 'Some DQ bad pixels have been masked as CRs'
                n_crs = np.count_nonzero(CR_mask)
                if t.cr_plot and False:
                    copy1 = subexposure.SCI.data.copy()
                    copy1[subexposure.DQ_mask] = np.nan
                    if False:
                        p.subplot(1,2,1)
                        view(copy1, cbar=False, show=False, title='Image')
                        p.subplot(1,2,2)
                        view(CR_mask, cmap='binary_r', cbar=False, show=False, title='CR Mask: {}'.format(n_crs))
                        p.tight_layout()
                        p.show()
                    # and a zoom on spectrum
                    xpix, ypix = x_order + int((192-15)/2.), y_order
                    if ypix-t.psf_h < 0: y1=0; y2=t.psf_h
                    elif ypix+t.psf_h > subexposure.SCI.data.shape[0]: y1=-t.psf_h; y2=None
                    else: y1 = ypix-t.psf_h/2; y2=ypix+t.psf_h/2
                    if xpix-75 < 0: x1=0; x2=150
                    elif xpix+75 > subexposure.SCI.data.shape[1]: x1=-150; x2=None
                    else: x1 = xpix-75; x2=xpix+75
                    copy2 = subexposure.SCI.data[y1:y2,x1:x2].copy()
                    dq2 = subexposure.DQ_mask[y1:y2,x1:x2]
                    copy2[dq2] = np.nan
                    p.subplot(1,3,1)
                    view(copy2, cbar=False, show=False, title='Before')
                    '''
                    ax = p.gca()
                    ax.set_autoscale_on(False)
                    for y, row in enumerate(subexposure.DQ_mask):
                        for x, pix in enumerate(row):
                            if pix == 1: p.plot(x,y,marker='o',mec='w',mfc='None', ms=1)
                    '''
                    p.subplot(1,3,2)
                    view(CR_mask[y1:y2,x1:x2], cmap='binary_r', cbar=False, show=False, title='CR Mask: {}'.format(np.count_nonzero(CR_mask[y1:y2,x1:x2])))
                    p.subplot(1,3,3)
                    copy3 = CR_clean[y1:y2,x1:x2].copy()
                    copy3[dq2] = np.nan
                    #view(copy2-copy3, cbar=False, show=False, title='Difference ({:.0f} to {:.0f} e-)'.format(np.nanmin(copy2-copy3), np.nanmax(copy2-copy3)))
                    view(copy3, cbar=False, show=False, title='After')
                    p.tight_layout()
                    p.show()
                subexposure.SCI.data = CR_clean
                #n_crs = np.count_nonzero(CR_mask[ypix-t.psf_h/2:ypix+t.psf_h/2,xpix-100:xpix+100])
                subexposure.n_crs = n_crs
                subexposure.CR_mask = CR_mask
                if n_crs > 10:
                    #logger.warning('Subexposure {}, large number of CRs detected after flat-fielding: {}'.format(i+1, n_crs))
                    pass
                else:
                    logger.info('Removed {} CRs pixels from subexposure {}'.format(n_crs,i+1))
                if t.dq_replace != 'median':
                    subexposure.mask = np.logical_or(DQ_mask, CR_mask)
                else:
                    subexposure.mask = DQ_mask
                # Remove CR hits, with NaN or custom CR_replace value
                subexposure.SCI.header['CRs'] = (n_crs, 'Number of crs detected in box (local median)')

                if t.cr_plot:
                    view(subexposure.SCI.data, cbar=False, show=False, cmap='binary_r', alpha=1)
                    ax=p.gca()
                    ax.set_autoscale_on(False)
                    for j in range(subexposure.SCI.data.shape[0]):
                        for k in range(subexposure.SCI.data.shape[1]):
                            if CR_mask[j,k]: p.plot(k, j, marker='o', mec='r', mfc='None')
                    p.title('{} crs'.format(n_crs))
                    p.show()


            else:
                subexposure.mask = DQ_mask
                subexposure.n_crs = 0
                subexposure.CR_mask = np.zeros_like(subexposure.SCI.data).astype(bool)
                subexposure.SCI.header['CRs'] = (0, 'Number of crs detected in box (local median)')

            # This is a mask of all pixels to be ignored in the subexposure
            subexposures.append(subexposure)

    # Now break out of loop
    # and apply sliding median filter using all subexposures
    # need to supply scan rate to correct, here use POSTARG2 keyword for direction and default magnitude of 0.12"/s
    # this does not work well for drift scanned spectra as the offsets are imperfect. Could be improved with interpolation
    if t.cr_slide:
        logger.info('Removing CRs using sliding median filter')
        # scan rates, should calculate on fly for generality
        POSTARG2 = exposure.Primary.header['POSTARG2']
        scan_rate = data.get_scan_rate(exposure)

        subexposures = r.slide_median_filter(exposure, tol=t.cr_sigma, width=t.cr_width, thresh=t.cr_thresh, POSTARG2=POSTARG2, scan_rate=scan_rate, logger=logger)

    if t.bg and t.bg_plot:
            pdf_file = t.source_dir+t.system+'/logs/'+exposure.rootname+'_bg.pdf'
            f.silentremove(pdf_file)
            pdf = PdfPages(pdf_file)
            show = False
    else: show = True

    exposure.subexposures = []
    for i, subexposure in enumerate(subexposures):
        # Reduce the subexposure
        logger.info('Subexposure {}'.format(i+1))

        if t.bg:
            # Using masks for spectrum and background stars
            if t.scanned: psf_h = None
            else: psf_h = t.psf_h
            bg, bg_err = r.calc_subexposure_background(subexposure, method='median', masks=t.n_masks, debug=t.bg_plot, neg_masks=t.neg_masks, mask_h=t.mask_h, psf_w=t.psf_w, psf_h=psf_h, show=True)
            if t.bg_plot and t.bg_pdf:
                p.tight_layout()
                pdf.savefig()
                p.close()
            elif t.bg_plot: p.show()

            # Using a fixed area of the detector to calculate bg mean
            #bg, bg_err = r.area_bg(subexposure.SCI.data, row=(0,15), col=(130,148), psf_h=t.psf_h, debug=t.bg_plot, pix=None) # Laura's area
            #bg = r.area_bg(subexposure.SCI.data, row=(15,30), col=(130,148), psf_h=t.psf_h, debug=t.debug, pix=subexposure.ypix) # New area
            #bg = r.area_bg(subexposure.SCI.data, row=(subexposure.ypix-120,subexposure.ypix-105), col=(100,115), psf_h=t.psf_h, debug=t.debug, pix=subexposure.ypix) # Drift area, traces pixel variations :/

            # eclipse 4
            #bg = r.rectangle_bg(subexposure.SCI.data, row=[180,210], col=[200,250], debug=t.debug)
            # eclipse 1
            #bg = r.rectangle_bg(subexposure.SCI.data, row=[0,5], col=[50,250], debug=t.debug)

            bg = np.ones_like(subexposure.SCI.data)*bg
            logger.info('Background median found to be {} electrons per pixel'.format(np.nanmedian(bg)))
            if np.nanmedian(bg) > 50:
                logger.warning('Large background of {} electrons per pixel found in subexposure {}'.format(np.nanmedian(bg), i))
            elif np.nanmedian(bg) == 0.:
                logger.warning('Background of 0 electrons per pixel found in subexposure {}'.format(i+1))

        else:
            bg = np.zeros_like(subexposure.SCI.data)
            bg_err = 0.

        subexposure.SCI.data += -bg
        subexposure.bg = bg
        subexposure.bg_err = bg_err
        subexposure.SCI.header['BG'] = np.median(bg)
        subexposure.SCI.header['BG_ERR'] = bg_err

        if t.dispersion:

            pix = subexposure.ypix # find the position of the spectrum on the image
            subexposure.wave_scale = subexposure.waves[pix]

            # Interpolate to a fixed scale given by the direct image of the spectrum
            original_image = subexposure.SCI.data.copy()

            subexposure.SCI.data, subexposure.scale_mask = cal.interp_full_image2(subexposure.wave_scale, subexposure.waves, subexposure.SCI.data, subexposure.mask.astype(float))

            logger.info('Pixels interpolated to fixed wavelength scale at direct image')
            # Do for ERR as well, disable for speed
            #subexposure.ERR.data, subexposure.scale_mask = cal.interp_full_image2(subexposure.wave_scale, subexposure.waves, subexposure.ERR.data, subexposure.mask.astype(float))

            # Overly contaminated pixels are masked
            subexposure.mask = subexposure.scale_mask > t.contam_thresh

        else:
            subexposure.wave_scale = np.array(range(len(subexposure.SCI.data[0])))
        exposure.subexposures.append(subexposure)

        '''
        view(subexposure.SCI.data, show=False, title=i+1)
        ax=p.gca()
        ax.set_autoscale_on(False)
        p.plot([0,512],[256,256], color='k', ls='--')
        p.plot([256,256],[0,512], color='k', ls='--')
        p.show()
        '''
    if t.bg and t.bg_plot:
        if hasattr(pdf, 'close'):
            pdf.close()
        del pdf
    if t.save_dir:
         logger.info('Saving reduced file to {}'.format(t.save_dir))
         data.write_reduced_fits(exposure.subexposures, exposure.Primary, t, dest_dir=t.save_dir)
    logger.info('Time taken: {}'.format(time.time()-t0))
    print 'Reduction took {:.2f}s'.format(time.time()-t0)
    logger.info('########################################')
    logger.info('###########Finished Reduction###########')
    logger.info('########################################')

    if t.logger:
        # need to close all the file handlers explicitly if performing multiple reductions
        for handler in logger.handlers:
            if hasattr(handler,'close'):
                handler.close()

    if t.spatial_plot:
        for sub in exposure.subexposures:
            image = sub.SCI.data.copy()
            image[sub.mask] = np.nan
            view(image, show=False, label='Spectrum')
            x, y = sub.SCI.data.shape
            ax = p.gca()
            ax.set_autoscale_on(False)
            p.plot([x/2,x/2],[0,y], color='w', ls='-')
            p.plot([0,x],[y/2,y/2], color='w', ls='-', label='Amplifiers')

            p.plot([x/4,x/4], [y/4,3*y/4], color='w', ls='--')
            p.plot([x/4,3*x/4], [3*y/4,3*y/4], color='w', ls='--')
            p.plot([3*x/4,3*x/4], [3*y/4,y/4], color='w', ls='--')
            p.plot([x/4,3*x/4], [y/4,y/4], color='w', ls='--', label='Banding')
            p.show()

    return exposure


def extract_spectra(reduced_exposure, toggle=None, conf_file=None, **kwargs):
    '''
    Extract spectra from reduced exposure, either as file or python object.

    kwargs: custom toggles, assigned by keyword arguments
            possibilities are:
            toggle: toggle all bools True or False

            data_dir: source of the file
            calc_var: Calculate variances for the spectra or use ERR extension
            extraction_box: n pixel extraction box before optimal extraction
            mask_neg: can mask all the negative pixels in case of strong negative persistence
            skip_start: number of subexposures to ignore from start (reverse time index)
            skip_end: number of subexposures to ignore from end(if None ignores none)
            save_dir: destination to save the spectrum
            ignore_blobs: avoid reads contaminated heavily by IR blobs
            pdf: save optimal extraction fits in a pdf
            debug: print and plots some steps for debugging
            logger: log results to terminal and file
            shift_spectra: shift spectra of each subexposure to correct for drift
            save_extension: _spec.txt default, extension for spectrum file

            OPTIMAL EXTRACTION
            opt_ext: use optimal extraction
            debug: plot some functional fits to spatial scans
            box_h: height of initial extraction box used on spectrum, default 40
            s: sky average or background noise (array-like or scalar)
            v_0: rms of readout noise
            q: effective photon number per pixel value
            s_clip: sigma threshold for outliers in spatial fit
            s_cosmic: sigma threshold for cosmic ray removal
            func_type: type of function used for fit (poly, gauss or heavi)
            method: which optimization method, lsq or one from scipy.optimize.minimize (e.g. CG)
            fit_tol: tolerance for the fit, fraction of each flux point
            step: stepsize in lstsq fitting (epsfcn)
            order: order of the spline fit (if func_type=spline) default to 2
            remove_bg: remove the background before optimal extraction or leave it in
            skip_fit: don't fit profile to spectrum but just use fluxes as weights
    '''
    t0 = time.time()
    toggles = {'toggle':toggle, 'source_dir': '/home/jacob/hst_data/', 'system': 'WASP-18', 'calc_var': True, 'extraction_box': True, \
                'opt_ext': True, 'mask_neg': False, 's': 0, 'v_0': 20**2, 'q': 1, 's_clip': None, 's_cosmic': None, 'func_type': 'spline',\
                'method': 'lsq', 'debug': False, 'box_h': 80, 'skip_start':1, 'skip_end': 0, 'fit_tol':0.01, 'save_dir':None, \
                'ignore_blobs': False, 'blob_thresh':0.7, 'pdf':False, 'step': None, 'order': 2, 'skip_fit': False, 'remove_bg': True, \
                'logger':True, 'shift_spectra': True, 'k': 9, 'save_extension': '_spec.txt', 'view_d':False, 'refine_ypix':False, \
                'refine_xpix':False, 'object_ind':0}
    if conf_file:
        conf_kwargs = read_conf_file(conf_file)
        for kw in kwargs:
            conf_kwargs[kw] = kwargs[kw]
        full_kwargs = conf_kwargs
    else:
        full_kwargs = kwargs

    t = f.bunch_kwargs(kwargs=full_kwargs, toggle=toggle, toggles=toggles, verbose=True, name='extraction')

    # check if we need to open the reduced fits file
    if type(reduced_exposure) is str:
        if not reduced_exposure.endswith('_red.fits'):
            if not reduced_exposure.endswith('.fits'):
                reduced_exposure += '_red.fits'
            else:
                reduced_expsure = reduced_exposure[:-5] + '_red.fits'
        reduced_exposure = data.Data_red(t.source_dir+t.system+'/'+reduced_exposure)

    # Set up logging
    if t.logger:
        rootname = reduced_exposure.rootname
        if t.pdf:
            pdf_file = t.source_dir+t.system+'/logs/'+rootname+'_n_fit.pdf'
            f.silentremove(pdf_file)
        else: pdf_file = None
        log_file = t.source_dir+t.system+'/logs/'+rootname+'_ext.log'
        f.silentremove(log_file)
        warnings_file = t.source_dir+t.system+'/logs/ext_warnings.log'
        f.silentremove(warnings_file)
        logger = logging.getLogger(rootname+'_ext')
        logger.setLevel(logging.DEBUG)
        if len(logger.handlers) == 0:
            if t.debug:
                level = logging.DEBUG
            else:
                level = logging.INFO
            logger = add_handlers(logger, log_file, warnings_file, level)
    else:
        logger = f.EmptyLogger()
    logger.info('########################################')
    logger.info('###########Starting Extraction##########')
    logger.info('########################################')
    logger.info('Spectral extraction pipeline performed on exposure {}'.format(reduced_exposure.filename))
    logger.info('For configuration, see conf file: {}'.format(conf_file))
    if t.debug and t.opt_ext:
        logger.info('Saving optimal extraction fits to {} for each subexposure'.format(pdf_file))

    # check if the observation is compatible with this pipeline
    # need to have multiple controlled readouts.
    if not reduced_exposure.Primary.header['SAMP_SEQ'].startswith('SPARS'):
        logger.warning('{} is not a suitable observation routine for this pipeline.'.format(reduced_exposure.Primary.header['SAMP_SEQ']))

    spectra, variances, good_cols = [], [], []

    try:
        start, end = t.skip_start, len(reduced_exposure.subexposures) - t.skip_end - 1
        if end == len(reduced_exposure.subexposures) - 1: end = None
        subexposures = reduced_exposure.subexposures[start:end]
    except AttributeError:
        # hasnt been through my pipeline, use list of reads as subs
        subexposures = reduced_exposure.reads
        for sub in subexposures:
            sub.DQ_mask = sub.remove_bad_pix(replace=None, int_flags=[4,32,512])
            sub.CR_mask = np.zeros_like(sub.SCI.data)

    # first check the location of the spectrum in wavelength, calculated in dispersion solution
    # if not find spectrum crudely
    if not hasattr(subexposures[0], 'xpix'):
        logger.warning('Finding location of spectrum using maximal flux, not direct image.')
        xpixs = [ r.find_box(subexposure.SCI.data.T, h=200, refine=False)[1] for subexposure in subexposures ]
        xpix = int(np.median(xpixs))
        for sub in subexposures: sub.xpix = xpix

    Ds, DQs, CRs, Ps = [], [], [], [] # check images to see if any obvious CRs are not removed
    for n_sub, subexposure in enumerate(subexposures):

        logger.info('Extracting from subexposure {}'.format(n_sub+1))

        if t.mask_neg and n_sub == 0:
            logger.warning('Masking negative pixels')
            subexposure.mask = np.logical_or(subexposure.SCI.data<0,subexposure.mask)
            subexposure.SCI.data[subexposure.SCI.data < 0] = 0

        # remove the value of bad pixels, retain mask. Let this be done in optimal extraction
        #subexposure.SCI.data[subexposure.mask] = 0

        # In theory want to comput a variance before hand
        # Instead below use the ERR extension data to estimate variances

        try: bg = subexposure.bg; bg_err = subexposure.bg_err # background that has been removed
        except AttributeError: bg = 0.; bg_err = 0.; logger.warning('No background defined')

        if t.extraction_box:
            if n_sub == 0: logger.info('Extracting spectrum with a {} high extraction box'.format(t.box_h))
            if subexposure.SCI.data.shape[-1] > 266: xshift = True
            else: xshift = False # no 0th order so don't need to worry about boxing in the spatial direction
            xshift = True
            if not hasattr(subexposure, 'waves'): subexposure.waves = np.repeat(np.arange(subexposure.SCI.data.shape[1]), subexposure.SCI.data.shape[0]).reshape(subexposure.SCI.data.shape).T
            if xshift:
                xpix = subexposure.xpix
                D, DQ_mask, CR_mask, bg, waves = map(lambda image: r.box_cut(xpix, image, 200, horizontal=True), [subexposure.SCI.data, subexposure.DQ_mask, subexposure.CR_mask, bg, subexposure.waves])
                logger.info('Spectrum located at pixel {} in spatial direction'.format(xpix))
            else:
                D, DQ_mask, CR_nask, bg = subexposure.SCI.data, subexposure.DQ_mask, subexposure.CR_mask, bg

            ypix = subexposure.ypix

            if t.refine_ypix:
                # This was used to refine the ypixel location
                # problem was actually due to reference pixels and filter offset
                # with these properly corrected, change becomes 1 pixel or less
                # (probably due to trace) so do not refine ypix
                n_box_h = 3
                y_image = r.box_cut(ypix, subexposure.SCI.data, h=n_box_h*t.box_h)
                _, ypix2 = r.find_box(y_image, t.box_h)
                ypix_old = ypix
                ypix = ypix + ypix2 - n_box_h/2.*t.box_h

            D, DQ_mask, CR_mask, bg, waves = map(lambda image: r.box_cut(ypix, image, t.box_h), [D, DQ_mask, CR_mask, bg, waves])
            #view(D, vmin=0, vmax=500, title='Extraction box, xpix: {}'.format(xpix))
            if len(subexposure.wave_scale) > len(D.T) and xshift:
                subexposure.wave_scale = r.box_cut(xpix, subexposure.wave_scale, 200, horizontal=True)
                subexposure.waves = waves

            if not t.calc_var:
                if n_sub == 0: logger.info('Using ERR extension for variance estimate')
                err = r.box_cut(ypix, subexposure.ERR.data, t.box_h)
                if xshift: err = r.box_cut(xpix, err, 200, horizontal=True)

                V = np.square(err) # gain correction already done to _ima files
                t.v_0 = 0
                # Use errors to estimate variance per pixel
            else:
                if n_sub == 0: logger.info('Using flux for variance estimate')
                V = np.abs(D) + bg_err**2 + t.v_0 # gain correction already done to _ima files
            # also box the mask
            # D has now been cut to an h tall box
        else:
            if n_sub == 0: logger.warning('No extraction box used')
            D = subexposure.SCI.data
            DQ_mask = subexposure.DQ_mask
            CR_mask = subexposure.CR_mask
            V = np.abs(D)

        if False:
            D2 = D.copy()
            mask = np.logical_or(DQ_mask, CR_mask)
            D2[mask] = np.nan
            view(D2)


        # Extraction box
        Ds.append(D)
        DQs.append(DQ_mask)
        CRs.append(CR_mask)
        if t.view_d:
            D2 = D.copy()
            D2[DQ_mask] = np.nan
            D2[CR_mask] = np.nan
            view(D2, title='Image: {:6g} e-s'.format(np.nansum(D2)))
            #view(DQ_mask, title='DQ_mask')
            #view(CR_mask, title='CR_mask')

        if False:
            x, y = np.median(waves, axis=0), np.median(D, axis=0)
            p.plot(x,y)
            p.title('Median row')
            p.show()
            interp_row = []
            for wave, row in zip(waves, D):
                row = row / np.sum(row) * np.sum(y)
                shift = r.spec_pix_shift(x, y, wave, row, debug=False)
                shift_y = np.interp(x, x+shift, row)
                interp_row.append(shift_y)
            interp_D = np.vstack(interp_row)
            view(D, title='D')
            view(interp_D, title='Interp')
            view(D-interp_D, title='Diff')
            D = interp_D

        if t.opt_ext:
            # Change mask to optimal extraction format
            # 1s where good, 0s where bad
            M_DQ, M_CR = np.ones_like(DQ_mask), np.ones_like(CR_mask)
            M_DQ[DQ_mask.astype(bool)] = 0
            M_CR[CR_mask.astype(bool)] = 0
            if t.pdf:
                pdf_file = t.source_dir+t.system+'/logs/{}_{}_fit.pdf'.format(rootname, n_sub)
                f.silentremove(pdf_file)
            else: pdf_file = None
            if t.skip_fit: logger.warning('Not fitting profile, using flux instead')

            # toggle removing the background before optimal extraction
            # or can handle the background in the extraction
            if t.remove_bg:
                D, S = D, 0
            else:
                D, S = D + bg, bg

            #view(D)
            #view(M_DQ,title='M_DQ')
            #view(M_CR,title='M_CR')
            #view(M_DQ.astype(int)+M_CR.astype(int))]
            spec, specV, P, V = ea.extract_spectrum(D=D, S=S, V_0=t.v_0, Q=t.q, V=V, s_clip=t.s_clip, s_cosmic=t.s_cosmic, func_type=t.func_type, method=t.method, debug=t.debug, tol=t.fit_tol, M_DQ=M_DQ, M_CR=M_CR, pdf_file=pdf_file, step=t.step, order=t.order, skip_fit=t.skip_fit, bg=bg, k=t.k, logger=logger)
            M = np.logical_and(M_CR, M_DQ)
            # best to make only a couple of pdf files as they are large
            #if n_sub == 1: t.pdf = None
            if t.debug and np.any(np.isnan(P)): view(D); view(P, show=False); view(np.isnan(P), alpha=0.5, cmap='binary',cbar=False)
            if np.any(np.isnan(spec)) and t.debug: p.plot(spec); p.title('NaNs in spectrum'); p.show()
            assert not np.any(np.isnan(P)), 'NaNs in spatial profile for subexposure {}'.format(n_sub)
            if np.any(np.isnan(spec)):
                logger.warning('NaNs in spectrum for subexposure {}'.format(n_sub))
                logger.warning('Replaced with zeros')
                spec[np.isnan(spec)] = 0.

            '''
            # Run through and check quality of the fits
            total, nbad = M.shape[0], M.shape[0] - np.sum(M, axis=0)
            for j in range(M.shape[1]):
                if nbad[j] / float(total) > 0.1:
                    print nbad[j], total
                    p.title('Column {} spatial fit'.format(j))
                    for k, point in enumerate(D[:,j]/np.sum(D[:,j])):
                        if M[k,j] == 0: color='r'
                        else: color='g'
                        p.plot(k, point, marker='x',color=color,ls='None')
                    p.plot(P[:,j], color='b')
                    p.show()
            '''


            P2 = np.empty(D.shape)
            P2[:,:] = 1./D.shape[0]
            V2 = ea.initial_variance_estimate(D=D, V_0=t.v_0, Q=t.q) + bg_err**2 # background calc affects
            spec2, specV2 = np.nansum(D, axis=0), np.nansum(V2, axis=0)/len(V2)

            spec3 = np.nansum(D, axis=0) # column sum
            '''
            for i, row in enumerate(D):
                if i > 70 and i < 80: p.plot(row, label=i)
                else: pass #p.plot(row)
            p.legend(fontsize='small')
            p.show()

            p.subplot(1,2,1)
            p.plot(spec, label='Optimal weight')
            p.plot(spec2, label='Equal weight')
            #p.plot(spec3, label='Column sum')
            p.legend()
            p.subplot(1,2,2)
            p.plot((spec-spec2), label='Difference', color='r')
            p.legend()
            p.show()
            '''
        else:
            if n_sub == 0: logger.warning('Not using optimal extraction - results will be noisy')
            mask = np.logical_or(CR_mask, DQ_mask).astype(int)
            M = np.logical_not(mask)
            P = np.ones_like(D) / len(D)
            # equal weights
            V = ea.initial_variance_estimate(D=D, V_0=t.v_0, Q=t.q) + bg_err**2 # background calc affects variance, normally handled by opt_ext

            spec, specV = np.nansum(D, axis=0), np.nansum(V, axis=0)/len(V)
            #spec, specV = ea.optimized_spectrum(D, t.s, P, V, M)
        Ps.append(P)
        # Optimal Extraction
        # Sum spatial

        spectrum = f.Spectrum(subexposure.wave_scale,spec,x_unit='Wavelength (microns)', y_unit='electrons')
        spectra.append(spectrum)
        variances.append(specV)

    '''
    rcParams['figure.figsize'] = 10, 10
    p.figure()
    p.subplot(2,1,1)
    for i, spec in enumerate(spectra):
        spec.plot(show=False, label=i)
    p.legend(fontsize='x-small')
    p.subplot(2,1,2)
    for i, spec in enumerate(spectra):
        spec.plot(show=False, label=i)
    p.xlim([1.1,1.6])
    p.ylim([400000,600000])
    p.tight_layout()
    p.show()
    rcParams['figure.figsize'] = 10, 5
    '''
    # Can't use subexposures where there are too many bad pixels on spectrum
    # threshold is what percentage of the flux missing
    if t.ignore_blobs:
        Ms = [ np.logical_not(np.logical_or(M_CR, M_DQ)) for M_CR, M_DQ in zip(CRs, DQs) ]
        bad_subs = np.array([ np.any(np.sum(M*P, axis=0)[50:150] < t.blob_thresh) for P, M in zip(Ps, Ms) ])
        # only check the inner 100 pixels, outside is mostly bg and so can be masked
        good_subs = np.logical_not(bad_subs)
        logger.info('Ignoring subexposures {} due to bad pixels on spectrum'.format(np.arange(len(bad_subs))[bad_subs]))
        spectra = [ spec for spec, gd in zip(spectra, good_subs) if gd == 1 ]
        variances = [ var for var, gd in zip(variances, good_subs) if gd == 1 ]
    # then rescale at the end by the number of good subexposure spectra
    # in each wavelength bin

    # So now add all the scaled spectra together, interpolated to a common wavelength scale
    interp_spectra = []
    # templates
    if len(spectra) > 1:
        x, y = np.median([spec.x for spec in spectra], axis=0), np.median([spec.y for spec in spectra], axis=0)
    elif len(spectra) == 1:
        x, y = spectra[0].x, spectra[0].y
    else: assert False, 'No spectra after reduction.'

    for spec in spectra:
        shift = r.spec_pix_shift(x, y, spec.x, spec.y, debug=False)
        shift_y = np.interp(x, x+shift, spec.y)
        interp_spectra.append(shift_y)

    #interp_spectra = [np.interp(scale, spec.x, spec.y) for spec in spectra]
    if len(interp_spectra) > 1:
        y = np.nansum(interp_spectra, axis=0)
        variance = np.nansum(variances, axis=0)
    elif len(interp_spectra) == 1:
        y = interp_spectra[0]
        variance = variances[0]

    if t.ignore_blobs:
        y_before, var_before = y.copy(), variance.copy()
        n_subs = np.sum(good_subs)
        tot_subs = np.sum(good_subs) + np.sum(bad_subs)
        y = y * float(tot_subs) / n_subs
        variance = variance * (float(tot_subs) / n_subs)**2
        logger.info('{} subexposures used out of'.format(n_subs, len(subexposures)))
        unit = 'Electrons'
    else:
        # Don't rescale
        unit = 'Electrons'

    exp_spectrum = f.Spectrum(x, y, x_unit='Spectral Pixel', y_unit=unit)

    if True:
        if len(Ds) % 2 == 0: rows = len(Ds)/2
        else: rows = (len(Ds)+1)/2
        p.figure(figsize=(10,10))
        for i in range(len(Ds)):
            if len(Ds) > 1: p.subplot(rows,2,i+1)
            im = Ds[i].copy()
            im[DQs[i]] = np.nan
            title = 'Subexposure {}'.format(i+1)
            if t.ignore_blobs:
                if bad_subs[i]: title = 'Subexposure {}, Ignored'.format(i+1)
            view(im, title=title, show=False, cbar=False, ylabel='', xlabel='')
        p.tight_layout()
        p.savefig(t.source_dir+t.system+'/logs/'+rootname+'_box_'+str(t.object_ind)+'.png')
        p.close()


    if t.save_dir:
        # mark if the scan is forward or reverse, better not to do it here, instead use header of original file
        end = t.save_extension

        fname = t.save_dir + reduced_exposure.Primary.header['ROOTNAME'] + end
        f.silentremove(fname)
        logger.info('Saving spectrum to {}'.format(fname))
        text = '\n'.join([ '\t'.join([str(a),str(b),str(c)]) for a,b,c in zip(subexposure.wave_scale,exp_spectrum.y,np.sqrt(variance))])
        with open(fname, 'w') as txtf:
            # this assumes you did wavelength calibration
            txtf.write('wave\tflux\terror\n')
            txtf.write(text)
    logger.info('Time taken: {}'.format(time.time()-t0))
    print 'Extraction took {:.2f}s'.format(time.time()-t0)
    logger.info('########################################')
    logger.info('###########Finished Extraction##########')
    logger.info('########################################')

    # Logging
    if t.logger:
        # need to close all the file handlers explicitly if performing multiple extractions
        for handler in logger.handlers:
            if hasattr(handler,'close'):
                handler.close()

    return exp_spectrum, variance, interp_spectra, variances

def transit_depths_dw(transit_file=None, source_dir='/home/jacob/hst_data/', system='GJ-1214', save_file=None,debug=False, plot=False, u_fix=None, coefs0=None, temp_offset=0., timings='J', fluxes='J', wmin=None, wmax=None, bins=15, direction='r', discard=True, fit=['rp', 'C', 'V', 'R', 'tau', 'dt0' ], **kwargs):
    '''
    Calculate transit depths as a function of wavelength using divide white technique.
    Assume only scanned in one direciton, input file of rootnames/filters/times.
    transit_file: file where the fluxes and times of each exposure in the transit are stored
    save_file: where to save the results
    u_fix: can fix the limb darkening coefficient to broadband light curve if already known
    coefs0: estimates for the parameters in the model-ramp fit to the broadband light curve
    temp_offset: add a temporary time offset to the light curve (there appears to be an error)
    kwargs: any additional transit parameters to be passed to r.custom_params then to batman.TransitParams
    '''
    if wmin is None: wmin = wavelength_bins[0]
    if wmax is None: wmax = wavelength_bins[-1]
    if not bins is None: wavelength_bins = bins
    times, broad_flux, errors, sdirection = data.broadband_fluxes(files=transit_file, source_dir=source_dir, system=system, wmin=wmin,wmax=wmax, direction=direction)
    if times[2] < 2400000:
        times = np.array([ t + 2400000.5 for t in times]) # MJD to JD
    if timings=='L': # use Laura's timings
        with open('/scratch/jarcang1/Downloads/GJ1214b_raw_white_lc.txt','r') as g:
            lines = g.readlines()
        lines = [ line.strip().split(' ') for line in lines ]
        lines = [ line for line in lines if float(line[3]) == 0.] # first visit
        ltimes, lfluxes, lerrors = zip(*[(float(line[0]), float(line[1]), float(line[2])) for line in lines ])
        ltimes, lfluxes, lerrors = zip(*[(t, f, e) for t, f, e in sorted(zip(ltimes,lfluxes,lerrors))])
        times = np.array(sorted(ltimes))
        if fluxes == 'L':
            times, broad_flux, errors = zip(*sorted(zip(ltimes,lfluxes,lerrors)))
            times, broad_flux, errors = np.array(times), np.array(broad_flux), np.array(errors)



    # First fit visit long slope, ramp amplitudes and timescale to broadband light curve.
    params = r.custom_transit_params(system=system, **kwargs)
    #res1 = r.fit_systematics(broad_flux, times, errors, params, fit_type='model-ramp', u_fix=u_fix, debug=debug, coefs0=coefs0, temp_offset=temp_offset)
    res1 = systematics.fit_model_ramp(broad_flux, times, errors, params, debug=debug, temp_offset=temp_offset, discard=discard, fit=fit, **kwargs)
    V, R, tau, Z, i, a, P, t0, u = res1['V'], res1['R'], res1['tau'], res1['Z'], res1['i'], res1['a'], res1['P'], res1['t0'], res1['u']

    #return res1

    # Then, for each wavelength bin, fit normalization constant and Rp/R. Fix u using Laura's findings for GJ1214.
    depths, errs = [], []
    for i in range(len(wavelength_bins)-1):
        wmin, wmax = wavelength_bins[i], wavelength_bins[i+1]
        ignore_times, fluxes, errors, sdirection = data.broadband_fluxes(files=transit_file, system=system, source_dir=source_dir, wmin=wmin,wmax=wmax, direction=direction)
        fit2 = [fi for fi in fit if fi != 'u']
        res2 = systematics.fit_model_ramp(fluxes, times, errors, params, debug=debug, discard=discard, coefs0={'u':[u]}, **kwargs)
        #res2 = systematics.fit_divide_white(fluxes, times, errors, params, Z, V, R, tau, i, a, P, t0, debug=debug, temp_offset=temp_offset, discard=discard)
        depths.append(res2['depth'])
        errs.append(res2['rms'])

    waves = wavelength_bins[:-1]+np.diff(wavelength_bins)/2.
    if save_file:
        with open(save_file,'w') as g:
            g.write('Wavelength\tTransit depth\n')
            for w, d in zip(waves, depths):
                g.write(str(w)+'\t'+str(d)+'\n')
    if plot:
        #p.plot(waves, depths, ls='None', marker='o')
        p.errorbar(waves, depths, np.array(errs)/10**6, ls='None', marker='o')
        p.title('Transit depth as a function of wavelength')
        p.xlabel('Wavelength (microns)')
        p.ylabel('Transit depth')
        p.show()

        relative_depths = (depths - np.mean(depths)) * 10**6
        p.plot(waves, relative_depths, ls='None', marker='o')
        p.title('Relative depths, errors ~ {:.0f} ppm'.format(np.mean(errs)))
        p.xlabel('Wavelength (microns)')
        p.ylabel('Relative transit depth (ppm)')
        p.savefig('relative_transit_depths.png')
        p.show()
    return waves, depths, errs

def create_orbit_cats(target='GJ-1214', data_dir='/home/jacob/hst_data/'):
    '''
    Creates orbit catalogue files by first finding deep visit catalogue,
    then projecting this onto each individual orbit direct image.
    Requires creation of an input_image.lis and visit_driz.lis
    in the source data directory.
    input_image.lis has the format [rootname]t[FILTER]t[TIME OBS]t[SCAN DIRECTION]
    and associates orbits to their visit direct image.
    visit_driz.lis is just a list of visit direct image files that have
    been drizzled together.
    '''
    # Remember to set up environment for iraf before e.g. using ur_setup in bash OR conda, source activate iraf27
    import subprocess
    from pyraf import iraf
    from iraf import stsdas, analysis, slitless, axe
    data.clean_axe_dir(data_dir+'/aXe/')

    source_dir = data_dir + target + '/'

    command = 'pwd'
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    process.wait()
    previous_dir = process.communicate()[0][:-1]

    os.chdir(data_dir+'aXe/DATA/')
    try:
        if not os.path.exists(source_dir+'input_image.lis'):
            data.make_input_image_list(system=target, data_dir=data_dir)
        if not os.path.exists(source_dir+'visit_driz.lis'):
            data.make_driz_list(system=target, data_dir=data_dir)
        print target
        if target == 'GJ-1214':
            targ_filters = ['F130N']
            piv_ls = [1301] # pivot wavelength (nm)
            mag_zeros = [22.99] # from http://www.stsci.edu/hst/wfc3/phot_zp_lbn
        elif target == 'WASP-18' or target == 'WASP-43':
            targ_filters = ['F126N', 'F139M'] # or F139M, 1384 nm, 24.4793
            piv_ls = [1258, 1384] # pivot wavelength (1258.49 nm)
            mag_zeros = [22.8609, 24.4793]
        elif target in ['WASP-19','Kepler-9'] or target.startswith('2MASS-J19021775+3824032'):
            # 2MASS-J19021775+3824032 is Kepler-9
            targ_filters = ['F139M']
            piv_ls = [1383.8] # pivot wavelength (nm)
            mag_zeros = [24.4793]
        elif target == '55-Cancri':
            targ_filters = ['F132N']
            piv_ls = [1318.8]
            mag_zeros = [22.95]
        # assume
        targ_filter, piv_l, mag_zero = targ_filters[0], piv_ls[0], mag_zeros[0]

        with open(source_dir+'visit_driz.lis', 'r') as driz_list:
            for line in driz_list:
                fname = line[:-1]
                print fname
                visname = fname[:6] # e.g. ibxy01
                shutil.copyfile(source_dir+fname, data_dir+'/aXe/DATA/'+fname)

                # Copy SCI and WHT extensions to new files
                # pyraf: imcopy F130N.drz.fits[SCI] F130N_drz_sci.fits & WHT
                with pyfits.open(data_dir+'/aXe/DATA/'+fname, memmap=False) as HDUList:
                    for ext in HDUList:
                        try:
                            if ext.header['EXTNAME'] == 'SCI':
                                iraf.module.imcopy(fname+'[SCI]', fname.split('_')[0]+'_drz_sci.fits')
                            elif ext.header['EXTNAME'] == 'WHT':
                                iraf.module.imcopy(fname+'[WHT]', fname.split('_')[0]+'_drz_wht.fits')
                        except KeyError: pass
                # Run SExtractor to produce deep visit catalogue

                command = 'sex -c default.sex -WEIGHT_IMAGE '+fname.split('_')[0]+'_drz_wht.fits'+' -MAG_ZEROPOINT '+str(mag_zero)+ ' -CATALOG_NAME '+fname.split('_')[0]+'.cat ' + fname.split('_')[0]+'_drz_sci.fits'
                print command
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                process.wait() # !!!!

                # check we found a star
                with open(fname.split('_')[0]+'.cat', 'r') as check:
                    lines = [line for line in check]
                lines = [line for line in lines if not line.startswith('#')]
                if lines == []:
                    print 'No stars found by SExtractor for exposure '+ fname.split('_')[0]
                    continue

                # Copy over orbit _flt files
                dest_files = []
                for row in open(source_dir+'input_image.lis','r'):
                    if row.startswith('#'): continue

                    for k in range(len(targ_filters)):
                        targ_filter, piv_l, mag_zero = targ_filters[k], piv_ls[k], mag_zeros[k]
                        flt_fname, FILTER, EXPSTART, scan = row.split('\t')
                        if flt_fname.startswith(visname) and FILTER == targ_filter:
                            flt_fname = flt_fname+'_flt.fits'
                            shutil.copyfile(source_dir+flt_fname, data_dir+'/aXe/DATA/'+flt_fname)

                            dest_file = flt_fname.split('_')[0]+'_flt_1.cat'
                            f.silentremove(source_dir+dest_file)
                            dest_files.append(dest_file)
                            true_k = k

                targ_filter, piv_l, mag_zero = targ_filters[true_k], piv_ls[true_k], mag_zeros[true_k]

                # rename MAG_AUTO -> MAG_F1300 in _prep.cat file
                with open(fname.split('_')[0]+'.cat','r') as visit_cat:
                    lines = visit_cat.readlines()

                for i, row in enumerate(lines):
                    if row.startswith('#  12 MAG_AUTO'):
                        row = '#  12 MAG_F'+ str(piv_l)+row[14:]
                        lines[i] = row
                input_cat  = fname.split('_')[0]+'_prep.cat'

                with open(input_cat, 'w') as visit_prep_cat:
                    for row in lines:
                        visit_prep_cat.write(row)


                #Load in axe, pyraf: stsdas; analysis; slitless; axe
                #pyraf: iolprep F130_drz.fits F130N_prep.cat dimension_in='+100,0,0,0'
                print 'Mdrizzle image', fname
                print 'Input cat', input_cat

                axe.iolprep(mdrizzle_image=fname, input_cat=input_cat, dimension_info='0,0,0,0')

                # Copy the results over to the source dir
                for cat_fname in dest_files:
                    shutil.copyfile(cat_fname, source_dir + cat_fname)

        data.clean_axe_dir(data_dir+'/aXe/')
    except:
        raise
    finally:
        os.chdir(previous_dir)

def create_orbit_cats_gauss(target='GJ-1214', source_dir='/home/jacob/hst_data/'):
    '''
    Fit a gaussian to direct image to create visit catalogue.
    Use same format as SExtractor for catalogue files.
    '''
    from lmfit import minimize, Parameters
    import astropy
    from astropy import modeling

    data_dir = source_dir + target + '/'
    if not os.path.exists(data_dir+'input_image.lis'):
        data.make_input_image_list(system=target, data_dir=source_dir)
    if not os.path.exists(data_dir+'visit_driz.lis'):
        data.make_driz_list(system=target, data_dir=source_dir)

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
