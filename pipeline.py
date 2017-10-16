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

def add_handlers(logger, log_file, warnings_file, level):
        # Add handlers to a logging file
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

def arrays_plot(arrays, name='Read', tight_layout=True, size=3, **kwargs):
        nrows = np.ceil(len(arrays)/4.)
        rcParams['figure.figsize'] = size*4, size*nrows
        for i, array in enumerate(arrays):
            p.subplot(nrows, 4, i+1)
            view(array, show=False, **kwargs)
            p.title('{} {}'.format(name, i))
            p.axis('off')
        if tight_layout: p.tight_layout()
        p.show()
        rcParams['figure.figsize'] = 10, 5

def reduce_exposure(exposure, conf_file=None, **kwargs):
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

    t0 = time.time() # time reduction run

    # store all the possible external configs in an object 't'
    toggles = {'system': 'WASP-18', 'source_dir': '/home/jacob/hst_data/WASP-18/', 'read_noise': 20, 'flat_field': True, 'bg': True, \
                'n_masks': 3, 'dispersion': True, 'units': True, 'log': True, 'cr_local': True, 'cr_slide': False, 'cr_tol': 30, 'cr_sigma': 3, \
                'cr_width': 5, 'cr_thresh': 500,  'cr_replace': 'median', 'dq_replace': None, 'save_dir': None, 'contam_thresh': 0.01, \
                'debug': False, 'bg_box':200, 'bg_plot': False, 'psf_h':130, 'logger': True, 'force_no_offset': False, 'neg_masks': 1, \
                'shift_spectra': True, 'cr_plot': False, 'cr_x': 1, 'cr_y': 1, 'bg_pdf':False, 'mask_h':40, 'dq_flags': [4,32,512], 'psf_w':220, \
                'spatial_plot':False, 'disp_coef':'default', 'scanned':True, 'scan_rate':None, 'object_ind': 0, 'check_sextractor':False, \
                'fit_image':False, 'cat':None}
    # read in conf_file to update default values
    if conf_file:
        conf_kwargs = data.read_conf_file(conf_file)
        for kw in kwargs:
            conf_kwargs[kw] = kwargs[kw]
        full_kwargs = conf_kwargs
    else:
        full_kwargs = kwargs
    t = f.bunch_kwargs(kwargs=full_kwargs, toggle=None, toggles=toggles, name='reduction', verbose=True)
    conf_file_G141 = './src/WFC3.G141/WFC3.IR.G141.V2.5.conf'
    flat_file_G141 = './src/WFC3.G141/WFC3.IR.G141.flat.2.fits'

    # open up an exposure if you input a filename
    if type(exposure) == str:
        #if exposure.endswith('_ima.fits'):
        #    exposure = data.Data_ima(t.source_dir+exposure, bjd=False)
        # don't need timings in bjd for reduction
        #else:
        exposure = data.load(t.source_dir+exposure)

    # Set up logging
    if t.logger:
        log_file = t.save_dir+'logs/'+exposure.rootname+'_red.log'
        f.silentremove(log_file)
        warnings_file = t.save_dir+'logs/red_warnings.log'
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

    # Start reduction process
    masks, reads = [], []
    for read in exposure.reads:

        # Remove reference pixels from edge of image
        if not read.trimmed: read.trim_pix(replace=t.dq_replace)

        # Mask bad pixel flags
        mask = read.remove_bad_pix(replace=t.dq_replace, int_flags=t.dq_flags)

        # convert to e-s from e-s/s
        if t.units and read.SCI.header['BUNIT'] == 'ELECTRONS/S':
            try: int_time = read.TIME.header['PIXVALUE']
            except KeyError: int_time = np.median(read.TIME.data)
            read.SCI.data = read.SCI.data * int_time
            read.ERR.data = read.ERR.data * int_time
            read.SCI.header['BUNIT'] = 'ELECTRONS'
            read.ERR.header['BUNIT'] = 'ELECTRONS'

        masks.append(mask)
        reads.append(read)
    if t.debug: view(masks[0], title='DQ Flagged pixels', cmap='binary_r', cbar=False)
    if t.debug:
        vmin = 0.; vmax = np.max(reads[-3].SCI.data)
        arrays_plot([read.SCI.data for read in reads], cbar=False, vmin=vmin, vmax=vmax, \
                        tight_layout=False, size=2)


    # Work on subexposures
    subexposures = []
    if exposure.filename.endswith('_flt.fits'): _n = 0 # only 1 read!
    else: _n = 2 # skip 1 for zeroth, 1 for fast read, 1 for difference

    for i in range(len(exposure.reads)-_n):
        if exposure.filename.endswith('_flt.fits'):
            # treat flt file as a single subexposure
            subexposure = exposure.reads[i]
            DQ_mask = subexposure.remove_bad_pix(replace=t.dq_replace, int_flags=t.dq_flags)
            subexposure.DQ_mask = DQ_mask
        else:
            read2, read1 = exposure.reads[i], exposure.reads[i+1]
            DQ_mask = np.logical_or(masks[i], masks[i+1])
            # Form subexposure
            subexposure = r.create_sub_exposure(read1, read2, read_noise=t.read_noise)
            subexposure.DQ_mask = DQ_mask
        subexposures.append(subexposure)

    # Calculate dispersion solution
    if t.dispersion:
        new_subs = []
        POSTARG1, POSTARG2, PA_V3 = exposure.Primary.header['POSTARG1'], exposure.Primary.header['POSTARG2'], exposure.Primary.header['PA_V3']
        # Find scan direction from positional offset
        if POSTARG2 >= 0.: scan_direction = +1; logger.info('Forward scan')
        else: scan_direction = -1; logger.info('Reverse scan')

        if t.scanned:
            if not t.scan_rate: scan_rate = data.get_scan_rate(exposure)
        else: t.scan_rate = 0.

        # Find direct image position
        catalogue, di_name = data.find_catalogue(exposure.rootname, data_dir=t.source_dir)
        direct_image = data.load(t.source_dir+di_name)
        di_size = direct_image.reads[0].SCI.shape[0] - 10 # ref pix
        t.cat = catalogue
        di_ps1, di_ps2, di_pav3 = direct_image.Primary.header['POSTARG1'], direct_image.Primary.header['POSTARG2'], direct_image.Primary.header['PA_V3']
        with open(t.cat,'r') as cat:
            lines = cat.readlines()
            objects = [line[:-1].split() for line in lines if line[0] != '#']
            objects = [[float(val) for val in obj] for obj in objects ]
            objects = sorted(objects, key=lambda obj: obj[-1])
            obj = objects[t.object_ind]

            image_fname = t.source_dir+t.cat.split('_')[-3].split('/')[-1]+'_flt.fits'
            SEx, SEy = obj[1], obj[2]
            #SEx, SEy = cal.center_of_flux(t.source_dir+di_name, SEx, SEy, size=10)
            # Location of the direct image
            x_di, y_di = SEx, SEy # w.r.t to reference pixel of direct image exposure

        for subexposure in subexposures:
            # Calculate offset
            if t.scanned:
                exp_time, deltatime = subexposure.SCI.header['SAMPTIME'], subexposure.SCI.header['DELTATIM']
                scan_time = exp_time
            else:
                exp_time, deltatime, scan_time = subexposure.SCI.header['SAMPTIME'], subexposure.SCI.header['DELTATIM'], 0.

            y_len, x_len = subexposure.SCI.data.shape
            # n = 'A' specifies first order beam

            scan_x = 0.
            offset_x = (POSTARG1- di_ps1) / 0.135
            # moving the telescope right moves the target right on the image as it is facing the other way
            scan_y = (scan_direction * (t.scan_rate * scan_time)) / 0.121
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

            #if t.check_sextractor: print 'PA_V3 offset of {:.2f} arcseconds, ({:.1f},{:.1f}) in pix'.format(dV3, pa_offset_x, pa_offset_y)
            #logger.debug('exposure time of {} with a scan rate of {}\"/s'.format(scan_time,t.scan_rate))


            # Different filters have different inherent direct image offsets, refer to:
            #http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2010-12.pdf
            # this is because the conf file was made using the F140W filter
            # For WFC3/IR F130N: dx = 0.033 +- 0.014, dy = 0.004 +- 0.019
            # For WFC3/IR F126N: dx = 0.264 +- 0.018, dy = 0.287 +- 0.025
            # For WFC3/IR F139M: dx = 0.11 +- 0.022, dy = 0.029 +- 0.028
            filt = direct_image.Primary.header['FILTER']
            if filt == 'F126N': XOFF, YOFF = 0.264, 0.287
            elif filt == 'F130N': XOFF, YOFF = 0.033, 0.004
            elif filt == 'F129M': XOFF, YOFF = 0.11, 0.029
            else:
                logger.warning('Filter {} offset not known.'.format(filt))
                XOFF, YOFF = 0., 0.
            XOFF += scan_x + offset_x
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
            subexposure.ypix = y_order + (scan_direction * (t.scan_rate * deltatime)) / 0.121 /2.

            dispersion_solution, beam = cal.disp_poly(conf_file_G141, catalogue, deltatime, t.scan_rate, -scan_direction, n='A', x_len=x_len, y_len=y_len, \
                                            XOFF=XOFF, YOFF=YOFF, data_dir=t.source_dir, debug=False, log=t.logger, original_image=subexposure.SCI.data, \
                                            disp_coef=t.disp_coef, object_ind=t.object_ind, x=None, y=None)
            subexposure.waves = dispersion_solution

            if t.debug and False:
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

            # Flat field correction (requres wavelength solution)
            before = subexposure.SCI.data.copy()
            if t.flat_field:
                subexposure.SCI.data, ff, ff_error = cal.flat_field_correct(subexposure.waves, subexposure.SCI.data, flat_file_G141)
                # Pixels before the start of the first order dont have a defined wavelength
                # and therefore dont have a defined flat-field
                # zero order (BEAM B) *******************
                #BEAMB -207 -177
                # First order (BEAM A) *******************
                #BEAMA 15 196
                # Diff: 192 + 15

                # Set areas without wavelengths to 1.
                x1 = int(x_order)+10
                x2 = int(x_order)+197 # 192-15 is the length of the first order BEAM
                ff[:,:x1+1] = 1.
                ff[:,x2:] = 1.
                subexposure.SCI.data = before/ff
                logger.info('Flat-field correction performed')
                # Wavelength dependent flat field correction, some NaNs/inf creep in due to dead pixels in ff, change these to zeros
                bad_pixels = np.logical_or(np.isinf(subexposure.SCI.data),np.isnan(subexposure.SCI.data))
                subexposure.SCI.data[bad_pixels] = 0
                # Turns out this isnt enough, there are some very small values in ff that can cause spikes in flux
                # Hence just run a second round of CR removal on this image for bad pixels in flat-field

            new_subs.append(subexposure)
        subexposures = new_subs

    # Plot subexposures
    if t.debug:
        view(ff, title='Flat-field', cbar=True, cmap='binary_r', vmin=0.9, vmax=1.1)
        arrays_plot([sub.SCI.data for sub in subexposures], name='Subexp', cbar=False, size=2, \
                        tight_layout=False, vmin=0., vmax=np.max(subexposures[-3].SCI.data))
        #arrays_plot([sub.waves for sub in subexposures], name='Wavelengths')

    # Local CR removal
    if t.cr_local:
        new_subs = []
        for subexposure in subexposures:
            #CR_clean, CR_mask = r.clean_cosmic_rays(subexposure.SCI.data, np.sqrt(subexposure.SCI.data), tol=t.cr_tol, replace=t.cr_replace, debug=False)
            CR_clean, CR_mask = r.spatial_median_filter(subexposure.SCI.data.copy(), subexposure.DQ_mask, tol=t.cr_tol, replace=t.cr_replace, debug=False, thresh=t.cr_thresh, read_noise=t.read_noise, sx=t.cr_x, sy=t.cr_y)

            view(CR_mask, cmap='binary_r', cbar=False,show=False)
            p.axis('off')
            p.show()

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

    return exposure
