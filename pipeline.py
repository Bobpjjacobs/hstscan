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

def arrays_plot(arrays, name='Read', tight_layout=True, size=3, show=True, **kwargs):
        nrows = np.ceil(len(arrays)/4.)
        rcParams['figure.figsize'] = size*4, size*nrows
        for i, array in enumerate(arrays):
            p.subplot(nrows, 4, i+1)
            view(array, show=False, **kwargs)
            p.title('{} {}'.format(name, i))
            p.axis('off')
        if tight_layout: p.tight_layout()
        if show: p.show()
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
                'debug': False, 'pdf': False, 'bg_box':200, 'bg_plot': False, 'psf_h':130, 'logger': True, 'force_no_offset': False, 'neg_masks': 1, \
                'shift_spectra': True, 'cr_plot': False, 'cr_x': 1, 'cr_y': 1, 'mask_h':40, 'dq_flags': [4,32,512], 'psf_w':220, \
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
        try:
            exposure = data.Data_ima(exposure, bjd=False)
        except IOError:
            exposure = data.Data_ima(t.source_dir+exposure+'_ima.fits', bjd=False)

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

    # Set up pdf file for plots
    if t.pdf and t.debug:
        pdf_file = t.save_dir+'logs/'+exposure.rootname+'_red.pdf'
        f.silentremove(pdf_file)
        pdf = PdfPages(pdf_file)
        logger.info('Pdf debug file saved in: {}'.format(pdf_file))

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

    def save_fig():
        if t.pdf:
            pdf.savefig()
            p.close()
        else:
            p.show()

    if t.debug:
        view(masks[0], title='DQ Flagged pixels', cmap='binary_r', cbar=False, show=False)
        save_fig()
    if t.debug:
        vmin = 0.; vmax = np.max(reads[-3].SCI.data)
        arrays_plot([read.SCI.data for read in reads], cbar=False, vmin=vmin, vmax=vmax, \
                        tight_layout=False, size=2, show=False)
        save_fig()


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
        direct_image = data.Data_ima(t.source_dir+di_name+'_ima.fits', bjd=False)
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
        if t.flat_field:
            view(ff, title='Flat-field', cbar=True, cmap='binary_r', vmin=0.9, vmax=1.1, show=False)
            save_fig()

        arrays_plot([sub.SCI.data for sub in subexposures], name='Subexp', cbar=False, size=2, \
                        tight_layout=False, vmin=0., vmax=np.max(subexposures[-3].SCI.data), show=False)
        save_fig()

    # Local CR removal
    if t.cr_local:
        new_subs = []
        for subexposure in subexposures:
            #CR_clean, CR_mask = r.clean_cosmic_rays(subexposure.SCI.data, np.sqrt(subexposure.SCI.data), tol=t.cr_tol, replace=t.cr_replace, debug=False)
            # Mask everything outside of 1st order
            if t.dispersion:
                x1 = int(subexposure.x_order)+10
                x2 = int(subexposure.x_order)+197 # 192-15 is the length of the first order BEAM
                mask_0 = np.zeros_like(subexposure.SCI.data)
                mask_0[:,:x1+1] = 1.
                mask_0[:,x2:] = 1.
                ignore_mask = np.logical_or(subexposure.DQ_mask, mask_0)
            else:
                ignore_mask = subexposure.DQ_mask

            CR_clean, CR_mask = r.spatial_median_filter(subexposure.SCI.data.copy(), ignore_mask, tol=t.cr_tol, replace=t.cr_replace, debug=False, \
                                    thresh=t.cr_thresh, read_noise=t.read_noise, sx=t.cr_x, sy=t.cr_y)


            assert np.count_nonzero(np.logical_and(CR_mask, subexposure.DQ_mask)) == 0, 'Some DQ bad pixels have been masked as CRs'
            n_crs = np.count_nonzero(CR_mask)

            subexposure.SCI.data = CR_clean
            #n_crs = np.count_nonzero(CR_mask[ypix-t.psf_h/2:ypix+t.psf_h/2,xpix-100:xpix+100])
            subexposure.n_crs = n_crs
            subexposure.CR_mask = CR_mask
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
            new_subs.append(subexposure)
        if t.debug:
            # Plot number of CRs over subexposures
            # and plot cr distribution in x and y (?)
            n_crs = [subexposure.n_crs for subexposure in subexposures]
            p.bar(np.arange(len(n_crs)), n_crs, width=1., edgecolor='k')
            p.title('Number of CRs / subexposure')
            p.ylabel('Cosmic Rays')
            p.xlabel('Subexposure number')
            save_fig()

            CR_masks = [ subexposure.CR_mask for subexposure in subexposures ]
            x_count = np.sum([ np.sum(mask, axis=0) for mask in CR_masks ], axis=0)
            p.subplot(2,1,1)
            p.title('Distribution of CRs over pixels')
            p.bar(np.arange(CR_masks[0].shape[0]), x_count, width=1.)
            p.xlabel('x pixel')
            p.subplot(2,1,2)
            y_count = np.sum([ np.sum(mask, axis=1) for mask in CR_masks ], axis=0)
            p.bar(np.arange(CR_masks[0].shape[0]), y_count, width=1.)
            p.xlabel('y pixel')
            p.tight_layout()
            save_fig()

            all_CRs = np.sum(CR_masks, axis=0)
            view(all_CRs, units='No. of CRs', title='Distribution of CRs', show=False)
            save_fig()

        subexposures = new_subs

    else:
        for i in range(len(subexposures)):
            subexposure = subexposures[i]
            subexposure.mask = DQ_mask
            subexposure.n_crs = 0
            subexposure.CR_mask = np.zeros_like(subexposure.SCI.data).astype(bool)
            subexposure.SCI.header['CRs'] = (0, 'Number of crs detected in box (local median)')
            new_subs.append(subexposure)
            subexposures[i] = subexposure

    # Background removal
    if t.bg:
        for i in range(len(subexposures)):
            subexposure = subexposures[i]
            if t.bg:
                # Using masks for spectrum and background stars
                if t.scanned: psf_h = None
                else: psf_h = t.psf_h
                bg, bg_err = r.calc_subexposure_background(subexposure, method='median', masks=t.n_masks, \
                                debug=t.bg_plot, neg_masks=t.neg_masks, mask_h=t.mask_h, psf_w=t.psf_w, psf_h=psf_h, show=not t.pdf)
                if t.bg_plot:
                    p.subplot(1,2,1)
                    p.title('Subexposure {}'.format(i))
                    save_fig()
                    t.bg_plot = False

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
            subexposures[i] = subexposure

        if t.debug:
            bgs = [ subexposure.SCI.header['BG'] for subexposure in subexposures ]
            ts = [ subexposure.SCI.header['SAMPTIME'] for subexposure in subexposures ]
            p.subplot(2,1,1)
            p.title('Background estimates')
            p.plot(bgs, marker='o', ls='None', color='g', ms=10, mec='k')
            p.xlabel('Subexposure number')
            p.ylabel('Background (electrons)')
            p.subplot(2,1,2)
            p.plot(ts, bgs, marker='o', ls='None', color='c', mec='k', ms=10)
            p.xlabel('Time (seconds)')
            p.tight_layout()
            save_fig()

    exposure.subexposures = subexposures

    if t.pdf:
        pdf.close()

    if t.save_dir:
        # Save reduced fits file (_red.fits format)
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

    return exposure


def extract_spectra(reduced_exposure, conf_file=None, **kwargs):
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
    toggles = {'source_dir': '/home/jacob/hst_data/', 'system': 'WASP-18', 'calc_var': True, 'extraction_box': True, \
                'opt_ext': True, 'mask_neg': False, 's': 0, 'v_0': 20**2, 'q': 1, 's_clip': None, 's_cosmic': None, 'func_type': 'spline',\
                'method': 'lsq', 'debug': False, 'box_h': 80, 'skip_start':1, 'skip_end': 0, 'fit_tol':0.01, 'save_dir':None, \
                'ignore_blobs': False, 'blob_thresh':0.7, 'pdf':False, 'step': None, 'order': 2, 'skip_fit': False, 'remove_bg': True, \
                'logger':True, 'shift_spectra': True, 'k': 9, 'save_extension': '_spec.txt', 'view_d':False, 'refine_ypix':False, \
                'refine_xpix':False, 'object_ind':0}
    if conf_file:
        conf_kwargs = data.read_conf_file(conf_file)
        for kw in kwargs:
            conf_kwargs[kw] = kwargs[kw]
        full_kwargs = conf_kwargs
    else:
        full_kwargs = kwargs

    t = f.bunch_kwargs(kwargs=full_kwargs, toggles=toggles, verbose=True, name='extraction')

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
        log_file = t.save_dir+'logs/'+rootname+'_ext.log'
        f.silentremove(log_file)
        warnings_file = t.save_dir+'logs/ext_warnings.log'
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

    # Set up pdf file for plots
    if t.pdf and t.debug:
        pdf_file = t.save_dir+'logs/'+reduced_exposure.rootname+'_ext.pdf'
        f.silentremove(pdf_file)
        pdf = PdfPages(pdf_file)
        logger.info('Pdf debug file saved in: {}'.format(pdf_file))
    def save_fig():
        if t.pdf:
            pdf.savefig()
            p.close()
        else:
            p.show()

    # check if the observation is compatible with this pipeline
    # need to have multiple controlled readouts.
    if not reduced_exposure.Primary.header['SAMP_SEQ'].startswith('SPARS'):
        logger.warning('{} is not a suitable observation routine for this pipeline.'.format(reduced_exposure.Primary.header['SAMP_SEQ']))

    subexposures = reduced_exposure.subexposures

    Ds, DQs, CRs, Ps = [], [], [], [] # check images to see if any obvious CRs are not removed
    spectra, variances = [], []
    for n_sub, subexposure in enumerate(subexposures):
        logger.info('Extracting from subexposure {}'.format(n_sub+1))

        # Compute Error estimates
        # Need bg_err to compute errors
        try: bg = subexposure.bg; bg_err = subexposure.bg_err # background that has been removed
        except AttributeError: bg = 0.; bg_err = 0.; logger.warning('No background defined')


        # Cut out extraction box
        if t.extraction_box:
            if n_sub == 0: logger.info('Extracting spectrum with a {} high extraction box'.format(t.box_h))
            if subexposure.SCI.data.shape[-1] > 266: xshift = True
            else: xshift = False # no 0th order so don't need to worry about boxing in the spatial direction
            xshift = True
            if not hasattr(subexposure, 'waves'): subexposure.waves = np.repeat(np.arange(subexposure.SCI.data.shape[1]), subexposure.SCI.data.shape[0]).reshape(subexposure.SCI.data.shape).T
            if xshift:
                xpix = subexposure.xpix
                D, DQ_mask, CR_mask, bg, waves, err = map(lambda image: r.box_cut(xpix, image, 200, horizontal=True), \
                                [subexposure.SCI.data, subexposure.DQ_mask, subexposure.CR_mask, bg, subexposure.waves, subexposure.ERR.data])
                logger.info('Spectrum located at pixel {} in spatial direction'.format(xpix))
            else:
                D, DQ_mask, CR_nask, bg, err = subexposure.SCI.data, subexposure.DQ_mask, subexposure.CR_mask, bg, subexposure.ERR.data

            ypix = subexposure.ypix
            D, DQ_mask, CR_mask, bg, waves, err = map(lambda image: r.box_cut(ypix, image, t.box_h), [D, DQ_mask, CR_mask, bg, waves, err])

        else:
            if n_sub == 0: logger.warning('No extraction box used')
            D = subexposure.SCI.data
            DQ_mask = subexposure.DQ_mask
            CR_mask = subexposure.CR_mask
            err = subexposure.ERR.data

        if not t.calc_var:
            if n_sub == 0: logger.info('Using ERR extension for variance estimate')
            V = np.square(err) # gain correction already done to _ima files
            t.v_0 = 0
            # Use errors to estimate variance per pixel
        else:
            if n_sub == 0: logger.info('Using flux for variance estimate')
            V = np.abs(D) + bg_err**2 + t.v_0 # gain correction already done to _ima files

        Ds.append(D)
        DQs.append(DQ_mask)
        CRs.append(CR_mask)

        if t.opt_ext:
            # Change mask to optimal extraction format
            # 1s where good, 0s where bad
            M_DQ, M_CR = np.logical_not(DQ_mask).astype(int), np.logical_not(CR_mask).astype(int)
            if t.skip_fit: logger.warning('Not fitting profile, using flux instead')

            # toggle removing the background before optimal extraction
            # or can handle the background in the extraction
            if t.remove_bg:
                D, S = D, 0
            else:
                D, S = D + bg, bg

            spec, specV, P, V = ea.extract_spectrum(D=D, S=S, V_0=t.v_0, Q=t.q, V=V, s_clip=t.s_clip, s_cosmic=t.s_cosmic, \
                                    func_type=t.func_type, method=t.method, debug=False, tol=t.fit_tol, M_DQ=M_DQ, M_CR=M_CR, \
                                    pdf_file=None, step=t.step, order=t.order, skip_fit=t.skip_fit, bg=bg, k=t.k, logger=logger)
            M = np.logical_and(M_CR, M_DQ)

            if t.debug and np.any(np.isnan(P)): view(D); view(P, show=False); view(np.isnan(P), alpha=0.5, cmap='binary',cbar=False)
            if np.any(np.isnan(spec)) and t.debug: p.plot(spec); p.title('NaNs in spectrum'); save_fig()
            assert not np.any(np.isnan(P)), 'NaNs in spatial profile for subexposure {}'.format(n_sub)
            if np.any(np.isnan(spec)):
                logger.warning('NaNs in spectrum for subexposure {}'.format(n_sub))
                logger.warning('Replaced with zeros')
                spec[np.isnan(spec)] = 0.

            P2 = np.ones_like(D)/D.shape[0]
            V2 = ea.initial_variance_estimate(D=D, V_0=t.v_0, Q=t.q) + bg_err**2 # background calc affects
            spec2, specV2 = np.nansum(D, axis=0), np.nansum(V2, axis=0)/len(V2)

            spec3 = np.nansum(D, axis=0) # column sum

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
        spectrum = f.Spectrum(waves[waves.shape[0]/2,:],spec,x_unit='Wavelength (microns)', y_unit='electrons')
        spectra.append(spectrum)
        variances.append(specV)

    if t.debug:
        arrays = [ ]
        for D, DQ_mask, CR_mask in zip(Ds, DQs, CRs):
            array = D.copy()
            array[DQ_mask] = np.nan
            array[CR_mask] = np.nan
            arrays.append(array)
        vmin = 0.; vmax = np.nanmax(arrays[-3])
        arrays_plot(arrays, vmin=vmin, vmax=vmax, cbar=False, tight_layout=False,\
                        size=3, name='Box', show=False)
        save_fig()

    if t.debug:
        p.figure()
        p.subplot(2,1,1)
        p.title('Subexposure spectra')
        for i, spec in enumerate(spectra):
            spec.plot(show=False, label=i)
        p.legend(fontsize='x-small')
        p.subplot(2,1,2)
        for i, spec in enumerate(spectra):
            spec.plot(show=False, label=i)
        p.xlim([1.1,1.6])
        p.ylim([400000,600000])
        p.tight_layout()
        save_fig()


    # Can't use subexposures where there are too many bad pixels on spectrum
    # threshold is what percentage of the flux missing
    if t.ignore_blobs:
        Ms = [ np.logical_not(np.logical_or(M_CR, M_DQ)) for M_CR, M_DQ in zip(CRs, DQs) ]
        bad_subs = np.array([ np.any(np.sum(M*P, axis=0)[50:150] < t.blob_thresh) for P, M in zip(Ps, Ms) ])
        # only check the inner 100 pixels, outside is mostly bg and so can be masked
        good_subs = np.logical_not(bad_subs)
        logger.warning('Ignoring subexposures {} due to bad pixels on spectrum'.format(np.arange(len(bad_subs))[bad_subs]))
        spectra = [ spec for spec, gd in zip(spectra, good_subs) if gd == 1 ]
        variances = [ var for var, gd in zip(variances, good_subs) if gd == 1 ]
    # then rescale at the end by the number of good subexposure spectra
    # in each wavelength bin

    # So now add all the scaled spectra together, interpolated to a common wavelength scale
    interp_spectra = []
    # templates wavelength from median
    if len(spectra) > 1:
        x, y = np.median([spec.x for spec in spectra], axis=0), np.median([spec.y for spec in spectra], axis=0)
    elif len(spectra) == 1:
        x, y = spectra[0].x, spectra[0].y
    else: assert False, 'No spectra after reduction.'

    # actual interpolation
    for spec in spectra:
        shift = r.spec_pix_shift(x, y, spec.x, spec.y, debug=False)
        shift_y = np.interp(x, x+shift, spec.y)
        interp_spectra.append(shift_y)

    # Compute total spectrum of combined subexposures
    if len(interp_spectra) > 1:
        y = np.nansum(interp_spectra, axis=0)
        variance = np.nansum(variances, axis=0)
    elif len(interp_spectra) == 1:
        y = interp_spectra[0]
        variance = variances[0]

    # Rescale if ignore one or more subexposures
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

    if t.debug:
        p.plot(exp_spectrum.x, exp_spectrum.y, label='Combined spectrum')
        p.title('Combined spectrum')
        p.xlabel('Wavelength (micron)')
        p.ylabel('Electrons')
        save_fig()


    if t.save_dir:
        # mark if the scan is forward or reverse, better not to do it here, instead use header of original file
        end = t.save_extension

        fname = t.save_dir + reduced_exposure.Primary.header['ROOTNAME'] + end
        f.silentremove(fname)
        logger.info('Saving spectrum to {}'.format(fname))
        text = '\n'.join([ '\t'.join([str(a),str(b),str(c)]) for a,b,c in zip(exp_spectrum.x,exp_spectrum.y,np.sqrt(variance))])
        with open(fname, 'w') as txtf:
            # this assumes you did wavelength calibration
            txtf.write('wave\tflux\terror\n')
            txtf.write(text)
    logger.info('Time taken: {}'.format(time.time()-t0))
    print 'Extraction took {:.2f}s'.format(time.time()-t0)
    logger.info('########################################')
    logger.info('###########Finished Extraction##########')
    logger.info('########################################')

    if t.pdf:
        pdf.close()

    # Logging
    if t.logger:
        # need to close all the file handlers explicitly if performing multiple extractions
        for handler in logger.handlers:
            if hasattr(handler,'close'):
                handler.close()

    return exp_spectrum, variance, interp_spectra, variances
