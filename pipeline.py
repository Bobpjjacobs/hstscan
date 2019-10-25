from __future__ import print_function
import reduction as r
import my_fns as f
import data, systematics
import extraction_algorithm as ea
import calibration as cal
from my_fns import np, p, os
import shutil, logging, time, multiprocessing, sys
from multiprocessing.pool import Pool
from matplotlib.pyplot import rcParams
from scipy.optimize import leastsq, curve_fit
from matplotlib.backends.backend_pdf import PdfPages

import dispersion as disp
import pyfits
view = data.view_frame_image

def add_handlers(logger, log_file, warnings_file, level):
        '''
        Set up logging file to include handlers for info and warnings.
        '''
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

class Empty_logger():
    '''Default logging to print to terminal'''
    def info(self,x): print(x)
    def warning(self,x): print(x)
    def debug(self,x): print(x)

def arrays_plot(arrays, name='Read', tight_layout=True, size=3, height=1, show=True, **kwargs):
        '''Neatly plot a list of arrays, using data.view_frame_image'''
        view = data.view_frame_image
        nrows = np.ceil(len(arrays)/4.)
        if len(arrays) > 1: rcParams['figure.figsize'] = size*4, size*nrows*height
        for i, array in enumerate(arrays):
            if len(arrays) > 1: p.subplot(nrows, 4, i+1)
            view(array, show=False, **kwargs)
            if not name is None: p.title('{} {}'.format(name, i))
            p.axis('off')
        if tight_layout: p.tight_layout()
        if show: p.show()
        rcParams['figure.figsize'] = 10, 5

def compute_exposure_shifts(visits, source_dir, save_dir=None, verbose=True, fname='exposure_shifts.lis', save=False):
    '''
    Compute shift of raw spectrum between exposures in the same visit

    Shifts are defined as the difference between the position of the star on the exposure
    and the position of the star on the reference exposure.
    So the -ve of the output of r.spec_pix_shift.

    Save outputs to file for use during pipeline run
    '''
    all_rootnames, all_shifts = [], []
    for visit in visits:
        print('Visit: {}'.format(visit))
        with open(source_dir+visit,'r') as g: lines = g.readlines()
        rootnames = [ line.split('\t')[0] for line in lines 
                      if (not line.strip()=='') and not line.startswith('#') 
                      and not line.split('\t')[1].startswith('F') ]

        # Reference exposure
        exp1 = data.load('{}{}_ima.fits'.format(source_dir, rootnames[0]))
        fl1 = np.sum(exp1.reads[0].SCI.data, axis=0) / np.sum(exp1.reads[0].SCI.data)
        x = np.arange(len(fl1))

        # Compute shifts for other exposures
        all_rootnames.append(rootnames[0])
        all_shifts.append(0)
        for rn in rootnames[1:]:
            exp2 = data.load('{}{}_ima.fits'.format(source_dir,rn))
            fl2 = np.sum(exp2.reads[0].SCI.data, axis=0) / np.sum(exp2.reads[0].SCI.data)
            shift, err = r.spec_pix_shift(x, fl1, x, fl2)

            all_rootnames.append(rn)
            all_shifts.append(-shift)

    if save_dir is None: save_dir=source_dir
    with open(save_dir+fname, 'w') as g:
        print('Saving to ', save_dir+fname)
        for rn, sh in zip(all_rootnames, all_shifts):
            line = '{}\t{:.3f}\n'.format(rn, sh)
            g.write(line)
    return np.array(all_shifts)

def reduce_exposure(exposure, conf_file=None, **kwargs):
    '''
    Input a data.Data_ima instance of an exposure file, or the filename.
    Returns a reduced exposure Data_red.
    Reductions involves:
        - forming subexposures with unit correction
        - remove cosmic rays with local (slow), sliding median filter, reference image
        - calculating dispersion solution
        - interpolate image to common wavelength scale (slow)
        - apply wavelength-dependant flat-field correction
        - calculate and remove background
        - save to new fits file (_red.fits)

    kwargs: custom toggles, assigned by keyword arguments
            priority is given to **kwargs > conf_file > default
            possibilities are:

            debug (bool):       print or log debugging information and plots
            logger (bool):      set up custom logging
            pdf (bool):         save debug plots to pdf (slow)

            system (str):       name of the exoplanet system
            source_dir (str):   directory where files are stored
            save_dir (str):     directory to store the reduced files

            units (bool): unit conversion from electrons/s to electrons
            object_ind (int): object index (brightest first) to extract from image, from catalogue
            scanned (bool): whether the exposure is spatially scanned or not
            scan_rate (float): scan rate of exposure in "/s if known
            nlincorr (bool): perform non-linearity correction or not
            read_noise (float): read noise of detector in e- per read
            
            DQ_flags (list): data quality flags to remove from image
            DQ_replace: what to replace pixels with (local median, mean, NaNs)
            DQ_mean_width (int): size of local mean/median area

            BG (bool): whether to perform background removal
            BG_plot (bool): make debug plots
            BG_area (bool): use specific area of image to calculate background, if not use masks
            BG_x, BG_y (ints): position of bottom left corner of area
            BG_w, BG_h (ints): size of background area (width and height)
            otherwise using masks: 
            n_masks (int): number of spectral masks
            neg_masks (int): number of negative spectral masks (for persistence)
            mask_h (int): height of masks in pixels
            psf_w (int): width of spectra trace
            psf_h (int): height of spectral trace

            CR_local (bool): perform local CR removal (only method left)
            CR_plot (bool): debug plots
            CR_tol (float): tolerance for local CR hits in sigma
            CR_thresh (float): threshold, in electrons, for CR hit detection
            CR_x, CR_y (ints): number of pixels over which to check deviation
            CR_replace: change the value that replaces a CR pixel (local mean, median, NaN...)

            dispersion (bool): perform wavelength calculation and corrections
            XOFF_file (str): file containing x-shifts between exposures (pre-calculated)
            exp_shift (bool): use exposure shifts during wavelength calibration
            flat_field (bool): whether to perform flat_field correction, wavelength dep if dispersion==True
            ff_min (float): minimum value for flat-field, values below this are set to 1
            nysig (int): number of spectrum gaussian widths to calculate wavelength for on image
            grid_y, grid_lam (ints): wavelength dependent photon trajectory grid resolution  
            interp_kind (str): type of interpolation if using basic method (linear, quadratic...)
            flat_file_g141 (str): config file for flat-field
            conf_file_g141 (str): config file for dispersion 
            contam_thresh: contamination threshold of bad pixels for interpolation
    '''

    t0 = time.time() # time reduction run

    # Store all the possible external configs in a dict 't', default values defined in 'toggles'
    toggles = {'system': 'WASP-18', 'source_dir': '/home/jacob/hst_data/WASP-18/', 'save_dir': None,
                'debug': False, 'pdf': False, 'logger': True,
                'scanned':True, 'scan_rate':None, 'units': True, 'nlincorr':False, 'read_noise': 20, 'remove_scan': False,
                'dq_replace': None, 'dq_mean_width':1, 'dq_flags': [4, 32], 'skip_start': 1, 'skip_end': 0,
                'bg': True, 'bg_plot': False, 'bg_area':True, 'bg_x':0, 'bg_y':0, 'bg_h':50, 'bg_w':50,
                'psf_h':130, 'mask_h':40,  'psf_w':220, 'n_masks': 3, 'neg_masks': 0, 
                'postarg_yguess':True, 'default_scan': 'f',
                'cr_local': True, 'cr_tol': 15, 'cr_replace': 'median', 'cr_plot': False, 
                'cr_x': 5, 'cr_y': 5, 'cr_thresh': 50.,
                'cr_master': False, 'cr_mname': None, 
                'dispersion': True, 'exp_shift':True, 'ref_exp': None, 'ref_wv0':0.9, 'ref_wv1':1.92, 
                'pre_shift_ff':False, 'peak': False, 'xshift_ext':0.,
                'flat_field': True, 'ff_min':0.5,
                'nysig':5, 'grid_y':20, 'grid_lam':20, 'two_scans':False, 'interp_kind':'linear',
                'flat_file_g141':'None', 'conf_file_g141':'None', 
                'contam_thresh': 0.01, 
                'object_ind': 0}
    # Read in conf_file and kwargs to update default toggle values
    # priority is kwargs > conf_file > default
    if conf_file:
        conf_kwargs = data.read_conf_file(conf_file)
        conf_kwargs.update(**kwargs)
    else: conf_kwargs = kwargs
    t = f.bunch_kwargs(kwargs=conf_kwargs, toggles=toggles, name='reduction', verbose=True)

    # Open up an exposure if you input a filename
    if type(exposure) == str:
        exposure = data.load(t.source_dir+exposure, bjd=False)

    # Set up logging for errors and info
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
        logger = Empty_logger()
    logger.info('########################################')
    logger.info('###########Starting Reduction###########')
    logger.info('########################################')
    logger.info('Data reduction pipeline performed on exposure {}'.format(exposure.filename))
    logger.info('For configuration, see pipeline conf file: {}'.format(conf_file))

    # Set up pdf file for debug plots
    if t.pdf and t.debug:
        pdf_file = t.save_dir+'logs/'+exposure.rootname+'_red.pdf'
        f.silentremove(pdf_file)
        pdf = PdfPages(pdf_file)
        logger.info('Pdf debug file saved in: {}'.format(pdf_file))
    def save_fig():
        # save figure to pdf or display
        # depending on t.pdf toggle
        if t.pdf:
            pdf.savefig()
            p.close()
        else:
            p.show()

    # Start reduction process
    masks, reads = [], []
    for read in exposure.reads:

        # Remove reference pixels from edge of image
        if not read.trimmed: read.trim_pix()

        # Mask bad pixel flags
        mask = read.remove_bad_pix(replace=t.dq_replace, int_flags=t.dq_flags, width=t.dq_mean_width)

        # convert to e- from e-/s
        if t.units and read.SCI.header['BUNIT'] == 'ELECTRONS/S':
            try: int_time = read.TIME.header['PIXVALUE']
            except KeyError: int_time = np.median(read.TIME.data)
            read.SCI.data = read.SCI.data * int_time
            read.ERR.data = read.ERR.data * int_time
            read.SCI.header['BUNIT'] = 'ELECTRONS'
            read.ERR.header['BUNIT'] = 'ELECTRONS'

        masks.append(mask)
        reads.append(read)

    if t.debug:
        # Plot flagged data quality pixels
        if t.dq_flags is not None:
            n_rows = np.ceil(len(t.dq_flags)/2.)
            if len(t.dq_flags) % 2 == 0: n_rows += 1 # for the combined DQ mask  

            p.figure(figsize=(6,3*n_rows))
            p.subplot(n_rows,2,1)
            view(masks[0], title='All DQ pixels ({})'.format(t.dq_flags), cmap='binary_r', cbar=False, show=False, xlabel='', ylabel='')
            for k, flag in enumerate(t.dq_flags):
                _mask = reads[0].DQ.data/flag % 2 == 1
                p.subplot(n_rows,2,k+2)
                title = '{}: {}\n{}'.format(flag, data.dq_info[flag][0], data.dq_info[flag][1])
                view(_mask, title=title, cmap='binary_r', cbar=False, show=False,xlabel='', ylabel='')
            p.tight_layout()        
            save_fig()
    if t.debug:
        # Plot all reads
        vmin = 0.; 
        #vmax = np.max(reads[0].SCI.data)
        vmax = None
        arrays_plot([read.SCI.data for read in reads], cbar=False, vmin=vmin, vmax=vmax, \
                        tight_layout=False, size=2, show=False)
        save_fig()


    # Build subexposures (difference reads)
    # also calculate errors
    subexposures = []
    if exposure.filename.endswith('_flt.fits'):
        # treat flt file as a single subexposure
        subexposure = exposure.reads[0]
        DQ_mask = subexposure.remove_bad_pix(replace=t.dq_replace, int_flags=t.dq_flags, width=t.dq_mean_width)
        subexposure.DQ_mask = DQ_mask
        subexposure.mask = DQ_mask
        subexposures.append(subexposure)
    else:
        for i in range(len(exposure.reads))[t.skip_end:-t.skip_start-1]:
            read2, read1 = exposure.reads[i], exposure.reads[i+1]
            DQ_mask = np.logical_or(masks[i], masks[i+1])
            # Form subexposure
            subexposure = r.create_sub_exposure(read1, read2, read_noise=t.read_noise, nlincorr=t.nlincorr)
            subexposure.DQ_mask = DQ_mask
            subexposure.mask = DQ_mask # track total mask (DQ + CR hits)
            subexposures.append(subexposure)

    # Background removal
    for i in range(len(subexposures)):
        subexposure = subexposures[i]
        if t.bg:
            if t.bg_area:
                # Using a fixed area of the detector to calculate bg mean
                t.bg_x, t.bg_y, t.bg_h, t.bg_w = map(int, [t.bg_x, t.bg_y, t.bg_h, t.bg_w])
                bg_mask = np.zeros_like(subexposure.SCI.data)
                bg_mask[t.bg_y:t.bg_y+t.bg_h, t.bg_x:t.bg_x+t.bg_w] = 1
                bg_image = subexposure.SCI.data[bg_mask.astype(bool)]
                bg_image = bg_image[np.isfinite(bg_image)]
                bg, bg_err = np.nanmedian(bg_image), np.nanstd(bg_image)
            else:
                # Using masks for spectrum and background stars
                if t.scanned: psf_h = None
                else: psf_h = t.psf_h
                bg, bg_err = r.calc_subexposure_background(subexposure, method='median', masks=t.n_masks, \
                                debug=t.bg_plot, neg_masks=t.neg_masks, mask_h=t.mask_h, psf_w=t.psf_w, psf_h=psf_h, show=not t.pdf)
                if t.bg_plot and False:
                    p.subplot(1,2,1) 
                    p.title('Subexposure {}'.format(i))
                    save_fig()
                t.bg_plot = False

            bg = np.ones_like(subexposure.SCI.data)*bg
            #logger.info('Background median found to be {} electrons per pixel'.format(np.nanmedian(bg)))
            if np.nanmedian(bg) > 50:
                logger.warning('Large background of {} electrons per pixel found in subexposure {}'.format(np.nanmedian(bg), i))
            elif np.nanmedian(bg) == 0.:
                logger.warning('Background of 0 electrons per pixel found in subexposure {}'.format(i+1))
        else:
            # No background removal
            bg = np.zeros_like(subexposure.SCI.data)
            bg_err = 0.
        subexposure.SCI.data -= bg
        subexposure.bg = bg
        subexposure.bg_err = bg_err
        subexposure.SCI.header['BG'] = np.median(bg)
        subexposure.SCI.header['BG_ERR'] = bg_err
        subexposures[i] = subexposure

    if t.debug:
        # Show background area
        if t.bg and t.bg_area:
            view(exposure.reads[0].SCI.data, show=False, vmin=0, vmax=50*len(subexposures), title='Background Area')
            view(bg_mask, alpha=0.5, show=False, cbar=False, cmap='binary_r')
            save_fig()
        if t.bg:
            # Plot of backgrounds over time
            bgs = [ np.median(subexposure.bg) for subexposure in subexposures ]
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
            if t.bg_area:
                # Plot of areas used for bg in each subexposure
                h = t.bg_h / float(t.bg_w)
                arrays_plot([sub.SCI.data[t.bg_y:t.bg_y+t.bg_h, t.bg_x:t.bg_x+t.bg_w] for sub in subexposures], 
                                name=None, cbar=False, size=4, height=h, tight_layout=False, vmin=0., vmax=100, show=False)
                p.suptitle('Bg Area for each subexposure')
                save_fig()

    # Calculate dispersion solution
    if t.dispersion:
        # First get relevant config values
        DISP_COEFFS, TRACE_COEFFS = disp.get_conf_coeffs()
        POSTARG1, POSTARG2, PA_V3 = exposure.Primary.header['POSTARG1'], exposure.Primary.header['POSTARG2'], exposure.Primary.header['PA_V3']
        # Find scan direction from positional offset
        if POSTARG2 >= 0.: scan_direction = +1; logger.info('Forward scan')
        else: scan_direction = -1; logger.info('Reverse scan')
        if t.scanned:
            if not t.scan_rate: 
                t.scan_rate = exposure.Primary.header['SCAN_RAT']
                if t.scan_rate == 0.:
                    logger.warning('SCANNED=True while exposure scan rate is zero')
            else:
                if 'SCAN_RAT' in exposure.Primary.header:
                    assert abs(t.scan_rate-exposure.Primary.header['SCAN_RAT'])/t.scan_rate < 0.01, \
                        'Scan rates do not match (input {}, fits {})'.format(t.scan_rate, exposure.Primary.header['SCAN_RAT'])
        else: t.scan_rate = 0.

        # Find direct image position from catalogue file (pre-computed)
        catalogue, di_name = data.find_catalogue(exposure.rootname, data_dir=t.source_dir)
        logger.info('Catalogue used: {}\nDirect Image name: {}'.format(catalogue, di_name))
        if not os.path.isfile(catalogue):
            logger.warning('No catalogue file found for {}'.format(catalogue))
            # then catalogue may be for visit drizzled file
            catalogue_split = catalogue.split('_')
            catalogue_split[-3] = catalogue_split[-3][:-3]+'011'
            catalogue = '_'.join(catalogue_split)
            if not os.path.isfile(catalogue):
                catalogue_split[-3] = catalogue_split[-3][:-3]+'010'
                catalogue = '_'.join(catalogue_split)
        try:
            direct_image = data.Data_ima(t.source_dir+di_name+'_ima.fits', bjd=False)
            logger.warning('Catalogue file: {}'.format(di_name+'_ima.fits'))
        except IOError: # no _ima for _drz files of multiple direct images
            direct_image = data.Data_ima(t.source_dir+di_name+'_drz.fits', bjd=False)
            logger.warning('No catalogue file found for {}'.format(di_name+'_drz.fits'))
        di_size = direct_image.reads[0].SCI.shape[0] 
        di_image = direct_image.reads[0].SCI.data
        if np.log2(di_size) % 1 != 0: 
            # Remove reference pixles from direct image
            di_size -= 10 # ref pix
            di_image = di_image[5:-5,5:-5]
        t.cat = catalogue
        di_ps1, di_ps2, di_pav3 = direct_image.Primary.header['POSTARG1'], direct_image.Primary.header['POSTARG2'], direct_image.Primary.header['PA_V3']
        with open(t.cat,'r') as cat:
            lines = cat.readlines()
            objects = [line[:-1].split() for line in lines if line[0] != '#']
            objects = [[float(val) for val in obj] for obj in objects ]
            objects = sorted(objects, key=lambda obj: obj[0])
            obj = objects[t.object_ind]

            image_fname = t.source_dir+t.cat.split('_')[-3].split('/')[-1]+'_flt.fits'
            SEx, SEy = obj[1], obj[2]
            #SEx, SEy = cal.center_of_flux(t.source_dir+di_name, SEx, SEy, size=10)
            # Location of the direct image
            x_di, y_di = SEx, SEy # w.r.t to reference pixel of direct image exposure
            logger.debug('Direct image location of ({},{})'.format(x_di, y_di))

        # Apply external shift (if any) to direct image (pre-computed)
        if t.xshift_ext != 0.:
                x_di += t.xshift_ext
                logger.debug('External xshift of direct image applied, of {} pix'.format(t.xshift_ext))
                exposure.xshift_ext = t.xshift_ext
        exposure.Primary.header['SHIFTEXT'] = t.xshift_ext

        # Now do a partial dispersion solution to allow us to calculate accurate exposure shifts
        # Skip last step if shifts corrected before or not at all.
        for step in ['partial', 'final']:
            if step == 'partial':
                xshift = 0. # temporarily
            if step == 'final':
                # Find reference exposure for shift calculation
                if t.ref_exp is None:
                    ref_exp = data.find_reference_exp(exposure.rootname, data_dir=t.source_dir)
                else:
                    ref_exp = t.ref_exp
                if ref_exp == exposure.rootname:
                    if t.exp_shift: logger.info('Reference exposure used for exposure shift calculation')
                    xshift = 0.
                else:
                    if t.exp_shift:
                        # Load in reference exposure
                        ref_exp = data.load(t.save_dir+ref_exp+'_red.fits')
                        ref_mask = np.sum([sub.mask for sub in ref_exp.subexposures], axis=0).astype(bool)
                        ref_image = np.nansum([sub.SCI.data for sub in ref_exp.subexposures], axis=0)

                        if t.pre_shift_ff: #not working
                            # Apply flat-field correction before shift calculation
                            L = subexposure.SCI.data.shape[0]
                            dL = (1014-L)/2
                            tot_image, ff, ff_error = cal.flat_field_correct(0, tot_image,
                                                        dL, L+dL, dL, L+dL,
                                                        t.flat_file_g141, ff_min=t.ff_min)
                        else:
                            tot_image = ref_image

                        # Compute total spectrum for this exposure
                        tot = np.nansum(tot_image, axis=0)[int(xpix):int(xpix)+200]
                        tot = tot / np.nansum(tot)
                        x = np.arange(len(tot))
                        ref_tot = np.nansum(image, axis=0)[int(xpix):int(xpix)+200]
                        ref_tot = ref_tot / np.nansum(ref_tot)
                              
                        # Select peak of the grism response (toggle)
                        if t.peak: x0, x1 = np.argmin(abs(wave_ref-1.14)), np.argmin(abs(wave_ref-1.6))
                        else: x0, x1 = 0, None
                        x, ref_tot, tot = map(lambda arr: arr[x0:x1], [x, ref_tot, tot])

                        # Calculate shift
                        xshift, err = r.spec_pix_shift(x, ref_tot, x, tot)
                        logger.info('exposure shift of {} relative to reference exposure {}'.format(xshift, ref_exp.rootname))
                    else:
                        xshift = 0.

                    if t.debug and t.exp_shift:
                        view(tot_image, show=False, title='Tot', vmin=0, vmax=100)
                        save_fig()
                        new_tot = np.interp(x, x-xshift, tot) # 1D linear

                        p.subplot(2,1,1)
                        p.title('xshift = {:.4f}'.format(xshift))
                        p.plot(x, ref_tot, label='Ref')
                        p.plot(x, tot, label='Exp')
                        p.plot(x, new_tot, label='Shifted')
                        p.legend()
                        p.subplot(2,1,2)
                        # zoom in on the only clear spectral line
                        xstart, xend = 70, 85
                        p.plot(x[xstart:xend], ref_tot[xstart:xend], label='Ref', marker='s', ms=5)
                        p.plot(x[xstart:xend], tot[xstart:xend], label='Exp', marker='o')
                        p.plot(x[xstart:xend], new_tot[xstart:xend], label='Shifted', marker='x')
                        p.legend()
                        save_fig()

                        #p.subplot(1,2,1)
                        p.title('xshift = {:.4f}'.format(xshift))
                        xstart, xend = 25, 40
                        p.plot(x[xstart:xend], ref_tot[xstart:xend], label='Ref', marker='s', ms=5)
                        p.plot(x[xstart:xend], tot[xstart:xend], label='Exp', marker='o')
                        p.plot(x[xstart:xend], new_tot[xstart:xend], label='Shifted', marker='x')
                        #p.subplot(1,2,2)
                        # zoom in on the only clear spectral line
                        #xstart, xend = 145, 160
                        #p.plot(x[xstart:xend], ref_tot[xstart:xend], label='Ref', marker='s', ms=5)
                        #p.plot(x[xstart:xend], tot[xstart:xend], label='Exp', marker='o')
                        #p.plot(x[xstart:xend], new_tot[xstart:xend], label='Shifted', marker='x')
                        #p.legend()
                        save_fig()
                exposure.xshift = xshift
                exposure.Primary.header['XSHIFT'] = xshift

            # Now compute wavelength solution given direct image position
            tot_image = 0.
            new_subs = []
            for i, subexposure in enumerate(subexposures):
                # Calculate various offsets
                image = subexposure.SCI.data.copy()

                Dxref = (POSTARG1- di_ps1) / 0.135
                # moving the telescope right moves the target right on the image as it is facing away

                # Different filters have small inherent direct image offsets
                filt = direct_image.Primary.header['FILTER']
                XOFF, YOFF = data.get_wfc3_filter_offs(filt)
                if XOFF is None:
                    logger.warning('Filter {} offset not known.'.format(filt))
                    XOFF, YOFF = 0., 0.

                # Guess position of x and y from DI
                xpix = x_di+xshift
                y0 = y_di + (scan_direction * (t.scan_rate * subexposure.SCI.header['SAMPTIME'])) / 0.121 # guess of y, in pixels
                if t.remove_scan and ((t.default_scan == 'r' and scan_direction == +1) or (t.default_scan == 'f' and scan_direction == -1)):
                    y0 -= (scan_direction * (t.scan_rate * exposure.Primary.header['EXPTIME'])) / 0.121 # undo full scan
                    if i==0: logger.info('default_scan=scan ({}). Undo full scan'.format(t.default_scan))
                if t.postarg_yguess:
                    y0 -= (exposure.Primary.header['POSTARG2'] - di_ps2) / 0.121
                    if i==0: logger.info('applying postarg offset to yguess, {:.2f} pix'.format((exposure.Primary.header['POSTARG2'] - di_ps2) / 0.121))
                if t.scanned:
                    width0 = subexposure.SCI.header['DELTATIM']*t.scan_rate/0.121 # initial guess of width
                else:
                    width0 = 40        
                if y0 + width0/2. > subexposure.SCI.data.shape[1]: y0 = subexposure.SCI.data.shape[1]-width0/2.
                elif y0 - width0/2. < 0: y0 = width0/2.
                # Fit for y scan height and position given guess
                ystart, yend = disp.get_yscan(image, x0=xpix, y0=y0, width0=width0, nsig=t.nysig, two_scans=t.two_scans, debug=False)

                subexposure.xpix = xpix
                subexposure.ystart = ystart; subexposure.yend = yend
                subexposure.ypix = (subexposure.ystart+subexposure.yend)/2.

                # Calculate wavelength solution
                subexposure.wave_grid = disp.dispersion_solution(x0=xpix, L=image.shape[0], Dxoff=XOFF, Dxref=Dxref, ystart=ystart, yend=yend, DISP_COEFFS=DISP_COEFFS, TRACE_COEFFS=TRACE_COEFFS, wdpt_grid_y=t.grid_y, wdpt_grid_lam=t.grid_lam)
                
                # Define wavelength grid to interpolate to
                # 0.9-1.92, 200
                wave_ref = np.linspace(t.ref_wv0, t.ref_wv1, 200) #subexposure.wave_grid[0] 
                # interpolate all rows to this row   
         
                subexposure.waves = wave_ref
                cut_image = image[ystart:yend,int(xpix):int(xpix)+200].copy() # cutout of spectral area
                if subexposure.SCI.data.shape[1] < xpix + 200:
                    subexposure.waves = subexposure.waves[:cut_image.shape[1]-200]
                    subexposure.wave_grid = subexposure.wave_grid[:,:cut_image.shape[1]-200]

                # Flat field correction (requres wavelength solution for more than 0th order)
                # Need to do before interpolating to a reference row
                if t.flat_field and step == 'final':
                    nys = subexposure.yend-subexposure.ystart
                    _waves = subexposure.wave_grid #subexposure.waves.repeat(nys).reshape(-1, nys).T
                    if subexposure.SCI.data.shape[1] < xpix + 200:
                        x1 = int(xpix)+cut_image.shape[1]
                    else: 
                        x1 = int(xpix)+200
                    L = subexposure.SCI.data.shape[0]
                    dL = (1014-L)/2
                    cut_image, ff, ff_error = cal.flat_field_correct( _waves, cut_image,
                                                    int(xpix)+dL, x1+dL, subexposure.ystart+dL, subexposure.yend+dL,
                                                    t.flat_file_g141, ff_min=t.ff_min)
                    subexposure.ff = ff
                    subexposure.ff_error = ff_error
                    #logger.info('Flat-field correction performed with full wavelength dependence')

                cut_mask = subexposure.mask[ystart:yend,int(xpix):int(xpix)+200]
                subexposure.cut_mask = cut_mask.copy()
                # New interpolation, area under pixel matches pixel flux
                interp_image, interp_mask = disp.interp_wave_grid_sane(subexposure.waves, subexposure.wave_grid,
                                                 cut_image, cut_mask, tol=t.contam_thresh)
                subexposure.interp_image = interp_image                
                subexposure.interp_mask = interp_mask

                subexposure.pre_mask = subexposure.mask.copy()
                subexposure.mask[ystart:yend,int(xpix):int(xpix)+200] = interp_mask
                if step == 'partial': subexposure.cut_image = cut_image # old image
                subexposure.interp_image = interp_image

                if step == 'partial':
                    image[subexposure.ystart:subexposure.yend,int(xpix):int(xpix)+200] = interp_image
                    tot_image += image

                if step == 'final':
                    subexposure.SCI.data[subexposure.ystart:subexposure.yend,int(xpix):int(xpix)+200] = interp_image

                    # Wavelength dependent flat field correction, some NaNs/inf creep in due to dead pixels in ff, change these to zeros
                    bad_pixels = np.logical_or(np.isinf(subexposure.SCI.data),np.isnan(subexposure.SCI.data))
                    subexposure.mask = np.logical_or(bad_pixels, subexposure.mask)
                new_subs.append(subexposure)
        subexposures = new_subs

    else:
        # just do the wavelength-indep flat-field correction
        if t.flat_field:
            for i, subexposure in enumerate(subexposures):
                # ff size is 1014x1014
                L = subexposure.SCI.data.shape[0]
                dL = (1014-L)/2
                if i == 0:
                    # only load ff file once
                    subexposure.SCI.data, ff0, ff0_error = cal.flat_field_correct(0, subexposure.SCI.data, 
                                                            x0=dL, x1=L+dL, ystart=dL, yend=L+dL,
                                                            flat_file=t.flat_file_g141, wave_dep=False, ff_min=t.ff_min)
                else:
                    subexposure.SCI.data /= ff0
                bad_pixels = np.logical_or(np.isinf(subexposure.SCI.data),np.isnan(subexposure.SCI.data))
                subexposure.SCI.data[bad_pixels] = 0
            logger.info('0th order flat-field correction performed')


    # Plot subexposures
    if t.debug:
        if t.flat_field:
            if not t.dispersion: ff = ff0
            mn_ff = np.nanmean(ff)
            arrays_plot([sub.ff for sub in subexposures], name=None, cbar=False, size=4, height=0.5, \
                                    tight_layout=False, vmin=0.9, vmax=1.1, show=False)
            p.suptitle('Flat-field for each exposure')
            save_fig()
            if t.dispersion:
                _, ff0, ff0_error = cal.flat_field_correct(0, np.ones((L, x1-int(xpix))), 
                                                            int(xpix)+dL, x1+dL, dL, L+dL,
                                                            t.flat_file_g141, wave_dep=False,
                                                            ff_min=t.ff_min)
                view(ff0, title='Zeroth flat-field: mean {:.4f}'.format(np.nanmean(ff0)), cbar=True, cmap='binary_r', vmin=0.9, vmax=1.1, show=False)
                save_fig()

        arrays_plot([sub.SCI.data for sub in subexposures], name='Subexp', cbar=False, size=2, \
                        tight_layout=False, vmin=0., vmax=100, show=False)
        if t.dispersion and len(subexposures) > 1: # add in marker of where the spectrum is centered
            nrows = np.ceil(len(subexposures)/4.)
            for i, subexposure in enumerate(subexposures):
                #x, y = subexposure.xpix, subexposure.ypix
                p.subplot(nrows, 4, i+1)
                #p.plot(x,y,marker='o', mfc='None', color='w')
                p.plot(xpix,y_di,marker='x', mfc='None', color='w')
                p.plot([xpix,xpix],[subexposure.ystart,subexposure.yend], color='w')
        save_fig()
        if t.dispersion:
            view(subexposures[0].wave_grid, show=False, title='Wavelength grid for subexposure 0')
            save_fig()

    # Local CR removal
    if t.cr_local:
        new_subs = []
        for i, subexposure in enumerate(subexposures):  
            ignore_mask = subexposure.mask # dont flag already masked pixels
            
            CR_clean, CR_mask = r.spatial_median_filter(subexposure.SCI.data.copy(), ignore_mask, tol=t.cr_tol, replace=t.cr_replace,\
                                    debug=False, sx=t.cr_x, sy=t.cr_y, thresh=t.cr_thresh)
            n_crs = np.count_nonzero(CR_mask)

            subexposure.SCI.data = CR_clean
            #n_crs = np.count_nonzero(CR_mask[ypix-t.psf_h/2:ypix+t.psf_h/2,xpix-100:xpix+100])
            subexposure.n_crs = n_crs
            subexposure.CR_mask = CR_mask
            logger.info('Removed {} CR pixels from subexposure {}'.format(n_crs,i+1))
            subexposure.mask = np.logical_or(subexposure.mask, CR_mask)

            subexposure.SCI.header['CRs'] = (n_crs, 'Number of crs detected in box (local median)')
            # Remove CR hits, with NaN or custom CR_replace value
            if t.cr_replace: subexposure.mean_mask_pix(CR_mask, replace=t.cr_replace)

            if t.cr_plot:
                # Plot locations of CR hits
                view(subexposure.SCI.data, cbar=False, show=False, cmap='binary_r', alpha=1)
                ax=p.gca()
                ax.set_autoscale_on(False)
                for j in range(subexposure.SCI.data.shape[0]):
                    for k in range(subexposure.SCI.data.shape[1]):
                        if CR_mask[j,k]: p.plot(k, j, marker='o', mec='r', mfc='None')
                p.title('{} crs'.format(n_crs))
                p.show()
            new_subs.append(subexposure)
        subexposures = new_subs
    # Master CR removal
    elif t.cr_master:
        # Use a master image, generated with no cr removal applied, to detect CRs
        master = pyfits.open(t.cr_mname)
        master_image = master[0].data
        master_std = master[1].data

        new_subs = []
        for i in range(len(subexposures)):
            subexposure = subexposures[i]
            
            CR_mask = abs((subexposure.SCI.data - master_image) / master_std) > t.cr_tol
            CR_mask = np.logical_and(CR_mask, abs(subexposure.SCI.data - master_image)>t.cr_thresh)
            n_crs = np.count_nonzero(CR_mask)

            subexposure.CR_mask = CR_mask
            subexposure.n_crs = n_crs

            logger.info('Detected {} CR pixels from subexposure {}'.format(n_crs,i+1))
            subexposure.mask = np.logical_or(subexposure.mask, CR_mask)
            subexposure.SCI.header['CRs'] = (n_crs, 'Number of crs detected cf to master image')
            new_subs.append(subexposure)

            p.subplot(1,2,1)
            p.suptitle('Subexposure {}'.format(i))
            view(subexposure.SCI.data - master_image, title='Diff', cbar=False, show=False)
            p.subplot(1,2,2)
            subexposure.cr_sigma = abs((subexposure.SCI.data - master_image) / master_std)
            view(abs((subexposure.SCI.data - master_image) / master_std), title='Sigma', vmin=0, vmax=10, cbar=False, show=False)
            save_fig()
        subexposures = new_subs

    else:
        new_subs = []
        for i in range(len(subexposures)):
            subexposure = subexposures[i]
            subexposure.n_crs = 0
            subexposure.CR_mask = np.zeros_like(subexposure.SCI.data).astype(bool)
            subexposure.SCI.header['CRs'] = (0, 'Number of crs detected in box (local median)')
            new_subs.append(subexposure)
        subexposures = new_subs

    if t.debug and (t.cr_local or t.cr_master):
        # Plot number of CRs over subexposures
        # and plot cr distribution over x and y pixels
        n_crs = [subexposure.n_crs for subexposure in subexposures]
        if len(n_crs) > 1:
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
        view(all_CRs, title='Distribution of CRs over exposure', cbar=False, show=False, vmin=0, vmax=1, cmap='binary_r')
        save_fig()

    exposure.subexposures = subexposures

    if t.pdf and t.debug:
        pdf.close()

    if t.save_dir:
        # Save reduced fits file (_red.fits format)
         logger.info('Saving reduced file to {}'.format(t.save_dir))
         data.write_reduced_fits(exposure.subexposures, exposure.Primary, t, dest_dir=t.save_dir)

    logger.info('Reduction took {:.2f}s'.format(time.time()-t0))
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
    (some issues loading from file, better to run reduction and extraction consecutively)

    kwargs: custom toggles, assigned by keyword arguments
            
            debug (bool): print and plots some steps for debugging
            logger (bool): log results to terminal (False) or file (True)
            pdf (bool): save plots to a pdf

            save_dir (str): destination to save the spectrum
            save_extension (str): _spec.txt default, extension for spectrum file
            save_sub (bool): save the subexposure spectra separately

            calc_var (bool): Calculate variances for the spectra or use ERR extension
            mask_neg (bool): can mask all the negative pixels in case of strong negative persistence
            extraction_box (int): n pixel extraction box before optimal extraction
            box_h (int): height of initial extraction box used on spectrum in pixels
            
            ignore_blobs (bool): avoid reads contaminated heavily by IR blobs
            blob_thresh (float): 0.7
            
            shift_spectra (bool): shift spectra of each subexposure to correct for drift (linear interp)
            peak (bool): use only spectrum peak for shift calculation (noisy)
            shift_wv0, shift_wv1 (floats): wavelengths to define peak
            
            OPTIMAL EXTRACTION
            opt_ext (bool): use optimal extraction
            box_h (int): height of extraction box
            s (float): sky average or background noise (array-like or scalar), zero if bg removed
            v_0 (float): variance of readout noise
            q (float): effective photon number per pixel value (=1)
            s_clip (float): sigma threshold for outliers in spatial fit, None skips
            s_cosmic (float): sigma threshold for cosmic ray removal, None skips cr removal
            func_type (str): type of function used for fit (poly, gauss or heavi)
            method (str): which optimization method, lsq or one from scipy.optimize.minimize (e.g. CG)
            fit_tol (float): tolerance for the fit, fraction of each flux point
            step (float): stepsize in lstsq fitting (epsfcn)
            order (int): order of the spline fit (if func_type=spline) default to 2
            remove_bg (bool): remove the background before optimal extraction or leave it in
            skip_fit (bool): don't fit profile to spectrum but just use fluxes as weights
            k (int): smoothing length for profile calculation
            oe_debug (bool): show debug plots for optimal extraction
            oe_pdf (bool): save plots to seperate pdf
    '''
    t0 = time.time()
    # Load config_file and kwargs into dictionaty 't'
    toggles = { 'debug': False, 'logger':True, 'pdf':False, 
                'save_dir': None, 'save_extension': '_spec.txt', 'save_sub':False,
                'calc_var': True, 'mask_neg': False, 'extraction_box': True, 'box_h': 80, 
                'ignore_blobs': False, 'blob_thresh':0.7, 
                'shift_spectra': False, 'shift_wv0':1.14, 'shift_wv1':1.6, 'peak':False,
                'opt_ext': True, 's': 0, 'v_0': 20**2, 'q': 1, 's_clip': None, 's_cosmic': None, 'func_type': 'spline',
                'method': 'lsq', 'fit_tol':0.01, 'save_dir':None, 
                'step': None, 'order': 2, 'skip_fit': False, 'remove_bg': True, 'top_half':False,
                'k_col': 9, 'k_row':None, 'object_ind':0, 'oe_debug':0, 'oe_pdf':None,
                }
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
        reduced_exposure = data.load(reduced_exposure)

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
        logger = Empty_logger()
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
    Ds, DQs, CRs, Ps = [], [], [], []
    spectra, variances = [], []
    for n_sub, subexposure in enumerate(subexposures):
        logger.info('Extracting from subexposure {}'.format(n_sub+1))
        
        # Get background level
        try: bg = subexposure.bg; bg_err = subexposure.bg_err # background that has been removed
        except AttributeError: bg = 0.; bg_err = 0.; logger.warning('No background defined')

        # Cut out extraction box
        if t.extraction_box:
            if n_sub == 0: logger.info('Extracting spectrum with a {} high extraction box'.format(t.box_h))
            xshift = True
            if not hasattr(subexposure, 'waves'): subexposure.waves = np.arange(subexposure.SCI.data.shape[1])

            xpix = subexposure.xpix + 100
            D, mask, bg, err = map(lambda image: r.box_cut(xpix, image, 200, horizontal=True, force_shape=False), \
                            [subexposure.SCI.data, subexposure.mask, bg, subexposure.ERR.data])
            logger.info('Spectrum located at pixel {} in spatial direction'.format(xpix))

            ypix = subexposure.ypix
            D, mask, bg, err = map(lambda image: r.box_cut(ypix, image, t.box_h, force_shape=False), [D, mask, bg, err])

        else:
            if n_sub == 0: logger.warning('No extraction box used')
            D = subexposure.SCI.data
            mask = subexposure.mask
            err = subexposure.ERR.da

        # Get errors
        if not t.calc_var:
            if n_sub == 0: logger.info('Using ERR extension for variance estimate')
            V = np.square(err) # gain correction already done to _ima files
            t.v_0 = 0
            # Use errors to estimate variance per pixel
        else:
            if n_sub == 0: logger.info('Using flux for variance estimate')
            V = np.abs(D) + bg_err**2 + t.v_0 # gain correction already done to _ima files

        if t.top_half:
            ih = 45 #D.shape[0]/2
            D, mask, V = map(lambda arr: arr[ih:], [D, mask, V])
            #D, mask, V = map(lambda arr: arr[:ih], [D, mask, V])

        Ds.append(D)
        DQs.append(mask)
        CRs.append(np.ones_like(mask))

        if t.opt_ext:
            # Change mask to optimal extraction format
            # 1s where good, 0s where bad
            M_DQ, M_CR = np.logical_not(mask).astype(int), np.ones_like(mask) # just put CRs in DQ mask
            if t.skip_fit: logger.warning('Not fitting profile, using flux instead')

            # toggle removing the background before optimal extraction
            # or can handle the background in the extraction
            if t.remove_bg:
                D, S = D, 0
            else:
                D, S = D + bg, bg

            if t.oe_pdf:
                oe_pdf_file = t.save_dir+'logs/{}_{}_fit.pdf'.format(reduced_exposure.Primary.header['ROOTNAME'], n_sub)
                f.silentremove(oe_pdf_file)
            else: oe_pdf_file = None
            logger.info('Extracting spectrum with optimal extraction...')
            spec, specV, P, V = ea.extract_spectrum(D=D, S=S, V_0=t.v_0, Q=t.q, V=V, s_clip=t.s_clip, s_cosmic=t.s_cosmic, \
                                    func_type=t.func_type, method=t.method, debug=t.oe_debug, tol=t.fit_tol, M_DQ=M_DQ, M_CR=M_CR, \
                                    pdf_file=oe_pdf_file, step=t.step, order=t.order, skip_fit=t.skip_fit, bg=bg,  \
                                    k_col=t.k_col, k_row=t.k_row, logger=logger)
            if t.debug and np.any(np.isnan(P)): view(D); view(P, show=False); view(np.isnan(P), alpha=0.5, cmap='binary',cbar=False)
            if np.any(np.isnan(spec)) and t.debug: p.plot(spec); p.title('NaNs in spectrum'); save_fig()
            assert not np.any(np.isnan(P)), 'NaNs in spatial profile for subexposure {}'.format(n_sub)
            if np.any(np.isnan(spec)):
                logger.warning('NaNs in spectrum for subexposure {}'.format(n_sub))
                logger.warning('Replaced with zeros')
                spec[np.isnan(spec)] = 0.

        else:
            if n_sub == 0: logger.info('Not using optimal extraction - results will be noisy')
            M = np.logical_not(mask)
            # Use a smoothed image for weights
            k = 5 # smooth along k pixels
            # Now do a median smooth along each column
            n_smoothloops = 0 #4
            P = D.copy()
            for _ in range(n_smoothloops):
                smooth_spec = []
                for col in P.T:
                    col = ea.medfilt(col, k)
                    smooth_spec.append(col)
                P = np.vstack(smooth_spec).T
            P = P / np.sum(P, axis=0)
            V = ea.initial_variance_estimate(D=D, V_0=t.v_0, Q=t.q) + bg_err**2 # background calc affects variance, normally handled by opt_ext
            spec, specV = ea.optimized_spectrum(D, t.s, P, V, M)
        Ps.append(P)
        # Optimal Extraction
        # Sum spatial
        spectrum = f.Spectrum(subexposure.waves,spec,x_unit='Wavelength (microns)', y_unit='electrons')
        spectra.append(spectrum)
        variances.append(specV)
    reduced_exposure.Ds = Ds; reduced_exposure.DQs = DQs; reduced_exposure.CRs = CRs

    if t.debug:
        arrays = [ ]
        for D, DQ_mask in zip(Ds, DQs):
            array = D.copy()
            array[DQ_mask] = np.nan
            arrays.append(array)
        vmin = 0.; vmax = np.nanmax(arrays[0])
        arrays_plot(arrays, vmin=vmin, vmax=vmax, cbar=False, tight_layout=False,\
                        size=3, name='Box', show=False, height=0.5)
        save_fig()
        vmax2 = reduced_exposure.Primary.header['EXPTIME']*2
        arrays_plot(arrays, vmin=vmin, vmax=vmax2, cbar=False, tight_layout=False,\
                        size=3, name='Box', show=False, height=0.5)
        save_fig()

    if t.debug:
        p.figure()
        #p.subplot(2,1,1)
        p.title('Subexposure spectra')
        for i, spec in enumerate(spectra):
            spec.plot(show=False, label=i)
        p.legend(fontsize='x-small')
        #p.subplot(2,1,2)
        #for i, spec in enumerate(spectra):
        #    spec.plot(show=False, label=i)
        #p.xlim([1.1,1.6])
        #p.ylim([400000,600000])
        #p.tight_layout()
        save_fig()


    # Can't use subexposures where there are too many bad pixels on spectrum
    # threshold is what percentage of the flux missing
    if t.ignore_blobs:
        Ms = DQs
        bad_subs = np.array([ np.any(np.sum(M*P, axis=0)[50:150] > t.blob_thresh) for P, M in zip(Ps, Ms) ]).astype(bool)
        # only check the inner 100 pixels, outside is mostly bg and so can be masked
        good_subs = np.logical_not(bad_subs)
        logger.warning('Ignoring subexposures {} due to bad pixels on spectrum'.format(np.arange(len(bad_subs))[bad_subs]))

        if t.debug:
            for i in range(len(good_subs)):
                spec = spectra[i]
                gd = good_subs[i]
                if not gd:
                    ls = '--'
                    alpha = 1.
                    color = None
                    label = 'Sub {}, ignored'.format(i)
                else:
                    ls = '-'
                    alpha=0.1
                    color='k'
                    label = 'Sub {}'.format(i)
                p.plot(spec.x, spec.y, label=label, ls=ls, alpha=alpha, color=color)
            p.xlabel('Wavelength (micron)')
            p.ylabel('electrons')
            p.title('Spectra of subexposures (-- ignored)')
            p.legend()
            save_fig()

            size = 3
            nrows = np.ceil(len(good_subs)/4.)
            rcParams['figure.figsize'] = size*4, size*nrows/2
            for i in range(len(good_subs)):
                p.subplot(nrows, 4, i+1)
                image = Ds[i].copy()
                M = Ms[i]
                image[M] = np.nan
                if not good_subs[i]: title='Ignored'
                else: title=''
                view(image, title=title, cbar=False, show=False)
                p.axis('off')
            p.tight_layout()
            save_fig()
            rcParams['figure.figsize'] = 10, 5


        old_combined = np.sum([spec.y for spec in spectra], axis=0)
        spectra = [ spec for spec, gd in zip(spectra, good_subs) if gd == 1 ]
        variances = [ var for var, gd in zip(variances, good_subs) if gd == 1 ]
    else:
        good_subs = np.ones(len(Ps))

    reduced_exposure.spectra = spectra
    reduced_exposure.variances = variances

    # So now add all the scaled spectra together, interpolated to a common wavelength scale
    interp_spectra = []
    # templates wavelength from first (in time)
    x, y = spectra[-1].x, spectra[-1].y
    sub_shifts = []
    # actual interpolation
    if t.shift_spectra:
        i0, i1 = np.argmin(abs(x-t.shift_wv0)), np.argmin(abs(x-t.shift_wv1))
        # Fit for an x offset between subexposures
        for spec in spectra:
            if t.peak:
                shift, err = r.spec_pix_shift(x[i0:i1], y[i0:i1], spec.x[i0:i1], spec.y[i0:i1], norm=True)
            else:
                shift, err = r.spec_pix_shift(x, y, spec.x, spec.y, norm=True)
            sub_shifts.append((shift,err))
            shift_y = np.interp(x, x+shift, spec.y)
            interp_spectra.append(shift_y)
    else:
        # Just interpolate to the first subexposure
        interp_spectra = [np.interp(x, spec.x, spec.y) for spec in spectra]
        sub_shifts = [0]*len(interp_spectra)
    reduced_exposure.sub_shifts = sub_shifts

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
        logger.info('{} subexposures used out of {}'.format(n_subs, len(subexposures)))
        unit = 'Electrons'
    else:
        # Don't rescale
        unit = 'Electrons'

    exp_spectrum = f.Spectrum(x, y, x_unit='Spectral Pixel', y_unit=unit)

    if t.debug:
        p.plot(exp_spectrum.x, exp_spectrum.y, label='Combined spectrum')
        if t.ignore_blobs: 
            p.plot(exp_spectrum.x, old_combined, ls='--', zorder=-1, label='Including bad spectra')
            p.legend()
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

        if t.save_sub:
            # mark if the scan is forward or reverse, better not to do it here, instead use header of original file

            fname = t.save_dir + reduced_exposure.Primary.header['ROOTNAME'] + '_subs'+end
            f.silentremove(fname)
            logger.info('Saving subexposure spectra to {}'.format(fname))

            lines = []
            for i, wv in enumerate(exp_spectrum.x):
                line =  str(wv)
                for spec, var in zip(interp_spectra, variances):
                    line += '\t{}\t{}'.format(spec[i], var[i]**0.5)
                line += '\n'
                lines.append(line)
            with open(fname, 'w') as txtf:
                # this assumes you did wavelength calibration
                txtf.write('wave\tflux\terror\tfor each subexposure number\n')
                for line in lines:
                    txtf.write(line)

    logger.info('Extraction took {:.2f}s'.format(time.time()-t0))
    logger.info('########################################')
    logger.info('###########Finished Extraction##########')
    logger.info('########################################')

    if t.pdf and t.debug:
        pdf.close()

    # Logging
    if t.logger:
        # need to close all the file handlers explicitly if performing multiple extractions
        for handler in logger.handlers:
            if hasattr(handler,'close'):
                handler.close()

    return exp_spectrum, variance, interp_spectra, variances

def create_orbit_cats_gauss(data_dir='/home/jacob/hst_data/GJ-1214/', gridsize=5, use_ima=False, nstars=2):
    '''
    Fit a gaussian to direct image to create visit catalogue.
    Use same format as SExtractor for catalogue files.
    '''
    reload(data)
    from lmfit import minimize, Parameters
    from astropy import modeling

    if not os.path.exists(data_dir+'input_image.lis'):
        data.make_input_image_list(data_dir=data_dir)
    if not os.path.exists(data_dir+'visit_driz.lis'):
        data.make_driz_list(data_dir=data_dir)

    Gaussian2D = modeling.functional_models.Gaussian2D

    with open(data_dir+'visit_driz.lis', 'r') as driz_list:
        for line in driz_list:
            fname = line[:-1]
            dest_file = data_dir + fname.split('_')[0]+'_flt_1.cat'
            print('Writing to: {}'.format(dest_file))
            if not use_ima: 
                flt_fname  = data_dir + fname.split('_')[0]+'_drz.fits'
                di = data.load(flt_fname)
                full_images = [di.SCI.data.copy()]
                full_image = full_images[0]
            else:
                # X02b data has the direct image spatially scanned
                # use the first read of the _ima before scanning as direct image
                flt_fname = data_dir + fname.split('_')[0]+'_ima.fits'
                di = data.load(flt_fname)
                full_images = [read.SCI.data.copy() for read in di.reads]
                full_images = [image[5:-5,5:-5] for image in full_images] # trim off reference pixels
                full_image = full_images[0]
            
            full_image[np.logical_not(np.isfinite(full_image))] = 0.
            full_image[full_image<0] = 0
            print('Shape '+str(full_image.shape))

            view(full_image, cmap='binary_r', title='Full image - {}'.format(flt_fname.split('/')[-1]), vmin=0, vmax=np.nanmax(full_image)/100, show=False)
            # Find the part of the image to search, in case there are multiple sources. Area of all pixels except brightest to avoid cosmics
            areas = np.array([[ np.sum(np.square(full_image[gridsize*i:gridsize*(i+1),gridsize*j:gridsize*(j+1)])) 
                        for j in range(full_image.shape[1]/gridsize)]
                            for i in range(full_image.shape[0]/gridsize)]) \
                    - np.array([[ np.max(np.square(full_image[gridsize*i:gridsize*(i+1),gridsize*j:gridsize*(j+1)])) 
                        for j in range(full_image.shape[1]/gridsize)]
                            for i in range(full_image.shape[0]/gridsize)])
            # Sort by largest area
            _areas = areas.flatten()
            _areas, indexes = zip(*sorted(zip(_areas, range(len(_areas))), reverse=True))
            # convert indexes back into coords
            indexes = [ (index % areas.shape[1], index / areas.shape[0]) for index in indexes ]
            for cnt, ind in enumerate(indexes[:nstars]):
                x_ind, y_ind = ind
                x_ind *= gridsize; y_ind *= gridsize
                p.plot(x_ind, y_ind, marker='o', ms=gridsize*2.5, mfc='None', color='r')
            p.show()

            for cnt, ind in enumerate(indexes[:nstars]):
                x_ind, y_ind = ind
                x_ind *= gridsize; y_ind *= gridsize
                image = full_image[y_ind-gridsize:y_ind+2*gridsize, x_ind-gridsize:x_ind+2*gridsize]

                params = Parameters()
                params.add_many(('amplitude', np.nanmax(image), True, 0.), \
                                ('x_mean', image.shape[1]/2, True, 0, image.shape[1]), \
                                ('y_mean', image.shape[0]/2, True, 0, image.shape[0]), \
                                ('x_stddev', 10, True, 0), ('y_stddev', 10, True, 0))

                size_y, size_x = image.shape
                x = np.repeat(np.arange(0,size_x),size_y).reshape(size_y,size_x).T
                y = np.repeat(np.arange(0,size_y),size_x).reshape(size_y,size_x)

                def residuals(params, image, x, y):
                    model = Gaussian2D.evaluate(x, y, amplitude=params['amplitude'], x_mean=params['x_mean'], y_mean=params['y_mean'], \
                                                x_stddev=params['x_stddev'], y_stddev=params['y_stddev'], theta=0)
                    return (image - model).flatten()

                out = minimize(residuals, params, args=(image, x, y))
                params = out.params
                fit_params = params

                params.pretty_print()
                x_image, y_image = fit_params['x_mean'].value, fit_params['y_mean'].value
                x_std, y_std = fit_params['x_stddev'].value, fit_params['y_stddev'].value
                # Restore to full image reference, also accidentally labeled them all wrong
                # x_image here refers to 0th axis, which is the spatial scan direction
                # y_image is 1st axis, spectral direction, normally called x
                x_image, y_image = x_image+x_ind-gridsize, y_image+y_ind-gridsize

                view(image, show=False, cmap='binary_r', vmin=0, vmax=np.nanmax(full_image)/100, title='({:.1f}, {:.1f})'.format(x_image, y_image))
                ax = p.gca()
                ax.set_autoscale_on(False)
                p.plot(params['x_mean'].value, params['y_mean'].value, marker='x', color='r')
                #p.xlim([params['x_mean'].value-25, params['x_mean'].value+25])
                #p.ylim([params['y_mean'].value-25, params['y_mean'].value+25])
                p.show()

                line = '\t'.join([str(cnt+1), str(x_image), str(y_image), '0', '0', \
                            str(y_std), str(x_std), '0.0', '0', '0', '0', '0.0']) + '\n'
                
                if cnt == 0:
                    with open(dest_file, 'w') as g:
                    
                        g.write('# 1  NUMBER  Running object number\n# 2  X_IMAGE  Object position along x  [pixel]\n')
                        g.write('# 3  Y_IMAGE  Object position along y  [pixel]\n# 4  X_WORLD  Barycenter position along world x axis  [deg]\n')
                        g.write('# 5  Y_WORLD  Barycenter position along world y axis  [deg]\n# 6  A_IMAGE  Profile RMS along major axis  [pixel]\n')
                        g.write('# 7  B_IMAGE  Profile RMS along minor axis  [pixel]\n# 8  THETA_IMAGE  Position angle (CCW/x)  [deg]\n')
                        g.write('# 9  A_WORLD  Profile RMS along major axis (world units)  [deg]\n# 10 B_WORLD  Profile RMS along minor axis (world units)  [deg]\n')
                        g.write('# 11 THETA_WORLD  Position angle (CCW/world-x)  [deg]\n# 12 MAG_F1384  Kron-like elliptical aperture magnitude  [mag]\n')

                        g.write(line)
                else:
                    with open(dest_file, 'a') as g:
                        g.write(line)
                # catalogue create for direct image
