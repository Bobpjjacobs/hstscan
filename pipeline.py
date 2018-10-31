from __future__ import print_function
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

import dispersion as disp

reload(r)
reload(ea)
reload(data)
reload(disp)
reload(cal)
rcParams['pdf.fonttype'] = 42
#rcParams['font.family'] = 'Calibri'

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

class Empty_logger():
    def info(self,x): print(x)
    def warning(self,x): print(x)
    def debug(self,x): print(x)

def arrays_plot(arrays, name='Read', tight_layout=True, size=3, height=1, show=True, **kwargs):
        # Neatly plot a list of arrays, using data.view_frame_image as view
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

def compute_exposure_shifts(visits, source_dir, save_dir=None, verbose=True):
    '''Compute shift of spectrum between exposures in the same visit

    Shifts are defined as the difference between the position of the star on the exposure
    and the position of the star on the reference exposure.
    So the -ve of the output of r.spec_pix_shift.
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
            shift = r.spec_pix_shift(x, fl1, x, fl2)

            all_rootnames.append(rn)
            all_shifts.append(-shift)

    if save_dir is None: save_dir=source_dir
    with open(save_dir+'exposure_shifts.lis', 'w') as g:
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
        - remove cosmic rays with local (slow) or sliding median filter
        - calculating dispersion solution
        - interpolate image to common wavelength scale (slow)
        - apply flat-field correction
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
            CR_local (bool): cosmic ray removal by taking local median
            CR_slide (bool): CR removal using a sliding median filter, should not be used for drift scans
            flat_field (bool): wavelength dep flat field correction
            bg: calculate and remove background using interpolation
            dispersion: interpolate to wavelength scale using dispersion solution

            VALUES
            data_dir: where the source fits file is stored
            object_ind: object index (brightest first) to extract from image
            CR_tol: change tolerance for local CR hits
            CR_sigma: change the sigma limit in sliding median for a CR detection
            CR_thresh: lower limit for a cosmic ray dump, default 500 electrons
            CR_replace: change the value that replaces a CR pixel
            CR_width: number of adjacent subexposures used for sliding median filter
            DQ_replace: value replacing DQ flagged bad pixels
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
                'cr_width': 5, 'cr_thresh': 500,  'cr_replace': 'median', 'dq_replace': None, 'dq_mean_width':1, 'save_dir': None, 'contam_thresh': 0.01, \
                'debug': False, 'pdf': False, 'bg_box':200, 'bg_plot': False, 'psf_h':130, 'logger': True, 'force_no_offset': False, 'neg_masks': 1, \
                'shift_spectra': True, 'cr_plot': False, 'cr_x': 1, 'cr_y': 1, 'mask_h':40, 'dq_flags': [4, 32], 'psf_w':220, \
                'spatial_plot':False, 'scanned':True, 'scan_rate':None, 'object_ind': 0, 'check_sextractor':False,
                'fit_image':False, 'cat':None, 'bg_area':False, 'bg_x':0, 'bg_y':0, 'bg_h':50, 'bg_w':50, \
                'flat_file_g141':'None', 'conf_file_g141':'None', 'nlincorr':False, 'nysig':5, 'grid_y':20, 'grid_lam':20, 'xoff_file':'exposure_shifts.lis', 'two_scans':False, 'oversample': None}
    # read in conf_file to update default values
    if conf_file:
        conf_kwargs = data.read_conf_file(conf_file)
        conf_kwargs.update(**kwargs)
    else: conf_kwargs = kwargs
    t = f.bunch_kwargs(kwargs=conf_kwargs, toggles=toggles, name='reduction', verbose=True)

    # open up an exposure if you input a filename
    if type(exposure) == str:
        exposure = data.load(t.source_dir+exposure, bjd=False)

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
        logger = Empty_logger()
        
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
        if not read.trimmed: read.trim_pix()

        # Mask bad pixel flags
        mask = read.remove_bad_pix(replace=t.dq_replace, int_flags=t.dq_flags, width=t.dq_mean_width)

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
        if t.dq_flags is not None:
            n_rows = np.ceil(len(t.dq_flags)/2.)
            if len(t.dq_flags) % 2 == 0: n_rows += 1 # for the combined DQ mask  

            p.figure(figsize=(6,3*n_rows))
            p.subplot(n_rows,2,1)
            view(masks[0], title='All DQ pixels ({})'.format(t.dq_flags), cmap='binary_r', cbar=False, show=False, xlabel='', ylabel='')
            for k, flag in enumerate(t.dq_flags):
                _mask = reads[0].DQ.data/flag % 2 == 1
                p.subplot(n_rows,2,k+2)
                view(_mask, title='DQ {}'.format(flag), cmap='binary_r', cbar=False, show=False,xlabel='', ylabel='')
            p.tight_layout()        
            save_fig()
    if t.debug:
        vmin = 0.; 
        #vmax = np.max(reads[0].SCI.data)
        vmax = None
        arrays_plot([read.SCI.data for read in reads], cbar=False, vmin=vmin, vmax=vmax, \
                        tight_layout=False, size=2, show=False)
        save_fig()


    # Work on subexposures
    subexposures = []
    if exposure.filename.endswith('_flt.fits'): _n = 0 # only 1 read!
    else: _n = 2 # skip 1 for zeroth, 1 for fast read

    for i in range(len(exposure.reads)-_n):
        if exposure.filename.endswith('_flt.fits'):
            # treat flt file as a single subexposure
            subexposure = exposure.reads[i]
            DQ_mask = subexposure.remove_bad_pix(replace=t.dq_replace, int_flags=t.dq_flags, width=t.dq_mean_width)
        else:
            read2, read1 = exposure.reads[i], exposure.reads[i+1]
            DQ_mask = np.logical_or(masks[i], masks[i+1])
            # Form subexposure
            subexposure = r.create_sub_exposure(read1, read2, read_noise=t.read_noise, nlincorr=t.nlincorr)
        subexposure.DQ_mask = DQ_mask
        subexposure.mask = DQ_mask # total mask
        
        # do the wavelength-indep flat-field correction
        if t.flat_field:
            # ff size is 1014x1014
            L = subexposure.SCI.data.shape[0]
            dL = (1014-L)/2
            if i == 0:
                # only load ff file once
                subexposure.SCI.data, ff0, ff0_error = cal.flat_field_correct(0, subexposure.SCI.data, 
                                                        x0=dL, x1=L+dL, ystart=dL, yend=L+dL,
                                                        flat_file=t.flat_file_g141, wave_dep=False)
            else:
                subexposure.SCI.data /= ff0
            logger.info('0th order flat-field correction performed')
        subexposures.append(subexposure)

    # Background removal
    for i in range(len(subexposures)):
        subexposure = subexposures[i]
        if t.bg:
            if t.bg_area:
                t.bg_x, t.bg_y, t.bg_h, t.bg_w = map(int, [t.bg_x, t.bg_y, t.bg_h, t.bg_w])
                # Using a fixed area of the detector to calculate bg mean
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
        # Example of background image
        if t.bg and t.bg_area:
            view(exposure.reads[0].SCI.data, show=False, vmin=0, vmax=50*len(subexposures), title='Background Area')
            view(bg_mask, alpha=0.5, show=False, cbar=False, cmap='binary_r')
            save_fig()

        if t.bg:
            # Plot of background over time
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
                                name=None, cbar=False, size=2, height=h, tight_layout=False, vmin=0., vmax=100, show=False)
                p.suptitle('Bg Area for each subexposure')
                save_fig()


    # Calculate dispersion solution
    if t.dispersion:
        DISP_COEFFS, TRACE_COEFFS = disp.get_conf_coeffs()
        new_subs = []
        POSTARG1, POSTARG2, PA_V3 = exposure.Primary.header['POSTARG1'], exposure.Primary.header['POSTARG2'], exposure.Primary.header['PA_V3']
        # Find scan direction from positional offset
        if POSTARG2 >= 0.: scan_direction = +1; logger.info('Forward scan')
        else: scan_direction = -1; logger.info('Reverse scan')

        if t.scanned:
            if not t.scan_rate: 
                t.scan_rate = exposure.Primary.header['SCAN_RAT']
                if t.scan_rate != 0.:
                    logger.warning('SCANNED=True while exposure scan rate is zero')
            else:
                if 'SCAN_RAT' in exposure.Primary.header:
                    assert t.scan_rate == exposure.Primary.header['SCAN_RAT'], \
                        'Scan rates do not match (input {}, fits {})'.format(t.scan_rate, exposure.Primary.header['SCAN_RAT'])
        else: t.scan_rate = 0.

        # Find direct image position
        catalogue, di_name = data.find_catalogue(exposure.rootname, data_dir=t.source_dir)
        if not os.path.isfile(catalogue):
            logger.warning('No catalogue file found for {}'.format(catalogue))
            # then catalogue may be for visit drizzled file
            catalogue = catalogue.split('_')
            catalogue[0] = catalogue[0][:-3]+'010'
            catalogue = '_'.join(catalogue)
        direct_image = data.Data_ima(t.source_dir+di_name+'_ima.fits', bjd=False)
        di_size = direct_image.reads[0].SCI.shape[0] 
        di_image = direct_image.reads[0].SCI.data
        if np.log2(di_size) % 1 != 0: 
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
    
        # Get pixel shift for given exposure
        try:
            with open(t.source_dir+t.xoff_file, 'r') as g: lines = g.readlines()
            rootnames, shifts = zip(*[ line.split() for line in lines ])
            ishift = list(rootnames).index(exposure.rootname)
            xshift = float(shifts[ishift])
        except IOError:
            logger.warning('Could not load in shift file {}'.format(t.xoff_file))
            xshift = 0
        exposure.xshift = xshift

        for i, subexposure in enumerate(subexposures):
            # Calculate offset
            logger.debug('Subexposure {}'.format(i))

            Dxref = (POSTARG1- di_ps1) / 0.135
            # moving the telescope right moves the target right on the image as it is facing the other way

            # Different filters have different inherent direct image offsets, refer to:
            #http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2010-12.pdf
            # this is because the conf file was made using the F140W filter
            # For WFC3/IR F130N: dx = 0.033 +- 0.014, dy = 0.004 +- 0.019
            # For WFC3/IR F126N: dx = 0.264 +- 0.018, dy = 0.287 +- 0.025
            # For WFC3/IR F139M: dx = 0.11 +- 0.022, dy = 0.029 +- 0.028
            filt = direct_image.Primary.header['FILTER']
            if   filt == 'F126N': XOFF, YOFF = 0.264, 0.287
            elif filt == 'F130N': XOFF, YOFF = 0.033, 0.004
            elif filt == 'F139M': XOFF, YOFF = 0.11,  0.029
            elif filt == 'F132N': XOFF, YOFF = 0.039, 0.154
            else:
                logger.warning('Filter {} offset not known.'.format(filt))
                XOFF, YOFF = 0., 0.

            xpix = x_di+xshift
            y0 = y_di + (scan_direction * (t.scan_rate * subexposure.SCI.header['SAMPTIME'])) / 0.121 \
                      - (exposure.Primary.header['POSTARG2'] - di_ps2) / 0.121 # in pixels
            #print('Offset PA_V2',(exposure.Primary.header['POSTARG2'] - di_ps2) / 0.121)
            #print('Scan len {:.1f}'.format((scan_direction * (t.scan_rate * subexposure.SCI.header['SAMPTIME'])) / 0.121))
            #print('y0: {:.1f}'.format(y0))
            width0 = subexposure.SCI.header['DELTATIM']*t.scan_rate/0.121 # initial guess of width
            ystart, yend = disp.get_yscan(subexposure.SCI.data, x0=xpix, y0=y0, width0=width0, nsig=t.nysig, two_scans=t.two_scans)

            subexposure.xpix = xpix
            subexposure.ystart = ystart; subexposure.yend = yend
            subexposure.ypix = (subexposure.ystart+subexposure.yend)/2.

            subexposure.wave_grid = disp.dispersion_solution(x0=xpix, L=subexposure.SCI.data.shape[0], Dxoff=XOFF, Dxref=Dxref, ystart=ystart, yend=yend, DISP_COEFFS=DISP_COEFFS, TRACE_COEFFS=TRACE_COEFFS, wdpt_grid_y=t.grid_y, wdpt_grid_lam=t.grid_lam)
            
            if i == 0: 
                wave_ref = subexposure.wave_grid[0]
            # choose a reference wavelength from the 1st subexposure (can be any)
            
            subexposure.waves = wave_ref
            cut_image = subexposure.SCI.data[ystart:yend,int(xpix):int(xpix)+200]
            if subexposure.SCI.data.shape[1] < xpix + 200:
                subexposure.waves = subexposure.waves[:cut_image.shape[1]-200]
                subexposure.wave_grid = subexposure.wave_grid[:,:cut_image.shape[1]-200]

            interp_image, interp_mask = disp.interp_wave_grid(subexposure.waves, subexposure.wave_grid,
                                             subexposure.SCI.data[ystart:yend,int(xpix):int(xpix)+200],
                                             subexposure.mask[ystart:yend,int(xpix):int(xpix)+200], 
                                             tol=t.contam_thresh )
            subexposure.mask[ystart:yend,int(xpix):int(xpix)+200] = interp_mask

            # Flat field correction (requres wavelength solution for more than 0th order)
            if t.flat_field:
                nys = subexposure.yend-subexposure.ystart
                _waves = subexposure.waves.repeat(nys).reshape(-1, nys).T
                interp_image = interp_image*ff0[subexposure.ystart:subexposure.yend,int(xpix):int(xpix)+200]
                if subexposure.SCI.data.shape[1] < xpix + 200:
                    x1 = int(xpix)+cut_image.shape[1]
                else: 
                    x1 = int(xpix)+200
                interp_image, ff, ff_error = cal.flat_field_correct( _waves, interp_image,
                                                int(xpix), x1, subexposure.ystart, subexposure.yend,
                                                t.flat_file_g141)
                # ff here has been done already for 0th (wavelength indep) term in subexposure creation
                # so need to undo 0th order correction
                logger.info('Flat-field correction performed with full wavelength dependence')

            subexposure.SCI.data[subexposure.ystart:subexposure.yend,int(xpix):int(xpix)+200] = interp_image

            # Wavelength dependent flat field correction, some NaNs/inf creep in due to dead pixels in ff, change these to zeros
            bad_pixels = np.logical_or(np.isinf(subexposure.SCI.data),np.isnan(subexposure.SCI.data))
            subexposure.SCI.data[bad_pixels] = 0
            # Turns out this isnt enough, there are some very small values in ff that can cause spikes in flux
            # Hence just run a second round of CR removal on this image for bad pixels in flat-field
            logger.debug('\n')
            new_subs.append(subexposure)
        subexposures = new_subs


    # Local CR removal
    if t.cr_local:
        new_subs = []
        for subexposure in subexposures:
            '''
            # Mask everything outside of 1st order, since don't care about those cosmic rays
            if t.dispersion:
                x1 = int(xpix)
                x2 = int(xpix)+200 # 192-15 is the length of the first order BEAM
                mask_0 = np.zeros_like(subexposure.SCI.data)
                mask_0[subexposure.ystart:subexposure.yend,:x1+1] = 1.
                mask_0[subexposure.ystart:subexposure.yend,x2:] = 1.
                ignore_mask = np.logical_or(subexposure.DQ_mask, mask_0)
            else:
                ignore_mask = subexposure.DQ_mask
            '''    
            ignore_mask = subexposure.mask
            
            CR_clean, CR_mask = r.spatial_median_filter(subexposure.SCI.data.copy(), ignore_mask, tol=t.cr_tol, replace=t.cr_replace, debug=False, \
                                    sx=t.cr_x, sy=t.cr_y)
            n_crs = np.count_nonzero(CR_mask)

            subexposure.SCI.data = CR_clean
            #n_crs = np.count_nonzero(CR_mask[ypix-t.psf_h/2:ypix+t.psf_h/2,xpix-100:xpix+100])
            subexposure.n_crs = n_crs
            subexposure.CR_mask = CR_mask
            logger.info('Removed {} CRs pixels from subexposure {}'.format(n_crs,i+1))
            subexposure.mask = np.logical_or(subexposure.mask, CR_mask)

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
            view(all_CRs, title='Distribution of CRs over exposure', cbar=False, show=False, vmin=0, vmax=1, cmap='binary_r')
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


    # Plot subexposures
    if t.debug:
        if t.flat_field:
            mn_ff = np.nanmean(ff)
            view(ff, title='Flat-field: mean {:.4f}'.format(mn_ff), cbar=True, cmap='binary_r', vmin=0.9, vmax=1.1, show=False)
            save_fig()

        arrays_plot([sub.SCI.data for sub in subexposures], name='Subexp', cbar=False, size=2, \
                        tight_layout=False, vmin=0., vmax=100, show=False)
        if t.dispersion: # add in marker of where the spectrum is centered
            nrows = np.ceil(len(subexposures)/4.)
            for i, subexposure in enumerate(subexposures):
                #x, y = subexposure.xpix, subexposure.ypix
                p.subplot(nrows, 4, i+1)
                #p.plot(x,y,marker='o', mfc='None', color='w')
                p.plot(xpix,y_di,marker='x', mfc='None', color='w')
                p.plot([xpix,xpix],[subexposure.ystart,subexposure.yend], color='w')
        save_fig()

    exposure.subexposures = subexposures

    if t.pdf and t.debug:
        pdf.close()

    if t.save_dir:
        # Save reduced fits file (_red.fits format)
         logger.info('Saving reduced file to {}'.format(t.save_dir))
         data.write_reduced_fits(exposure.subexposures, exposure.Primary, t, dest_dir=t.save_dir)

    logger.info('Time taken: {}'.format(time.time()-t0))
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
            save_sub: bool, save the subexposure spectra separately

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
                'refine_xpix':False, 'object_ind':0, 'oe_debug':0, 'oe_pdf':None, 'save_sub':False}
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
        #reduced_exposure = data.Data_red(t.source_dir+t.system+'/'+reduced_exposure)

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
            if not hasattr(subexposure, 'waves'): subexposure.waves = np.arange(subexposure.SCI.data.shape[1])
            if xshift:
                xpix = subexposure.xpix + 100
                D, mask, bg, err = map(lambda image: r.box_cut(xpix, image, 200, horizontal=True, force_shape=False), \
                                [subexposure.SCI.data, subexposure.mask, bg, subexposure.ERR.data])
                logger.info('Spectrum located at pixel {} in spatial direction'.format(xpix))
            else:
                D, mask, bg, err = subexposure.SCI.data, subexposure.mask, bg, subexposure.ERR.data

            ypix = subexposure.ypix
            D, mask, bg, err = map(lambda image: r.box_cut(ypix, image, t.box_h, force_shape=False), [D, mask, bg, err])

        else:
            if n_sub == 0: logger.warning('No extraction box used')
            D = subexposure.SCI.data
            mask = subexposure.mask
            err = subexposure.ERR.da

        if not t.calc_var:
            if n_sub == 0: logger.info('Using ERR extension for variance estimate')
            V = np.square(err) # gain correction already done to _ima files
            t.v_0 = 0
            # Use errors to estimate variance per pixel
        else:
            if n_sub == 0: logger.info('Using flux for variance estimate')
            V = np.abs(D) + bg_err**2 + t.v_0 # gain correction already done to _ima files

        Ds.append(D)
        DQs.append(mask)

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
                oe_pdf_file = t.save_dir+'logs/{}_{}_fit.pdf'.format(rootname, n_sub)
                f.silentremove(oe_pdf_file)
            else: oe_pdf_file = None

            spec, specV, P, V = ea.extract_spectrum(D=D, S=S, V_0=t.v_0, Q=t.q, V=V, s_clip=t.s_clip, s_cosmic=t.s_cosmic, \
                                    func_type=t.func_type, method=t.method, debug=t.oe_debug, tol=t.fit_tol, M_DQ=M_DQ, M_CR=M_CR, \
                                    pdf_file=oe_pdf_file, step=t.step, order=t.order, skip_fit=t.skip_fit, bg=bg, k=t.k, logger=logger)
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
            if n_sub == 0: logger.info('Not using optimal extraction - results will be noisy')
            M = np.logical_not(mask)

            # Use a smoothed image for weights
            k = 5 # smooth along k pixels
            # Now do a median smooth along each column
            n_smoothloops = 4
            P = D.copy()
            for _ in range(n_smoothloops):
                smooth_spec = []
                for col in P.T:
                    col = ea.medfilt(col, k)
                    smooth_spec.append(col)
                P = np.vstack(smooth_spec).T
            P = P / np.sum(P, axis=0)

            V = ea.initial_variance_estimate(D=D, V_0=t.v_0, Q=t.q) + bg_err**2 # background calc affects variance, normally handled by opt_ext

            #spec, specV = np.nansum(D, axis=0), np.nansum(V, axis=0)/len(V)
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
            save_fig()

            size = 3
            nrows = np.ceil(len(good_subs)/4.)
            rcParams['figure.figsize'] = size*4, size*nrows/2
            for i in range(len(good_subs)):
                p.subplot(nrows, 4, i+1)
                image = Ds[i]
                M = Ms[i]
                image[np.logical_not(M)] = np.nan
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
    if t.shift_spectra:
        for spec in spectra:
            # Want to interpolate spectrum to a wavelength scale, not fit an offset
            #shift_y = np.interp(x, spec.x, spec.y)
            shift = r.spec_pix_shift(x, y, spec.x, spec.y, debug=False)
            shift_y = np.interp(x, x+shift, spec.y)
            interp_spectra.append(shift_y)
    else:
        interp_spectra = [spec.y for spec in spectra]

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

    logger.info('Time taken: {}'.format(time.time()-t0))
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

def create_orbit_cats_gauss(target='GJ-1214', source_dir='/home/jacob/hst_data/', gridsize=5, use_ima=False, nstars=2):
    '''
    Fit a gaussian to direct image to create visit catalogue.
    Use same format as SExtractor for catalogue files.
    '''
    reload(data)
    from lmfit import minimize, Parameters
    import astropy
    from astropy import modeling

    data_dir = source_dir + target + '/'
    if not os.path.exists(data_dir+'input_image.lis'):
        data.make_input_image_list(data_dir=data_dir)
    if not os.path.exists(data_dir+'visit_driz.lis'):
        data.make_driz_list(data_dir=data_dir)

    Gaussian2D = astropy.modeling.functional_models.Gaussian2D

    with open(data_dir+'visit_driz.lis', 'r') as driz_list:
        for line in driz_list:
            fname = line[:-1]
            dest_file = data_dir + fname.split('_')[0]+'_flt_1.cat'
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
