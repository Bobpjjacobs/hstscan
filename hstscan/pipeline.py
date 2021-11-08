from __future__ import print_function
import reduction as r
import my_fns as f
import data
import extraction_algorithm as ea
import calibration as cal
import shutil, logging, time, multiprocessing, sys
from lmfit import minimize, Parameters
from astropy import modeling
from my_fns import np, p, os
from multiprocessing.pool import Pool
from matplotlib.pyplot import rcParams
from matplotlib.ticker import StrMethodFormatter
from scipy.optimize import leastsq, curve_fit
from matplotlib.backends.backend_pdf import PdfPages
from Telescope_characteristics import HST

import dispersion as disp
import astropy.io.fits as pyfits

reload(data)
reload(cal)
reload(ea)
reload(r)
reload(disp)
view = data.view_frame_image
plot = data.plot_data



def add_handlers(logger, log_file, warnings_file, level):
    """
        Set up logging file to include handlers for info and warnings.
        """
    # Add handlers to a logging file
    fh = logging.FileHandler(log_file, mode='w')  # file handling, remove exisiting log
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
    """Default logging to print to terminal"""

    def info(self, x): print(x)

    def warning(self, x): print(x)

    def debug(self, x): print(x)


def arrays_plot(arrays, name='Read', tight_layout=True, size=3, height=1, show=True, titles=[], **kwargs):
    """Neatly plot a list of arrays, using data.view_frame_image"""
    view = data.view_frame_image
    nrows = np.ceil(len(arrays) / 4.)
    if len(arrays) > 1: rcParams['figure.figsize'] = size * 4, size * nrows * height
    for i, array in enumerate(arrays):
        if len(arrays) > 1: p.subplot(nrows, 4, i + 1)
        if len(titles)> 0:
            view(array, show=False, title=titles[i], **kwargs)
        else:
            view(array, show=False, **kwargs)
        if not name is None: p.title('{} {}'.format(name, i))
        p.axis('off')
    if tight_layout: p.tight_layout()
    if show: p.show()
    rcParams['figure.figsize'] = 10, 5


def compute_exposure_shifts(visits, source_dir, save_dir=None, verbose=True, fname='exposure_shifts.lis', save=False):
    """
    Compute shift of raw spectrum between exposures in the same visit

    Shifts are defined as the difference between the position of the star on the exposure
    and the position of the star on the reference exposure.
    So the -ve of the output of r.spec_pix_shift.

    Save outputs to file for use during pipeline run
    """
    all_rootnames, all_shifts = [], []
    for visit in visits:
        print('Visit: {}'.format(visit))
        with open(source_dir + visit, 'r') as g:
            lines = g.readlines()
        rootnames = [line.split('\t')[0] for line in lines
                     if (not line.strip() == '') and not line.startswith('#')
                     and not line.split('\t')[1].startswith('F')]

        # Reference exposure
        exp1 = data.load('{}{}_ima.fits'.format(source_dir, rootnames[0]), hst_file=t.hst_eph_file,
                         tai_file=t.tai_utc_file)
        fl1 = np.sum(exp1.reads[0].SCI.data, axis=0) / np.sum(exp1.reads[0].SCI.data)
        x = np.arange(len(fl1))

        # Compute shifts for other exposures
        all_rootnames.append(rootnames[0])
        all_shifts.append(0)
        for rn in rootnames[1:]:
            exp2 = data.load('{}{}_ima.fits'.format(source_dir, rn), hst_file=t.hst_eph_file, tai_file=t.tai_utc_file)
            fl2 = np.sum(exp2.reads[0].SCI.data, axis=0) / np.sum(exp2.reads[0].SCI.data)
            shift, err = r.spec_pix_shift(x, fl1, x, fl2)

            all_rootnames.append(rn)
            all_shifts.append(-shift)

    if save_dir is None: save_dir = source_dir
    with open(save_dir + fname, 'w') as g:
        print('Saving to ', save_dir + fname)
        for rn, sh in zip(all_rootnames, all_shifts):
            line = '{}\t{:.3f}\n'.format(rn, sh)
            g.write(line)
    return np.array(all_shifts)


def reduce_exposure(exposure, conf_file=None, tel=HST(), **kwargs):
    """
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
            write (bool):       whether to write the file or not

            remove_subexps_list (list of integers): subexposure numbers that need to be removed from the total spectrum.
                                                    This may be because this subexposure is unreliable due to e.g. a
                                                    satellite crossing event.
                                                    This subexposure will STILL appear in the file with all subexposures
                                                    This does not affect the timing of the exposure

            use_2nd_order (bool): Whether to use the second order of the spectrum to fit the position of the spectrum
                                   on the detector.

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
            disp_coef (str): Take the dispersion coefficients from: 'default', 'wilkins' or 'custom'

            CR_local (bool): perform local CR removal (only method left)
            CR_plot (bool): debug plots
            CR_tolx/CR_toly (float): tolerance for local CR hits in x/y-direction in sigma
            CR_thresh (float): threshold, in electrons, for CR hit detection
            CR_x, CR_y (ints): number of pixels over which to check deviation
            CR_replace: change the value that replaces a CR pixel (local mean, median, NaN...)
            cr_persistence: (bool) Whether to mark CRs as DQ pixels in subsequent subexposures.
            cr_deficiency: (bool) Whether to mask pixels that are CR_tolx/CR_toly times BELOW the median. These are masked
                                   as DQ pixels

            ext_wshift (int): number of pixels with which to shift the wavelength solution manually.
            abs_wshift (bool): whether to use an absolute wavelength solution estimated by a fit to a stellar model.
            zeroth_ord_wshift (bool): whether to estimate the absolute wavelength solution with the 0th order spectrum.
            wshift_from_postarg (bool): Whether to estimate the absolute wavelength solution from the postarg argument
                                         of the exposure.
            Please pick only one of the above four.
            wshift_to_ref (bool): whether to shift the wavelength solution to the one from a reference image.
            stel_spec_gauss_kern_sig (float): A multiplication number used to estimate over how many pixels one should
                                              smear out the PHOENIX spectrum in the calculation of the absolute w_shift.

            postarg_yguess (bool): Whether to guess y0 from the difference in postarg between direct image and exposure
            yguess_offset (int): Number of pixels with which to offset this guess (because the default is often wrong)
            default_scan ('r'/'f'): What is the default scan direction.

            dispersion (bool): perform wavelength calculation and corrections
            XOFF_file (str): file containing x-shifts between exposures (pre-calculated)
            flat_field (bool): whether to perform flat_field correction, wavelength dep if dispersion==True
            ff_min (float): minimum value for flat-field, values below this are set to 1
            nysig (int): number of spectrum gaussian widths to calculate wavelength for on image
            tsiaras (bool): use wavlength dependent photon trajectory method (Tsiaras+18)
            grid_y, grid_lam (ints): wavelength dependent photon trajectory grid resolution
            interp_kind (str): type of interpolation if using basic method (linear, quadratic...)
            flat_file_g141 (str): config file for flat-field
            conf_file_g141 (str): config file for dispersion
            hst_eph_file (str): file with the HST ephemeris data.
            tai_utc_file (str): config file for international atomic time and utc conversion
            contam_thresh: contamination threshold of bad pixels for interpolation
    """
    t0 = time.time()  # time reduction run

    # Store all the possible external configs in a dict 't', default values defined in 'toggles'
    toggles = {'system':'WASP-18', 'source_dir':'/home/jacob/hst_data/WASP-18/', 'save_dir':None, 'write':False,
               'debug':False, 'pdf':False, 'logger':True,
               'scanned':True, 'scan_rate':None, 'units':True, 'nlincorr':False, 'read_noise':20, 'remove_scan':False,
               'bjd':True, 'hst_eph_file':'None',
               'use_2nd_order':False,
               'dq_replace':None, 'dq_mean_width':1, 'dq_flags':[4, 32], 'skip_start':1, 'skip_end':0,
               'bg':True, 'bg_plot':False, 'bg_area':True, 'bg_x':0, 'bg_y':0, 'bg_h':50, 'bg_w':50,
               'psf_h':130, 'mask_h':40, 'psf_w':220, 'n_masks':3, 'neg_masks':0, 'disp_coef': 'wilkins',
               'postarg_yguess':True, 'yguess_offset_f': -25, 'yguess_offset_r': 0, 'default_scan':'f',
               'cr_local':True, 'cr_tolx':5, 'cr_toly':10, 'cr_replace':'median', 'cr_plot':False,
               'cr_x':5, 'cr_y':5, 'cr_thresh':50., 'cr_mask_dq':True,
               'cr_master':False, 'cr_mname':None, 'cr_persistence': True, 'cr_deficiency': True,
               'dispersion':True, 'ref_exp':None, 'ref_wv0':0.9, 'ref_wv1':1.92, 'x':True,
               'wshift_from_postarg':False,
               #'tsiaras':True, 'peak':False, 'calc_abs_xshift': True, 'xshift_ext':0.,
               'tsiaras':True, 'peak':False, 'ext_wshift':0., 'zeroth_ord_wshift':False,
               'abs_wshift':True, 'stel_spec_gauss_kern_sig': 1., 'wshift_to_ref':True,
               #'exp_drift':False, 'drift_max':2., 'drift_width':4, 'drift_rowtol':1.1,
               'flat_field':True, 'ff_min':0.5,
               'nysig':5, 'grid_y':20, 'grid_lam':20, 'two_scans':False, 'interp_kind':'linear',
               'nlin_file':None, 'flat_file_g141':'None', 'conf_file_g141':'None', 'trans_file_g141':'None',
               'trans_file2_g141':'None', 'tai_utc_file':'None',
               'stellar_spectrum':'None', 'stellar_wavelengths': 'None',
               'contam_thresh':0.01, 'hard_e_limit':1e10,
               'object_ind':0, 'remove_subexps_list':[]}
    # Read in conf_file and kwargs to update default toggle values
    # priority is kwargs > conf_file > default
    if conf_file:
        conf_kwargs = data.read_conf_file(conf_file)
        conf_kwargs.update(**kwargs)
    else:
        conf_kwargs = kwargs
    t = f.bunch_kwargs(kwargs=conf_kwargs, toggles=toggles, name='reduction', verbose=True)

    wshift_error = ("Can only shift the wavelength solution in one way. Please pick one of 'ext_wshift', " +
                    "'abs_wshift', 'zeroth_ord_wshift', 'wshift_from_postarg' or none of those four.")
    assert bool(t.ext_wshift) + t.abs_wshift + t.zeroth_ord_wshift + t.wshift_from_postarg < 2, wshift_error
    if t.wshift_to_ref:
        assert t.ref_exp, "Please choose a reference exposure or disable the wshift_to_ref toggle."

    # Open up an exposure if you input a filename
    if type(exposure) == str:
        exposure = data.load(t.source_dir + exposure, hst_file=t.hst_eph_file, tai_file=t.tai_utc_file, bjd=t.bjd)

    #And a reference exposure
    assert t.ref_exp is not None, "Please enter the name of a reference exposure in the 'ref_exp' parameter."
    if t.ref_exp != exposure.rootname:
        try:
           ref_exp = data.load(t.save_dir + t.ref_exp + '_red.fits')
        except IOError:
            raise IOError("Make sure that file {} exists. This isn't your reference exposure. Please first reduce your "
                          "reference exposure first. Currently your reference exposure is {}".format(t.save_dir + t.ref_exp + '_red.fits', t.ref_exp))


    # Set up logging for errors and info
    logger = setup_logger('Reduction', exposure.rootname, t)
    logger.info('Data reduction pipeline performed on exposure {}'.format(exposure.filename))
    logger.info('For configuration, see pipeline conf file: {}'.format(conf_file))

    # Set up pdf file for debug plots
    if t.pdf and t.debug:
        pdf_file = t.save_dir + 'logs/' + exposure.rootname + '_red.pdf'
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
            try:
                int_time = read.TIME.header['PIXVALUE']
            except KeyError:
                int_time = np.median(read.TIME.data)
            read.SCI.data = read.SCI.data * int_time
            read.ERR.data = read.ERR.data * int_time
            read.SCI.header['BUNIT'] = 'ELECTRONS'
            read.ERR.header['BUNIT'] = 'ELECTRONS'

        masks.append(mask)
        reads.append(read)

    if t.debug:
        # Plot flagged data quality pixels
        if t.dq_flags is not None:
            n_rows = np.ceil(len(t.dq_flags) / 2.)
            if len(t.dq_flags) % 2 == 0: n_rows += 1  # for the combined DQ mask

            p.figure(figsize=(6, 3 * n_rows))
            p.subplot(n_rows, 2, 1)
            view(masks[0], title='All DQ pixels ({})'.format(t.dq_flags), cmap='binary_r', cbar=False, show=False,
                 xlabel='', ylabel='')
            for k, flag in enumerate(t.dq_flags):
                _mask = reads[0].DQ.data / flag % 2 == 1
                p.subplot(n_rows, 2, k + 2)
                title = '{}: {}\n{}'.format(flag, tel.dq_info[flag][0], tel.dq_info[flag][1])
                view(_mask, title=title, cmap='binary_r', cbar=False, show=False, xlabel='', ylabel='')
            p.tight_layout()
            save_fig()
        # Plot all reads
        vmin = 0.;
        # vmax = np.max(reads[0].SCI.data)
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
        subexposure.DQs = DQ_mask
        subexposure.DQ_mask = DQ_mask
        subexposure.mask = DQ_mask
        subexposures.append(subexposure)
    else:
        for i in range(len(exposure.reads))[t.skip_end:-t.skip_start - 1]:
            read2, read1 = exposure.reads[i], exposure.reads[i + 1]
            DQ_mask = np.logical_or(masks[i], masks[i + 1])
            # Form subexposure
            subexposure = r.create_sub_exposure(read1, read2, read_noise=t.read_noise, nlincorr=t.nlincorr,
                                                nlinfile=t.nlin_file)
            subexposure.DQs = DQ_mask
            subexposure.DQ_mask = DQ_mask
            subexposure.mask = DQ_mask  # track total mask (DQ + CR hits)
            subexposures.append(subexposure)

    # Background removal
    cr_previous = np.zeros_like(subexposures[0].SCI.data, dtype=bool)
    bg_cr_masks = []
    for i, subexp in enumerate(subexposures):
        subexposures[i], bg_mask, bg_cr_mask = r.background_removal(i, subexp, cr_previous, t, logger)
        cr_previous = bg_cr_mask
        bg_cr_masks.append(bg_cr_mask)

    # Show background area
    if t.debug and t.bg:
        if t.bg_area:
            view(exposure.reads[0].SCI.data, show=False, vmin=0, vmax=8 * exposure.Primary.header['EXPTIME'],
                 title='Background Area', bg_mask=bg_mask, cmap='viridis', origin='lower')
            save_fig()

            # Plot of areas used for bg in each subexposure
            bg_plot_titles = ['Bg area at time {}'.format(subexposure.SCI.header['SAMPTIME']) for subexposure in subexposures]
            arrays_plot([np.reshape(sub.SCI.data[bg_mask.astype(bool)], (t.bg_h, t.bg_w)) for sub in subexposures],
                        name=None, cbar=False, size=4, height=t.bg_h / float(t.bg_w), tight_layout=False, vmin=0.,
                        titles=bg_plot_titles, vmax= exposure.Primary.header['EXPTIME'], show=False)
            p.suptitle('Bg Area for each subexposure')
            save_fig()

            # Plot of CR masks for backgrounds.
            if t.bg_plot:
                bg_plot_titles = ['Bg mask at time {}'.format(subexposure.SCI.header['SAMPTIME']) for subexposure in subexposures]
                arrays_plot([np.reshape(bg_cr_masks[i], (t.bg_h, t.bg_w)) for i,sub in enumerate(subexposures)],
                        name=None, cbar=False, size=4, height=t.bg_h / float(t.bg_w), tight_layout=False, vmin=0.,
                        titles=bg_plot_titles, show=False)
                p.suptitle('Bg cr masks for each subexposure')
                save_fig()



        # Plot of backgrounds over time
        bgs = [np.median(subexposure.bg) for subexposure in subexposures]
        ts = [subexposure.SCI.header['SAMPTIME'] for subexposure in subexposures]
        plot(ts, bgs, title='Background estimates', xlabel='Time (seconds)', ylabel='Background (electrons)',
             marker='o', ls='None', color='g', ms=10, mec='k')
        p.tight_layout()
        save_fig()


    # Calculate dispersion solution
    if t.dispersion:
        # First get relevant config values
        BEAM, DISP_COEFFS, TRACE_COEFFS = disp.get_conf_coeffs(WFC_conf_file=t.conf_file_g141)
        scan_direction, t.scan_rate = data.read_scan_direction(exposure, t, logger)


        # Find direct image position from catalogue file (pre-computed)
        catalogue, di_name = data.find_catalogue(exposure.rootname, logger, data_dir=t.source_dir)
        logger.info('Catalogue used: {}\nDirect Image name: {}'.format(catalogue, di_name))
        x_di, y_di, di_ps1, di_ps2, di_pav3, direct_image = data.find_star_in_catalogue(catalogue, di_name, t, logger)
        Dxref = (exposure.Primary.header['POSTARG1'] - di_ps1) / tel.xscale

        # Different filters have small inherent direct image offsets
        filt = direct_image.Primary.header['FILTER']
        XOFF, YOFF = tel.get_wfc3_filter_offs(filt)
        if XOFF is None:
            logger.warning('Filter {} offset not known.'.format(filt))
            XOFF, YOFF = 0., 0.

        if t.ref_exp == exposure.rootname or (t.ref_exp != exposure.rootname and not t.wshift_to_ref):
            if t.ext_wshift != 0.:
                # Apply external shift (if any) to direct image (pre-computed)
                shift_in_x = t.ext_wshift
                exposure.Primary.header['SHIFTEXT'] = int(shift_in_x)
                logger.debug('External xshift of direct image applied, of {} pix'.format(shift_in_x))
            elif t.abs_wshift:
                #This absolute wavelength calibration does not need to be perfect.
                #After all, we need it to be correct up to 1 pixel.
                shift_in_x = calc_abs_xshift(direct_image, exposure, subexposures[0], tel, t, scan_direction, x_di,
                                             y_di, catalogue, Dxref, DISP_COEFFS, TRACE_COEFFS, conf_file, plot=True)
                save_fig()
                logger.debug('Applied an xshift of {} pix from a comparison to a stellar spectrum'.format(shift_in_x))
            elif t.zeroth_ord_wshift:
                print ("nothing going on here yet")
                shift_in_x = 0
            elif t.wshift_from_postarg:
                shift_in_x = Dxref
                logger.info("Applying an xshift of {} because of the POSTARG argument".format(shift_in_X))
            else:
                #http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2015-10.pdf
                #http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2009-17.pdf
                shift_in_x = 0
                logger.debug('No xshift was applied')
        elif t.ref_exp != exposure.rootname and t.wshift_to_ref:
            # Temporarily use the xshift of the reference exposure. This will be updated after the first flat-field
            # estimate.
            shift_in_x = ref_exp.Primary.header['XSHIFT']


        #shift_in_x = shift_in_x + 10
        xpix = int(x_di + shift_in_x) #Round off to nearest integer
        exposure.Primary.header['XSHIFT'] = shift_in_x
        exposure.Primary.header['XPIX'] = xpix
        logger.info("Initial xpix is {} and shift_in_x: {}".format(xpix, shift_in_x))

        # Now do a partial dispersion solution to allow us to calculate accurate exposure shifts
        for step in ['partial', 'final']:
            if step == 'final':
                if t.ref_exp != exposure.rootname and t.wshift_to_ref:
                    # Cross correlate first subexposure to the first subexposure of the (reduced) reference image:
                    s2 = 0
                    if np.sign(ref_exp.Primary.header['POSTARG2']) == np.sign(exposure.Primary.header['POSTARG2']):
                        #Both ref and this exposure are forward/reverse
                        s = 0
                    else:
                        #Ref and this exposure have a different scan direction
                        s = -1
                    ref_tot = np.nansum(ref_exp.subexposures[s].SCI.data[new_subs[s2].ystart:new_subs[s2].yend,
                                        xpix:xpix + 200], axis=0)
                    ref_tot = (ref_tot / np.nansum(ref_tot))

                    spec = np.nansum(new_subs[s2].interp_image, axis=0)
                    spec = spec / np.nansum(spec)

                    # Calculate shift
                    shift_in_x, err = r.spec_pix_shift(np.arange(len(ref_tot)), spec, np.arange(len(ref_tot)), ref_tot,
                                                       fitpeak=t.peak)

                    xpix += int(shift_in_x)  # Round off to nearest integer
                    exposure.Primary.header['XSHIFT'] += shift_in_x
                    exposure.Primary.header['XPIX'] = xpix
                else:
                    s = -1#0

                if t.debug and shift_in_x != 0 and t.ref_exp!= exposure.rootname:
                    tot_image = np.nansum([sub.SCI.data for sub in new_subs], axis=0)
                    view(tot_image[:,int(xpix):int(xpix) + 200], show=False, title='Tot', vmin=0,
                         vmax=8 * exposure.Primary.header['EXPTIME'])
                    save_fig()

                    tot = np.nansum(tot_image, axis=0)[int(xpix):int(xpix) + 200]
                    tot = tot / np.nansum(tot)
                    x = np.arange(len(tot))
                    shift_to_ref = exposure.Primary.header['XSHIFT'] - ref_exp.Primary.header['XSHIFT']
                    new_tot = np.interp(x, x - shift_to_ref, tot)
                    ref_image = np.nansum([sub.SCI.data for sub in ref_exp.subexposures], axis=0)
                    ref_tot = np.nansum(ref_image, axis=0)[int(xpix):int(xpix) + 200]
                    ref_tot = ref_tot / np.nansum(ref_tot)
                    if t.wshift_to_ref:
                        data.plot_data(x=[x, x, x], y=[tot, new_tot, ref_tot], label=['Exp', 'Shifted', 'Ref'],
                                       title='Shifted exposure to reference exposure', xlabel='pixels')
                    else:
                        data.plot_data(x=[x, x, x], y=[tot, new_tot, ref_tot], label=['Exp', 'Shifted', 'Ref'],
                                       title='Compare this exposure to the reference exposure', xlabel='pixels')
                    p.legend()
                    save_fig()
                    print ("shifttoref", shift_to_ref)

                logger.info("The final xpix is {}".format(xpix))

            # Now compute wavelength solution given direct image position
            tot_image = 0.
            new_subs = []
            for i, subexposure in enumerate(subexposures):
                image = subexposure.SCI.data.copy()
                subexp_time = subexposure.SCI.header['SAMPTIME']

                # Guess position of y from DI
                y0, width0 = disp.get_y0(subexposure, y_di, [t.yguess_offset_f, t.yguess_offset_r], di_ps2,
                                         exposure.Primary.header['POSTARG2'], i, exposure.Primary.header['EXPTIME'],
                                         subexp_time, scan_direction, t, tel, logger)
                subexposure.SCI.header['EXPTIME'] = exposure.Primary.header['EXPTIME'] / len(subexposures)
                # Fit for y scan height and position given guess
                ystart, ymid, yend = disp.get_yscan(image, x0=xpix, y0=y0, width0=width0, nsig=t.nysig,
                                                    two_scans=t.two_scans, debug=True)

                subexposure.xpix = xpix
                subexposure.ystart = ystart;
                subexposure.yend = yend;
                subexposure.ypix = ymid

                # Calculate wavelength solution
                if t.tsiaras:
                    subexposure.wave_grid = disp.dispersion_solution(x0=xpix, L=image.shape[0], Dxoff=XOFF, Dxref=Dxref,
                                                                     ystart=ystart, yend=yend, DISP_COEFFS=DISP_COEFFS,
                                                                     TRACE_COEFFS=TRACE_COEFFS, wdpt_grid_y=t.grid_y,
                                                                     wdpt_grid_lam=t.grid_lam,
                                                                     WFC_conf_file=conf_file)

                    # Define wavelength grid to interpolate to: 0.9-1.92, 200
                    wave_ref = np.linspace(t.ref_wv0, t.ref_wv1, 200)  # subexposure.wave_grid[0]
                    # interpolate all rows to this row
                else:
                    # Regular wavelength correction
                    L = subexposure.SCI.data.shape[0]
                    wave_grid, trace = cal.disp_poly(t.conf_file_g141, catalogue, subexp_time, t.scan_rate,
                                                     scan_direction, order=1, x_len=L, y_len=L, XOFF=XOFF, YOFF=YOFF,
                                                     data_dir=t.source_dir, debug=False, x=subexposure.xpix,
                                                     y=subexposure.ypix, disp_coef=t.disp_coef)
                    subexposure.wave_grid = wave_grid[subexposure.ystart:subexposure.yend, int(xpix):int(xpix) + 200]
                    wave_ref = subexposure.wave_grid[0]

                subexposure.waves = wave_ref
                if step == 'partial':
                    cut_image = image[ystart:yend, int(xpix):int(xpix) + 200].copy()  # cutout of spectral area
                    subexposure.original_image = cut_image
                else:
                    cut_image = subexposure.original_image

                # Flat field correction (requres wavelength solution for more than 0th order)
                # Need to do before interpolating to a reference row
                if t.flat_field and (step == 'final' or t.ref_exp):
                    #nys = subexposure.yend - subexposure.ystart
                    _waves = subexposure.wave_grid  # subexposure.waves.repeat(nys).reshape(-1, nys).T
                    if subexposure.SCI.data.shape[1] < xpix + 200:
                        x1 = int(xpix) + cut_image.shape[1]
                    else:
                        x1 = int(xpix) + 200
                    L = subexposure.SCI.data.shape[0]
                    dL = (1014 - L) / 2
                    cut_image, ff, ff_error = cal.flat_field_correct(_waves, cut_image,
                                                                     int(xpix) + dL, x1 + dL, subexposure.ystart + dL,
                                                                     subexposure.yend + dL,
                                                                     t.flat_file_g141, ff_min=t.ff_min)
                    subexposure.ff = ff
                    subexposure.ff_error = ff_error
                    # logger.info('Flat-field correction performed with full wavelength dependence')
                    subexposure.cut_image = cut_image  # non-ff image

                cut_mask = subexposure.mask[ystart:yend, int(xpix):int(xpix) + 200]
                subexposure.cut_mask = cut_mask.copy()

                # New interpolation, area under pixel matches pixel flux
                if t.tsiaras:
                    interp_image, interp_mask = disp.interp_wave_grid_sane(subexposure.waves, subexposure.wave_grid,
                                                                           cut_image, cut_mask, tol=t.contam_thresh)
                else:
                    interp_image, interp_mask = cut_image, cut_mask


                subexposure.interp_image = interp_image
                subexposure.interp_mask = interp_mask

                subexposure.pre_mask = subexposure.mask.copy()
                subexposure.mask[ystart:yend, int(xpix):int(xpix) + 200] = interp_mask
                subexposure.interp_image = interp_image
                if step == 'partial':
                    image[subexposure.ystart:subexposure.yend, int(xpix):int(xpix) + 200] = interp_image
                    tot_image += image

                if step == 'final':
                    subexposure.SCI.data[subexposure.ystart:subexposure.yend, int(xpix):int(xpix) + 200] = interp_image

                    # Wavelength dependent flat field correction, some NaNs/inf creep in due to dead pixels in ff, change these to zeros
                    bad_pixels = np.logical_or(np.isinf(subexposure.SCI.data), np.isnan(subexposure.SCI.data))
                    subexposure.mask = np.logical_or(bad_pixels, subexposure.mask)
                new_subs.append(subexposure)
        subexposures = new_subs

    else:
        # just do the wavelength-indep flat-field correction
        if t.flat_field:
            for i, subexposure in enumerate(subexposures):
                # ff size is 1014x1014
                L = subexposure.SCI.data.shape[0]
                dL = (1014 - L) / 2
                if i == 0:
                    # only load ff file once
                    subexposure.SCI.data, ff0, ff0_error = cal.flat_field_correct(0, subexposure.SCI.data,
                                                                                  x0=dL, x1=L + dL, ystart=dL,
                                                                                  yend=L + dL,
                                                                                  flat_file=t.flat_file_g141,
                                                                                  wave_dep=False, ff_min=t.ff_min)
                else:
                    subexposure.SCI.data /= ff0
                bad_pixels = np.logical_or(np.isinf(subexposure.SCI.data), np.isnan(subexposure.SCI.data))
                subexposure.SCI.data[bad_pixels] = 0
            logger.info('0th order flat-field correction performed')

    # Plot subexposures
    if t.debug:
        if t.flat_field:
            if not t.dispersion: ff = ff0
            mn_ff = np.nanmean(ff)
            arrays_plot([sub.ff for sub in subexposures], name=None, cbar=False, size=4, height=0.5,
                        tight_layout=False, vmin=0.9, vmax=1.1, show=False)
            p.suptitle('Flat-field for each exposure')
            save_fig()
            if t.dispersion and False:


                #int(xpix) + dL, x1 + dL, subexposure.ystart + dL,
                #subexposure.yend + dL,
                _, ff0, ff0_error = cal.flat_field_correct(0, subexposures[s].SCI.data[subexposures[s].ystart:subexposures[s].yend, int(xpix):int(xpix) + 200],
                                                           int(xpix) + dL, x1 + dL, subexposures[s].ystart + dL,
                                                           subexposures[s].yend + dL,
                                                           t.flat_file_g141, wave_dep=False,
                                                           ff_min=t.ff_min)
                #view(ff0, title='Zeroth flat-field: mean {:.4f}'.format(np.nanmean(ff0)), cbar=True, cmap='binary_r',
                #     vmin=0.9, vmax=1.1, show=False)
                fig, (ax1, ax2) = p.subplots(2, 1, sharex=True, figsize=(10,10))
                fig.subplots_adjust(hspace=0)
                ax1.imshow(ff0, cmap='binary_r', vmin=0.9, vmax=1.1, aspect="auto")
                ax1.set_title('Flatfield in upper panel, uncorrected data in bottom panel. See how flat-field stripes may coincide with errors in data.')
                ax1.set_ylabel('pixels')

                #ax2.plot(np.arange(200), np.sum(subexposures[s].SCI.data[0:L, int(xpix): x1], axis=0))
                ax2.plot(np.arange(200), np.sum(subexposures[s].SCI.data[subexposures[s].ystart:subexposures[s].yend, int(xpix):int(xpix) + 200], axis=0))
                ax2.set_xlabel('Wavelength (microns)')
                ax2.set_ylabel('Total flux level')
                labels = [wave_ref[0], wave_ref[25], wave_ref[50],wave_ref[75], wave_ref[100], wave_ref[125],
                          wave_ref[150], wave_ref[175], wave_ref[199]]
                p.xticks([0, 25, 50, 75, 100, 125, 150, 175, 199], [str(round(float(label), 2)) for label in labels])
                p.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places

                save_fig()

            # Show effect of flat-field for first sub
            for expnr in range(len(subexposures)):#[0,1,2]:
                #print (L, xpix, dL, x1)
                #print(int(xpix) + dL, x1 + dL, dL, L + dL)
                view(subexposures[expnr].SCI.data[subexposures[expnr].ystart:subexposures[expnr].yend,
                                                  int(xpix):int(xpix) + 200], title='Image with flatfield applied')

                _, ff0, ff0_error = cal.flat_field_correct(0, subexposures[expnr].SCI.data[subexposures[expnr].ystart:subexposures[expnr].yend, int(xpix):int(xpix) + 200],
                                                           int(xpix) + dL, x1 + dL , subexposures[expnr].ystart + dL,
                                                           subexposures[expnr].yend + dL,
                                                           t.flat_file_g141, wave_dep=False,
                                                           ff_min=t.ff_min)
                #view(ff0, title='Zeroth flat-field: mean {:.4f}'.format(np.nanmean(ff0)), cbar=True, cmap='binary_r',
                #     vmin=0.9, vmax=1.1, show=False)
                fig, (ax1, ax2) = p.subplots(2, 1, sharex=True, figsize=(10,10))
                fig.subplots_adjust(hspace=0)
                ax1.imshow(ff0, cmap='binary_r', vmin=0.9, vmax=1.1, aspect="auto")
                ax1.set_title('Flatfield in upper panel, uncorrected data in bottom panel. See how flat-field stripes may coincide with errors in data.')
                ax1.set_ylabel('pixels')

                #ax2.plot(np.arange(200), np.sum(subexposures[s].SCI.data[0:L, int(xpix): x1], axis=0))
                ax2.plot(np.arange(200), np.sum(subexposures[expnr].SCI.data[subexposures[expnr].ystart:subexposures[expnr].yend, int(xpix):int(xpix) + 200], axis=0))
                ax2.set_xlabel('Wavelength (microns)')
                ax2.set_ylabel('Total flux level')
                labels = [wave_ref[0], wave_ref[25], wave_ref[50],wave_ref[75], wave_ref[100], wave_ref[125],
                          wave_ref[150], wave_ref[175], wave_ref[199]]
                p.xticks([0, 25, 50, 75, 100, 125, 150, 175, 199], [str(round(float(label), 2)) for label in labels])
                p.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places

                save_fig()


                sub = subexposures[expnr]
                pre_image = sub.original_image
                ff = sub.ff
                fig = p.figure(figsize=(8, 10))
                p.subplot(211)
                p.title('Column sum: subexposure {}'.format(expnr))
                p.plot(np.sum(pre_image, axis=0), label='Pre-FF')
                p.plot(np.sum(pre_image / ff, axis=0), label='Post-FF')
                p.plot([92,92], [0, max(np.sum(pre_image, axis=0))], alpha=0.5)
                p.plot([134,134], [0, max(np.sum(pre_image, axis=0))], alpha=0.5)
                # p.plot(np.sum(pre_image/np.random.normal(1,0.01,ff.shape), axis=0), label='Random FF')
                p.legend(fontsize=14)

                fig.subplots_adjust(hspace=0)

                p.subplot(212)
                p.plot(np.arange(ff.shape[1]), np.sum(ff, axis=0))
                p.plot([92, 92], [min(np.sum(ff, axis=0)), max(np.sum(ff, axis=0))], alpha=0.5)
                p.plot([134, 134], [min(np.sum(ff, axis=0)), max(np.sum(ff, axis=0))], alpha=0.5)
                p.xlabel('Flatfield pixel')
                p.ylabel('Flatfield value')
                save_fig()


        arrays_plot([sub.SCI.data for sub in subexposures], name='Subexp', cbar=False, size=2, \
                    tight_layout=False, vmin=0., vmax=100, show=False)
        if t.dispersion and len(subexposures) > 1:  # add in marker of where the spectrum is centered
            nrows = np.ceil(len(subexposures) / 4.)
            for i, subexposure in enumerate(subexposures):
                p.subplot(nrows, 4, i + 1)
                p.plot(xpix, y_di, marker='x', mfc='None', color='w')
                p.plot([xpix, xpix], [subexposure.ystart, subexposure.yend], color='w')
        save_fig()
        if t.dispersion:
            view(subexposures[0].wave_grid, show=False, title='Wavelength grid for subexposure 0')
            save_fig()

    # Local CR removal
    if t.cr_local:
        new_subs = [];
        cr_values = []
        cr_previous = np.zeros_like(subexposures[0].SCI.data, dtype=bool)
        for i, subexposure in enumerate(subexposures):
            if t.cr_persistence:
                subexposure.DQ_mask = np.logical_or(subexposure.DQ_mask, cr_previous)
                subexposure.mask = np.logical_or(subexposure.mask, cr_previous) #propagate CRs from previous subexposures as DQ flagged pixels.
            ignore_mask = subexposure.mask  # dont flag already masked pixels
            CR_clean, CR_mask, CR_info = r.spatial_median_filter(subexposure.SCI.data.copy(), ignore_mask,
                                                                 tolx=t.cr_tolx, toly=t.cr_toly,
                                                                 replace=t.cr_replace, \
                                                                 debug=False, sx=t.cr_x, sy=t.cr_y, thresh=t.cr_thresh,
                                                                 mask_dq=not t.cr_mask_dq,
                                                                 hard_e_limit = t.hard_e_limit)
            if t.cr_persistence:
                cr_previous = CR_mask#np.logical_or(cr_previous, CR_mask)
            n_crs = np.count_nonzero(CR_mask)
            cr_values.append(CR_info['cr_vals'])

            if t.cr_deficiency:
                Low_clean, Low_mask, Low_info = r.spatial_median_filter(CR_clean, ignore_mask,
                                                                 tolx=t.cr_tolx, toly=t.cr_toly,
                                                                 replace=t.cr_replace, \
                                                                 debug=False, sx=t.cr_x, sy=t.cr_y, thresh=t.cr_thresh,
                                                                 mask_dq=not t.cr_mask_dq,
                                                                 hard_e_limit = t.hard_e_limit, Low=True) #Search for pixels that have a flux *deficiency*
                subexposure.SCI.data = Low_clean
                subexposure.DQ_mask = np.logical_or(subexposure.DQ_mask, Low_mask)
                subexposure.mask = np.logical_or(subexposure.mask, Low_mask) #propagate CRs from previous subexposures as DQ flagged pixels.
                print("nr_CRs, Lows", n_crs, np.count_nonzero(Low_mask))
            else:
                print("nr_CRs", n_crs)
                subexposure.SCI.data = CR_clean
            subexposure.n_crs = n_crs
            subexposure.CR_mask = CR_mask
            subexposure.mask = np.logical_or(subexposure.mask, CR_mask)

            subexposure.SCI.header['CRs'] = (n_crs, 'Number of crs detected in box (local median)')
            # Remove CR hits, with NaN or custom CR_replace value
            if t.cr_replace: subexposure.mean_mask_pix(CR_mask, replace=t.cr_replace)

            if t.debug and False:
                # Plot locations of CR hits
                p.subplot(1, 2, 1)
                view(CR_info['mask_y'][subexposure.ystart:subexposure.yend, int(xpix):int(xpix) + 200],
                     show=False, cbar=False, cmap='binary_r', title='CRs (y-axis outliers)')
                p.subplot(1, 2, 2)
                view(CR_info['mask_x'][subexposure.ystart:subexposure.yend, int(xpix):int(xpix) + 200],
                     show=False, cbar=False, cmap='binary_r', title='CRs (x-axis outliers)')
                save_fig()

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
            CR_mask = np.logical_and(CR_mask, abs(subexposure.SCI.data - master_image) > t.cr_thresh)
            n_crs = np.count_nonzero(CR_mask)

            subexposure.CR_mask = CR_mask
            subexposure.n_crs = n_crs

            logger.info('Detected {} CR pixels from subexposure {}'.format(n_crs, i + 1))
            subexposure.mask = np.logical_or(subexposure.mask, CR_mask)
            subexposure.SCI.header['CRs'] = (n_crs, 'Number of crs detected cf to master image')
            new_subs.append(subexposure)

            p.subplot(1, 2, 1)
            p.suptitle('Subexposure {}'.format(i))
            view(subexposure.SCI.data - master_image, title='Diff', cbar=False, show=False)
            p.subplot(1, 2, 2)
            subexposure.cr_sigma = abs((subexposure.SCI.data - master_image) / master_std)
            view(abs((subexposure.SCI.data - master_image) / master_std), title='Sigma', vmin=0, vmax=10, cbar=False,
                 show=False)
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
    if t.cr_local or t.cr_master:
        CR_masks = [subexposure.CR_mask for subexposure in subexposures]
        all_CRs = np.sum(CR_masks, axis=0)
        exposure.CRs = all_CRs

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

        CR_masks = [subexposure.CR_mask for subexposure in subexposures]
        x_count = np.sum([np.sum(mask, axis=0) for mask in CR_masks], axis=0)
        p.subplot(2, 1, 1)
        p.title('Distribution of CRs over pixels')
        p.bar(np.arange(CR_masks[0].shape[0]), x_count, width=1.)
        p.xlabel('x pixel')
        p.subplot(2, 1, 2)
        y_count = np.sum([np.sum(mask, axis=1) for mask in CR_masks], axis=0)
        p.bar(np.arange(CR_masks[0].shape[0]), y_count, width=1.)
        p.xlabel('y pixel')
        p.tight_layout()
        save_fig()

        if t.cr_local:
            # plot also distribution of values
            p.title('Distribution of CR hits')
            p.xlabel('Value (electrons)')
            p.hist(np.hstack(cr_values), 20, edgecolor='k')
            p.gca().set_autoscale_on(False)
            p.plot([t.cr_thresh] * 2, [0, 1e4], ls='--', color='k')
            save_fig()
            print ("This should have a median energy of 1000 electrons")


        DQ_masks = [subexposure.DQs for subexposure in subexposures]
        all_DQs = np.sum(DQ_masks, axis=0)
        most_CRs = np.logical_not(all_DQs) & all_CRs

        logger.info("Nr. of DQs, CRs and CRs minus DQs {}, {}".format(np.sum(all_DQs), np.sum(all_CRs)) +
                    " and {}".format(np.sum(most_CRs)))
        logger.info("Percentage of pixels affected by CRs:{} ".format(np.sum(most_CRs) / (float(most_CRs.shape[0]) *
                                                                      most_CRs.shape[1]) * 100) +
                    "%, which should be ~{}".format(7 * exposure.Primary.header['EXPTIME'] / 1800) +
                    "% if there is no South Atlantic anomaly.")
        view(all_DQs, title='Distribution of DQs over exposure', cbar=False, show=False, vmin=0, vmax=1,
             cmap='binary_r')
        save_fig()
        view(all_CRs, title='Distribution of CRs over exposure', cbar=False, show=False, vmin=0, vmax=1,
             cmap='binary_r')
        save_fig()
        view(most_CRs, title='Distribution of CRs minus DQs over exposure', cbar=False, show=False, vmin=0, vmax=1,
             cmap='binary_r')
        save_fig()
        w0 = 1.04
        w1 = 1.12
        w2 = 1.655
        tot_image_new = np.nansum([sub.SCI.data for sub in new_subs], axis=0)
        wgrid_u = wave_grid[0][int(xpix):int(xpix) + 200]
        wgrid_l = wave_grid[-1][int(xpix):int(xpix) + 200]
        tot_flux = np.sum(tot_image_new[70:210, int(xpix):int(xpix) + 200], axis=0)
        fig, ax = p.subplots(figsize=(7, 15))
        fig.subplots_adjust(hspace=0)
        p.subplot(211)
        p.imshow(tot_image_new[:, int(xpix):int(xpix) + 200], vmin=0,
             vmax=5 * exposure.Primary.header['EXPTIME'], origin='lower', aspect="auto")

        i_change = np.searchsorted(wgrid_u, w1) - np.searchsorted(wgrid_l, w0)

        #p.plot([np.searchsorted(wgrid_u, w0),np.searchsorted(wgrid_l, w0)], [5,L-5], '--', color='maroon')
        p.plot([np.searchsorted(wgrid_u, w1),np.searchsorted(wgrid_l, w1)], [5,L-5], '--', color='maroon')
        p.plot([np.searchsorted(wgrid_u, w2),np.searchsorted(wgrid_l, w2)], [5,L-5], '--', color='maroon')
        #p.plot([np.searchsorted(wgrid_u, w2) + i_change,np.searchsorted(wgrid_l, w2) + i_change], [5,L-5], '--', color='maroon')
        p.title('Raw image with wavelength-range superimposed')


        p.subplot(212)
        p.plot(wgrid_u, np.log10(tot_flux), color='g')
        #p.plot([w0, w0], [min(np.log10(tot_flux)) * 0.9, max(np.log10(tot_flux)) * 1.1], '--', color='maroon')
        p.plot([w1, w1], [min(np.log10(tot_flux)) * 0.9, max(np.log10(tot_flux)) * 1.1], '--', color='maroon')
        p.plot([w2, w2], [min(np.log10(tot_flux)) * 0.9, max(np.log10(tot_flux)) * 1.1], '--', color='maroon')
        p.gca().set_xlim(min(wgrid_u), max(wgrid_u))
        p.ylabel('Total (uncorrected) flux in log-space')
        p.xlabel('Wavelength (um)')
        save_fig()



    # Check for position drift within an exposure and correct
    if False:#t.drift_wshift:
        tot_image = np.sum([sub.SCI.data for sub in subexposures], axis=0)
        tot_image = tot_image[:, int(xpix):int(xpix) + 200]
        ref_row = np.mean(tot_image, axis=0)
        x_row = np.arange(len(ref_row))
        shs, sts, ers, skips = [], [], [], []
        for j, row in enumerate(tot_image):
            skip = 0
            #sh, er = r.spec_pix_shift(x_row, ref_row, x_row, row)
            sh, st, er = r.spec_pix_shift(x_row, ref_row, x_row, row, stretch=True)
            if np.max(row) < np.max(tot_image) / t.drift_rowtol:
                sh, er, skip = 0, 0, 1
            if abs(sh) > t.drift_max:
                sh, er, skip = 0, 0, 1
            shs.append(sh)
            sts.append(st)
            ers.append(er)
            skips.append(skip)

            #p.plot(x_row, ref_row / np.sum(ref_row))
            #p.plot(x_row / st - sh, row / np.sum(row))
            #save_fig()

        shs, sts, ers, skips = map(np.array, [shs, sts, ers, skips])
        h0 = np.sum(shs[len(shs) / 2] == 0)
        h1 = np.sum(shs[len(shs) / 2:] == 0)

        corr_shs = shs.copy()
        if t.drift_width is not None:
            sh_stack = []
            for i in range(-t.drift_width, t.drift_width + 1):
                sh_rol = np.roll(shs[h0:-h1], i)
                if i > 0:
                    sh_rol[:i] = sh_rol[i + 1]
                if i < 0:
                    sh_rol[i:] = sh_rol[i - 1]
                sh_stack.append(sh_rol)
            m_stack, s_stack = np.mean(sh_stack, axis=0), np.std(sh_stack, axis=0)
            corr_shs[h0:-h1] = m_stack

        corr_shs = shs

        if t.debug:
            p.subplot(1, 2, 1)
            p.title('Reference spectrum for shift')
            p.plot(x_row, ref_row)
            p.subplot(1, 2, 2)
            view(tot_image, show=False, cbar=False)
            save_fig()
            p.title('Smoothing length {}'.format(t.drift_width))
            p.plot(corr_shs, np.arange(len(tot_image)), marker='o', ls='--', label='Shifted')
            p.plot(corr_shs[skips], np.arange(len(tot_image))[skips], marker='o', color='r', ls='None', label='Skipped')
            p.legend()
            p.gca().set_autoscale_on(False)
            p.plot(shs, np.arange(len(tot_image)), marker='o', ls='--', alpha=0.1, zorder=-10)
            # p.xlim(-1,1)
            save_fig()

            p.subplot(1,2,1)
            p.title('Smoothing scale {}'.format(t.drift_width))
            p.plot(sts, np.arange(len(tot_image)), marker='o', ls='--', label='Scaled')
            p.plot(sts[skips], np.arange(len(tot_image))[skips], marker='o', color='r', ls='None', label='Skipped')
            p.legend()
            p.gca().set_autoscale_on(False)
            p.plot(shs, np.arange(len(tot_image)), marker='o', ls='--', alpha=0.1, zorder=-10)
            p.gca().set_xlim(0.7, 1.3)
            p.subplot(1,2,2)
            p.title('Reference spectrum for shift')
            view(tot_image, show=False, cbar=False)
            # p.xlim(-1,1)
            save_fig()

        for i in range(len(subexposures)):
            image = subexposures[i].SCI.data
            x = np.arange(image.shape[1])
            new_image = []
            for row, sh, st in zip(image, corr_shs, sts):
                #new_row = np.interp(x, x - sh, row)
                new_row = np.interp(x, x / st - sh, row)
                new_image.append(new_row)

            subexposures[i].SCI.data = np.array(new_image)

    exposure.subexposures = subexposures

    print(exposure.Primary.header['EXPTIME'])
    exptime = exposure.Primary.header['EXPTIME']
    exposure.Primary.header['EXPTIME'] = exptime / len(subexposures) * (len(subexposures) - len(t.remove_subexps_list))
    print(exposure.Primary.header['EXPTIME'], exptime / len(subexposures) * (len(subexposures) - len(t.remove_subexps_list)))

    if t.pdf and t.debug:
        pdf.close()

    if t.save_dir and t.write:
        # Save reduced fits file (_red.fits format)
        logger.info('Saving reduced file to {}'.format(t.save_dir))
        data.write_reduced_fits(exposure.subexposures, exposure.Primary, t, dest_dir=t.save_dir)

    logger.info('Reduction took {:.2f}s'.format(time.time() - t0))
    logger.info('########################################')
    logger.info('###########Finished Reduction###########')
    logger.info('########################################')

    if t.logger:
        # need to close all the file handlers explicitly if performing multiple reductions
        for handler in logger.handlers:
            if hasattr(handler, 'close'):
                handler.close()

    return exposure

def calc_abs_xshift(direct_image, exposure, subexposure, tel, t, scan_direction, x_di, y_di, catalogue, Dxref,
                    DISP_COEFFS, TRACE_COEFFS, conf_file, plot):
    filt = direct_image.Primary.header['FILTER']
    L = subexposure.SCI.data.shape[0]
    subexp_time = subexposure.SCI.header['SAMPTIME']
    XOFF, YOFF = tel.get_wfc3_filter_offs(filt)
    y0 = y_di + (scan_direction * (
            t.scan_rate * subexposure.SCI.header['SAMPTIME'])) / tel.yscale  # guess of y, in pixels
    if t.scanned:
        width0 = subexposure.SCI.header['DELTATIM'] * t.scan_rate / tel.yscale  # initial guess of width
    else:
        width0 = 40
    image = subexposure.SCI.data.copy()
    ystart, ymid, yend = disp.get_yscan(image, x0=x_di, y0=y0, width0=width0, nsig=t.nysig,
                                        two_scans=t.two_scans, debug=True)
    wave_grid, trace = cal.disp_poly(t.conf_file_g141, catalogue, subexp_time, t.scan_rate,
                                     scan_direction, order=1, x_len=L, y_len=L, XOFF=XOFF, YOFF=YOFF,
                                     data_dir=t.source_dir, debug=False, x=x_di,
                                     y=ymid, disp_coef=t.disp_coef)

    if t.use_2nd_order:
        wave_grid2, trace2 = cal.disp_poly(t.conf_file_g141, catalogue, subexp_time, t.scan_rate,
                                           scan_direction, order=2, x_len=L, y_len=L, XOFF=XOFF, YOFF=YOFF,
                                           data_dir=t.source_dir, debug=False, x=x_di,
                                           y=ymid, disp_coef=t.disp_coef)
        wave_grid[wave_grid > 2.0] = wave_grid2[wave_grid > 2.0]

    cal_disp_poly_args = [t.conf_file_g141, catalogue, subexp_time, t.scan_rate,
                          scan_direction, 1, L, L, XOFF, YOFF,
                          t.source_dir, False, x_di, ymid, t.disp_coef]
    tsiaras_args = [L, XOFF, Dxref, ystart, yend, DISP_COEFFS, TRACE_COEFFS, t.grid_y, t.grid_lam, x_di, conf_file,
                    t.contam_thresh]
    return r.find_xshift_di(exposure, subexposure, direct_image, t, wave_grid, cal_disp_poly_args, tsiaras_args, plot,
                            use_2nd_order=t.use_2nd_order, fitpeak=t.peak)


def extract_spectra(reduced_exposure, conf_file=None, **kwargs):
    """
    Extract spectra from reduced exposure, either as file or python object.
    (some issues loading from file, better to run reduction and extraction consecutively)

    kwargs: custom toggles, assigned by keyword arguments

            debug (bool): print and plots some steps for debugging
            logger (bool): log results to terminal (False) or file (True)
            pdf (bool): save plots to a pdf
            Zoom_wavelength (float): Wavelength in microns at which to zoom into when shifting to a reference exposure.

            save_dir (str): destination to save the spectrum
            save_extension (str): _spec.txt default, extension for spectrum file
            save_sub (bool): save the subexposure spectra separately
            remove_subexps_list (list of integers): subexposure numbers that need to be removed from the total spectrum.
                                                    This may be because this subexposure is unreliable due to e.g. a
                                                    satellite crossing event.
                                                    This subexposure will STILL appear in the file with all subexposures
                                                    This does not affect the timing of the exposure

            calc_var (bool): Calculate variances for the spectra or use ERR extension
            mask_neg (bool): can mask all the negative pixels in case of strong negative persistence
            extraction_box (int): n pixel extraction box before optimal extraction
            box_h (float): height of extraction box divided by (16 * scan rate * subexposure time)

            ignore_blobs (bool): avoid reads contaminated heavily by IR blobs
            blob_thresh (float): 0.7

            wshift_sub_exps (bool): shift spectra of each subexposure to correct for drift (linear interp)
            wstretch_sub_exps (bool): when shifting spectra of each subexposure, also apply a stretch
            peak (bool): use only spectrum peak for shift calculation (noisy)
            shift_wv0, shift_wv1 (floats): wavelengths to define peak
            no_interp (bool): If wshift_sub_exps is disabled, you can choose to not interpolate the spectra. This can
                               help if you're going for very high resolution, or if you're worried about undersampling
                               of a spectral line.

            OPTIMAL EXTRACTION
            opt_ext (bool): use optimal extraction
            s (float): sky average or background noise (array-like or scalar), zero if bg removed
            v_0 (float): variance of readout noise
            q (float): effective photon number per pixel value (=1)
            s_clip (float): sigma threshold for outliers in spatial fit, None skips
            s_cosmic (float): sigma threshold for cosmic ray removal, None skips cr removal
            func_type (str): type of function used for fit (poly, gauss or heavi)
            method (str): which optimization method, lsq or one from scipy.optimize.minimize (e.g. CG)
            fit_tol (float): tolerance for the fit, fraction of each flux point
            ypix_F/ypix_R (list): list of central ypixels for the Forward/Reverse scanned subexposures.
            step (float): stepsize in lstsq fitting (epsfcn)
            order (int): order of the spline fit (if func_type=spline) default to 2
            remove_bg (bool): remove the background before optimal extraction or leave it in
            skip_fit (bool): don't fit profile to spectrum but just use fluxes as weights
            k (int): smoothing length for profile calculation
            oe_debug (bool): show debug plots for optimal extraction
            oe_pdf (bool): save plots to seperate pdf

            outliers_to_average(bool): Substitute ourliers with the average of the surrounding pixels
            slopefactor(float): The factor with which two subsequent pixels need to differ to be counted as a 'slope'
            slope_second_order (bool): Whether to use a second order polynomial spline fit on the slope of a column.
            custom_knots (list of np arrays): list with the custom knots to be used when fitting with splines.
                                              if None plots are made such that you can estimate the best knot positions.

            ACCOUNTING FOR DRIFT
            drift_wshift (bool): whether to account for the drift during an exposure. Choose either this or opt_ext
    """

    t0 = time.time()
    # Load config_file and kwargs into dictionaty 't'
    toggles = {'debug':False, 'logger':True, 'pdf':False, 'Zoom_wavelength':None,
               'telescope':HST(), 'scanned':True,
               'save_dir':None, 'save_extension':'_spec.txt', 'save_sub':False,
               'calc_var':True, 'mask_neg':False, 'extraction_box':True, 'box_h':1.,
               'ignore_blobs':False, 'blob_thresh':0.7,
               #'shift_spectra':False, 'stretch_spec':False, 'shift_wv0':1.14, 'shift_wv1':1.6, 'peak':False,
               'wshift_sub_exps':True, 'wstretch_sub_exps':True, 'shift_wv0':1.14, 'shift_wv1':1.6, 'peak':False,
               'drift_wshift':False, 'drift_max':2., 'drift_width':1, 'drift_rowtol':1.1,
               'opt_ext':True, 's':0, 'v_0':20 ** 2, 'q': 1, 's_clip': None, 's_cosmic': None, 'func_type': 'spline',
               'method':'lsq', 'fit_tol':0.01, 'no_interp': False, 'ypix_F':None, 'ypix_R':None,
               'step':None, 'order':1, 'skip_fit':False, 'remove_bg':True, 'fit_dq':False, 'fit_cr':False,
               'top_half':False, 'k_col':9, 'k_row':None, 'object_ind':0, 'oe_debug':0, 'oe_pdf':None,
               'outliers_to_average':False, 'slopefactor':0.1, 'slope_second_order':False,
               'custom_knots_F':None, 'custom_knots_R':None, 'show_knots':False,
               'wshift_to_ref':False, 'ref_exp': None, 'write':True, 'remove_subexps_list':[]
               }
    if conf_file:
        conf_kwargs = data.read_conf_file(conf_file)
        for kw in kwargs:
            conf_kwargs[kw] = kwargs[kw]
        full_kwargs = conf_kwargs
    else:
        full_kwargs = kwargs
    t = f.bunch_kwargs(kwargs=full_kwargs, toggles=toggles, verbose=True, name='extraction')

    assert t.opt_ext + t.drift_wshift < 2, "Choose either opt_ext or drift_wshift, but not both."
    if t.wshift_to_ref or t.wshift_sub_exps:
        assert t.ref_exp is not None, "Define a reference exposure please."
        # Does the scan direction of this exposure match with the subexposure?
        ref_exp_red = data.load(t.save_dir + t.ref_exp + '_red.fits')
        scan_dir_match = np.sign(ref_exp_red.Primary.header['POSTARG2']) == \
                         np.sign(reduced_exposure.Primary.header['POSTARG2'])
        assert t.no_interp is False, "Disable either no_interp, or wshift_to_ref please"

    # check if we need to open the reduced fits file
    if type(reduced_exposure) is str:
        reduced_exposure = data.load(reduced_exposure)

    # Set up logging
    logger = setup_logger('Extraction', reduced_exposure.rootname, t)
    logger.info('Spectral extraction pipeline performed on exposure {}'.format(reduced_exposure.filename))
    logger.info('For configuration, see conf file: {}'.format(conf_file))

    # Set up pdf file for plots
    if t.pdf and t.debug:
        pdf_file = t.save_dir + 'logs/' + reduced_exposure.rootname + '_ext.pdf'
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
        logger.warning('{} is not a suitable observation routine for this pipeline.'.format(
            reduced_exposure.Primary.header['SAMP_SEQ']))

    #Check the scan_direction
    scan_direction, t.scan_rate = data.read_scan_direction(reduced_exposure, t, logger)

    subexposures = reduced_exposure.subexposures
    Ds, DQs, BGs, CRs, Ps = [], [], [], [], []
    spectra, variances = [], []
    for n_sub, subexposure in enumerate(subexposures):
        logger.info('Extracting from subexposure {}'.format(n_sub + 1))

        # Get background level
        try:
            bg = subexposure.bg; bg_err = subexposure.bg_err  # background that has been removed
        except AttributeError:
            bg = 0.; bg_err = 0.; logger.warning('No background defined')

        # Cut out extraction box
        if t.extraction_box:
            box_h = int(t.box_h * 16 * subexposure.SCI.header['EXPTIME'] *
                        reduced_exposure.Primary.header['SCAN_RAT'])
            if n_sub == 0: logger.info('Extracting spectrum with a {} high extraction box'.format(box_h))
            if not hasattr(subexposure, 'waves'): subexposure.waves = np.arange(subexposure.SCI.data.shape[1])

            xpix = subexposure.xpix
            D, mask, bg, CR, err = map(lambda image:r.box_cut(xpix + 100, image, 200, horizontal=True,
                                                              force_shape=False),#, override1=True, override2=True),  #both overrides may not be necessary
                                       [subexposure.SCI.data, subexposure.mask, bg, subexposure.CR_mask,
                                        subexposure.ERR.data])
            print("nr. of CRs", np.sum(subexposure.mask))
            logger.info('Spectrum located at pixel {} in spatial direction'.format(xpix + 100))

            if (t.ypix_F is None) and (t.ypix_R is None):
                ypix = subexposure.ypix
            else:
                if scan_direction == 1:
                    ypix = t.ypix_F[n_sub]
                elif scan_direction == -1:
                    ypix = t.ypix_R[n_sub]
                else:
                    logger.warning("Could not detect a scan direction.")
            print("subexp", subexposure.ypix, t.ypix_F)
            print("ypix is {} for exposure {}".format(int(ypix), n_sub))
            if (t.custom_knots_F is None) and (t.custom_knots_R is None):
                custom_knots = None
            else:
                if scan_direction == 1:
                    custom_knots = t.custom_knots_F[n_sub]
                elif scan_direction == -1:
                    custom_knots = t.custom_knots_R[n_sub]
                else:
                    logger.warning("Could not detect a scan direction.")

            D, mask, bg, CR, err = map(lambda image:r.box_cut(ypix, image, box_h, force_shape=False),
                                       [D, mask, bg, CR, err])
            #print("hi1", D[int(box_h / 2),100])
            #if n_sub > 0:  print(bg), print(bg1)
            #if n_sub > 0: print (np.sum(CR1 & CR)), print(np.sum(CR)), print(CR), print(CR1)
            #Dsave = D[int(box_h / 2),100]
            #D1[int(box_h / 2),100] = D1[1,100]
            #D2[int(box_h / 2),100] = D2[1,100]
            #D3[int(box_h / 2),100] = D3[1,100]
            #D4[int(box_h / 2),100] = D4[1,100]
            #print ("meantje", np.mean(D[0,50:150]), np.std(D[0,50:150]), np.sum(D[:,100]))
            #D1[0,:] = D1[1,:]
            #D2[0,:] = D2[1,:]
            #D3[0,:] = D3[1,:]
            #D4[0,:] = D4[1,:]
            #D[int(box_h / 2), 100] = Dsave
            #print("hi1", D[int(box_h / 2),100], D1[int(box_h / 2),100])
            #print ("boxh", len(D[:,100]), box_h, len(D[0]), subexposure.waves[100])
            #print (D[0,:])
            #print (D[0])
            #print (D[1])
            #print ("summm", np.sum(D))
            tot_flux2 = np.sum(D, axis=1)
            p.plot(np.arange(len(tot_flux2)), np.log10(tot_flux2))
            p.title('A summation over the y-axis of this subexposure')
            p.ylabel('Flux (log scale)')
            save_fig()
            maxdif = np.max([np.abs(tot_flux2[0]- tot_flux2[-1]), np.abs(tot_flux2[1]- tot_flux2[-1]), np.abs(tot_flux2[0]- tot_flux2[-2])])
            print("The difference between the two edges (for a 5-pixel bin for all subexps is):", maxdif * len(subexposures) / 30)
            print("This value may not be larger than ", np.sqrt(4 * np.sum(tot_flux2) / 30), " or else you'll have variations induced by the moving of this subexposure on the y-axis.")
        else:
            if n_sub == 0: logger.warning('No extraction box used')
            D = subexposure.SCI.data
            mask = subexposure.mask
            CR = subexposure.CR_mask
            err = subexposure.ERR.da

        # Get errors
        if not t.calc_var:
            if n_sub == 0: logger.info('Using ERR extension for variance estimate')
            V = np.square(err)  # gain correction already done to _ima files
            t.v_0 = 0
            # Use errors to estimate variance per pixel
        else:
            if n_sub == 0: logger.info('Using flux for variance estimate')
            V = np.abs(D) + bg_err ** 2 + t.v_0  # gain correction already done to _ima files

        if t.top_half:
            ih = 45  # D.shape[0]/2
            D, mask, CR, V = map(lambda arr:arr[ih:], [D, mask, CR, V])
            # D, mask, V = map(lambda arr: arr[:ih], [D, mask, V])

        Ds.append(D)  # Data
        DQs.append(mask)  # Data quality mask
        BGs.append(bg)  # The field that was taken as background
        CRs.append(CR)  # (Empty array for CRs)

        if t.opt_ext:
            # Change mask to optimal extraction format
            # 1s where good, 0s where bad
            M_DQ, M_CR = np.logical_not(mask).astype(int), np.ones_like(mask)  # just put CRs in DQ mask
            M_blank = np.ones_like(M_DQ)
            if t.skip_fit: logger.warning('Not fitting profile, using flux instead')

            # toggle removing the background before optimal extraction
            # or can handle the background in the extraction

            if t.remove_bg:
                D, S = D, 0
            else:
                D, S = D + bg, bg

            if t.oe_pdf:
                oe_pdf_file = t.save_dir + 'logs/{}_{}_fit.pdf'.format(reduced_exposure.Primary.header['ROOTNAME'],
                                                                       n_sub)
                f.silentremove(oe_pdf_file)
            else:
                oe_pdf_file = None
            logger.info('Extracting spectrum with optimal extraction...')
            spec, specV, P, V = ea.extract_spectrum(D=D, S=S, V_0=t.v_0, Q=t.q, V=V, s_clip=t.s_clip,
                                                    s_cosmic=t.s_cosmic, \
                                                    func_type=t.func_type, method=t.method, debug=t.debug,
                                                    oe_debug=t.oe_debug, tol=t.fit_tol, M_DQ=M_DQ, M_CR=M_CR, \
                                                    pdf_file=oe_pdf_file, step=t.step, order=t.order,
                                                    skip_fit=t.skip_fit, bg=bg, slopefactor=t.slopefactor,
                                                    k_col=t.k_col, k_row=t.k_row, fit_dq=t.fit_dq, fit_cr=t.fit_cr,
                                                    outliers_to_average=t.outliers_to_average,
                                                    custom_knots=custom_knots, slope_second_order=t.slope_second_order,
                                                    logger=logger, show_knots=t.show_knots)
            if t.debug and np.any(np.isnan(P)):
                view(D);
                view(P, show=False); view(np.isnan(P), alpha=0.5, cmap='binary', cbar=False)
            if np.any(np.isnan(spec)) and t.debug: p.plot(spec); p.title('NaNs in spectrum'); save_fig()
            assert not np.any(np.isnan(P)), 'NaNs in spatial profile for subexposure {}'.format(n_sub)
            if np.any(np.isnan(spec)):
                logger.warning('NaNs in spectrum for subexposure {}'.format(n_sub))
                logger.warning('Replaced with zeros')
                spec[np.isnan(spec)] = 0.
        else:
            logger.info('Not using optimal extraction - results will be noisy')
            if t.drift_wshift:
                ###########################
                print("not yet shifting drifting")
            M = np.logical_not(mask)
            # Use a smoothed image for weights
            k = 5  # smooth along k pixels
            # Now do a median smooth along each column
            n_smoothloops = 0  # 4
            P = D.copy()
            for _ in range(n_smoothloops):
                smooth_spec = []
                for col in P.T:
                    col = ea.medfilt(col, k)
                    smooth_spec.append(col)
                P = np.vstack(smooth_spec).T
            P = P / np.sum(P, axis=0)
            spec = np.zeros(D.shape[1])
            specV = np.zeros(D.shape[1])
            for i in range(D.shape[1]):
                column = D[:,i]
                spec[i] = np.trapz(column[M[:,i]], np.arange(len(column))[M[:,i]])
                specV[i] = np.trapz(t.v_0 * np.ones(np.sum(M[:,i])) + column[M[:,i]]/t.q)
        Ps.append(P)
        # Optimal Extraction
        # Sum spatial
        spectrum = f.Spectrum(subexposure.waves, spec, x_unit='Wavelength (microns)', y_unit='electrons')
        spectra.append(spectrum)
        variances.append(specV)
    reduced_exposure.Ds = Ds;
    reduced_exposure.DQs = DQs  # ; reduced_exposure.CRs = CRs
    reduced_exposure.BGs = BGs

    if t.debug:
        arrays = []
        for D, DQ_mask in zip(Ds, DQs):
            array = D.copy()
            array[DQ_mask] = np.nan
            arrays.append(array)
        vmin = 0.;
        vmax = np.nanmax(arrays[0])
        arrays_plot(arrays, vmin=vmin, vmax=vmax, cbar=False, tight_layout=False, \
                    size=3, name='Box', show=False, height=0.5)
        save_fig()
        vmax2 = reduced_exposure.Primary.header['EXPTIME'] * 4
        arrays_plot(arrays, vmin=vmin, vmax=vmax2, cbar=False, tight_layout=False, \
                    size=3, name='Box', show=False, height=0.5)
        save_fig()

    # Can't use subexposures where there are too many bad pixels on spectrum
    # threshold is what percentage of the flux missing
    if t.ignore_blobs:
        Ms = DQs
        bad_subs = np.array(
            [np.sum(np.sum(M * P, axis=0)[50:150]) > t.blob_thresh * 100 for P, M in zip(Ps, Ms)]).astype(bool)
        # only check the inner 100 pixels, outside is mostly bg and so can be masked
        good_subs = np.logical_not(bad_subs)
        logger.warning(
            'Ignoring subexposures {} due to bad pixels on spectrum'.format(np.arange(len(bad_subs))[bad_subs]))
        logger.warning(['{:.2f} %'.format(np.sum(np.sum(M * P, axis=0)[50:150])) for P, M in zip(Ps, Ms)])
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
                    alpha = 0.1
                    color = 'k'
                    label = 'Sub {}'.format(i)
                p.plot(spec.x, spec.y, label=label, ls=ls, alpha=alpha, color=color)
            p.xlabel('Wavelength (micron)')
            p.ylabel('electrons')
            p.title('Spectra of subexposures (-- ignored)')
            p.legend()
            save_fig()

            size = 3
            nrows = np.ceil(len(good_subs) / 4.)
            rcParams['figure.figsize'] = size * 4, size * nrows / 2
            for i in range(len(good_subs)):
                p.subplot(nrows, 4, i + 1)
                image = Ds[i].copy()
                M = Ms[i]
                image[M] = np.nan
                if not good_subs[i]:
                    title = 'Ignored'
                else:
                    title = ''
                view(image, title=title, cbar=False, show=False)
                p.axis('off')
            p.tight_layout()
            save_fig()
            rcParams['figure.figsize'] = 10, 5

        old_combined = np.sum([spec.y for spec in spectra], axis=0)
        spectra = [spec for spec, gd in zip(spectra, good_subs) if gd == 1]
        variances = [var for var, gd in zip(variances, good_subs) if gd == 1]
    else:
        good_subs = np.ones(len(Ps))
        bad_subs = np.zeros(len(Ps))

    if len(t.remove_subexps_list) > 0:
        for subnr in t.remove_subexps_list:
            bad_subs[subnr] = 1
            good_subs[subnr] = 0

    reduced_exposure.spectra = spectra
    reduced_exposure.variances = variances

    # So now add all the scaled spectra together, interpolated to a common wavelength scale
    # templates wavelength from first (in time)
    x, y = np.mean([spec.x for spec in spectra], axis=0), np.mean([spec.y for spec in spectra], axis=0)

    #This spectrum needs to be shifted w.r.t. the reference spectrum
    # to account for telescope drifts during an orbit/visit
    if t.wshift_to_ref and t.ref_exp != reduced_exposure.rootname:
        if t.pdf:
            spectra, x_ref, refshift, refshifterr = ea.shift_to_ref(spectra, x, y, t, scan_dir_match, logger, pdf=pdf)
        else:
            spectra, x_ref, refshift, refshifterr = ea.shift_to_ref(spectra, x, y, t, scan_dir_match, logger)
    elif t.ref_exp != reduced_exposure.rootname and t.ref_exp is not None:
        try:
            ref = np.loadtxt(t.save_dir + t.ref_exp + t.save_extension, skiprows=2).T
        except IOError:
            raise Exception("No reduced spectrum found for {}. Please first extract a spectrum ".format(t.ref_exp) +
                            "from {} or disable the wshift_to_ref toggle".format(t.ref_exp))
        x_ref = ref[0]
        refshift, refshifterr = 0, 0
    else:
        refshift, refshifterr= 0, 0
        x_ref = x  #The reference wavelength: the wavelength grid on which to save the spectra


    # Shift the subexposures to go to the common wavelength solution of the first subexposure.
    #t.wshift_sub_exps = False
    #t.wstretch_sub_exps = False
    #t.no_interp = True
    if t.wshift_sub_exps:
        sub_shifts, interp_spectra = ea.match_subexposures(spectra, x_ref, logger, scan_dir_match, peak=t.peak,
                                                           Stretch=t.wstretch_sub_exps)
    elif t.no_interp:
        # Not interpolating the spectra
        #sub_shifts, interp_spectra = ea.match_subexposures(spectra, x_ref, logger, scan_dir_match, peak=t.peak,
        #                                                   Stretch=t.wstretch_sub_exps)
        #print(sub_shifts)
        interp_spectra = [spec.y for spec in spectra]
        sub_shifts = [0] * len(interp_spectra)
    else:
        # Just interpolate to x_ref
        interp_spectra = [np.interp(x_ref, spec.x, spec.y) for spec in spectra]
        sub_shifts = [0] * len(interp_spectra)
    reduced_exposure.sub_shifts = sub_shifts


    if False:
        #Increase the resolution of wavelengths
        x = xnew
        variancesnew = []
        j=0
        for v in variances:
            vnew = []
            for i in range(len(xnew)):
                vnew.append(v[j])
                if j < len(variances) - 1:
                    j+=1
                else:
                    j=0
            variancesnew.append(np.array(vnew))
        variances = variancesnew

    # Compute total spectrum of combined subexposures
    if len(np.array(interp_spectra)[good_subs.astype(bool)]) > 1:
        y = np.nansum(np.array(interp_spectra)[good_subs.astype(bool)], axis=0)
        variance = np.nansum(np.array(variances)[good_subs.astype(bool)], axis=0)
    elif len(np.array(interp_spectra)[good_subs.astype(bool)]) == 1 and good_subs[0] == 1:
        y = interp_spectra[0]
        variance = variances[0]

    #Rescale the exposure time if there are any bad subexposures
    if sum(bad_subs) > 0:
        logger.info('Ignoring subexposures {} for the full spectrum'.format(t.remove_subexps_list))

    # Rescale if ignore one or more subexposures
    if t.ignore_blobs:
        y_before, var_before = y.copy(), variance.copy()
        n_subs = np.sum(good_subs)
        tot_subs = np.sum(good_subs) + np.sum(bad_subs)
        y = y * float(tot_subs) / n_subs
        variance = variance * (float(tot_subs) / n_subs) ** 2
        logger.info('{} subexposures used out of {}'.format(n_subs, len(subexposures)))
        unit = 'Electrons'
    else:
        # Don't rescale
        unit = 'Electrons'

    if t.wshift_to_ref:
        newrefshift = refshift + xpix - ref_exp_red.Primary.header['XPIX']
    else:
        newrefshift = refshift
    exp_spectrum = f.Spectrum(x_ref, y, x_unit='Spectral Pixel', y_unit=unit, refshift=newrefshift,
                                  refshifterr=refshifterr)

    if t.debug and len(spectra) > 1:
        fig = p.figure(figsize=(10,14))
        p.subplot(311)
        p.title('Subexposure spectra (above as extracted, middle shifted and interpolated, below relative difference with first subexposure)')
        for i, spec in enumerate(spectra):
            if good_subs[i] == 1:
                spec.plot(show=False, label=i)
            else:
                spec.plot(show=False, label=i, alpha=0.3)
        p.subplot(312)
        for i, spec in enumerate(interp_spectra):
            if good_subs[i] == 1:
                p.plot(x_ref, spec, label=i)
            else:
                p.plot(x_ref, spec, label=i, alpha=0.3)
        p.ylabel('Electrons')
        p.subplot(313)
        for i, spec in enumerate(interp_spectra):
            if good_subs[i] == 1:
                p.plot(x_ref, spec / np.mean(interp_spectra, axis=0) / np.mean(spec / np.mean(interp_spectra, axis=0) ), label=i)
            else:
                p.plot(x_ref, spec / np.mean(interp_spectra, axis=0)  / np.mean(spec / np.mean(interp_spectra, axis=0) ), label=i, alpha=0.3)
        if t.ignore_blobs:
            p.plot(exp_spectrum.x, old_combined, ls='--', zorder=-1, label='Including bad spectra')
        p.gca().set_ylim(0.9,1.1)
        p.legend()
        fig.subplots_adjust(hspace=0)
        p.xlabel('Wavelength (microns)')

        save_fig()

        p.plot(exp_spectrum.x, exp_spectrum.y, label='Combined spectrum')
        if t.ignore_blobs:
            p.plot(exp_spectrum.x, old_combined, ls='--', zorder=-1, label='Including bad spectra')
            p.legend()
        p.title('Combined spectrum')
        p.xlabel('Wavelength (micron)')
        p.ylabel('Electrons')
        save_fig()






    if t.save_dir and t.write:
        # mark if the scan is forward or reverse, better not to do it here, instead use header of original file
        end = t.save_extension

        fname = t.save_dir + reduced_exposure.Primary.header['ROOTNAME'] + end
        f.silentremove(fname)
        logger.info('Saving spectrum to {}'.format(fname))
        text = '\n'.join(
            ['\t'.join([str(a), str(b), str(c)]) for a, b, c in zip(exp_spectrum.x, exp_spectrum.y, np.sqrt(variance))])
        with open(fname, 'w') as txtf:
            # this assumes you did wavelength calibration
            txtf.write('wave\tflux\terror\n')
            txtf.write('Observation-time: {} \n'.format(reduced_exposure.Primary.header['t']))
            txtf.write(text)

        if t.save_sub:
            fname = t.save_dir + reduced_exposure.Primary.header['ROOTNAME'] + '_subs' + end
            f.silentremove(fname)
            logger.info('Saving subexposure spectra to {}'.format(fname))

            lines = []
            for i, wv in enumerate(exp_spectrum.x):
                line = str(wv)
                for spec, var in zip(interp_spectra, variances):
                    line += '\t{}\t{}'.format(spec[i], var[i] ** 0.5)
                line += '\n'
                lines.append(line)
            with open(fname, 'w') as txtf:
                # this assumes you did wavelength calibration
                txtf.write('wave\tflux\terror\tfor each subexposure number\n')
                txtf.write('Observation-time: {} \n'.format(reduced_exposure.Primary.header['t']))
                for line in lines:
                    txtf.write(line)

    logger.info('Extraction took {:.2f}s'.format(time.time() - t0))
    logger.info('########################################')
    logger.info('###########Finished Extraction##########')
    logger.info('########################################')

    if t.pdf and t.debug:
        pdf.close()

    # Logging
    if t.logger:
        # need to close all the file handlers explicitly if performing multiple extractions
        for handler in logger.handlers:
            if hasattr(handler, 'close'):
                handler.close()

    return exp_spectrum, variance, interp_spectra, variances

def setup_logger(type, rootname, t):
    # Set up logging
    if t.logger:
        if type == 'Extraction':
            extension1 = '_ext.log'
            extension2 = 'ext'
        elif type == 'Reduction':
            extension1 = '_red.log'
            extension2 = 'red'
            type += '#'
        else:
            print ("Wrong type entered into setup_logger")
        log_file = t.save_dir + 'logs/' + rootname + extension1
        f.silentremove(log_file)
        warnings_file = t.save_dir + 'logs/' + extension2 + '_warnings.log'
        f.silentremove(warnings_file)
        logger = logging.getLogger(rootname + '_' + extension2)
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
    logger.info('############Started {}##########'.format(type))
    logger.info('########################################')
    return logger

def create_orbit_cats_gauss(data_dir='/home/jacob/hst_data/GJ-1214/', conf_file='/home/jacob/Project_1/js41_hst.vec',
                            gridsize=5, use_ima=False, nstars=2):
    '''
    Fit a gaussian to direct image to create visit catalogue.
    Use same format as SExtractor for catalogue files.
    '''
    reload(data)
    conf_kwargs = data.read_conf_file(conf_file)

    #if  os.path.isfile(data_dir + 'input_image.lis'):
    data.make_input_image_list(data_dir=data_dir)
    #if not os.path.isfile(data_dir + 'visit_driz.lis'):
    data.make_driz_list(data_dir=data_dir)

    Gaussian2D = modeling.functional_models.Gaussian2D

    with open(data_dir + 'visit_driz.lis', 'r') as driz_list:
        for line in driz_list:
            fname = line[:-1]
            dest_file = data_dir + fname.split('_')[0] + '_flt_1.cat'
            print('Writing to: {}'.format(dest_file))
            if not use_ima:
                if os.path.isfile(data_dir + fname.split('_')[0] + '_drz.fits'):
                    flt_fname = data_dir + fname.split('_')[0] + '_drz.fits'
                elif os.path.isfile(data_dir + fname.split('_')[0] + '_flt.fits'):
                    flt_fname = data_dir + fname.split('_')[0] + '_flt.fits'
                else:
                    raise Exception('No flt or drz file found for {}'.format(fname))
                print(flt_fname)
                di = data.load(flt_fname, hst_file=conf_kwargs['hst_eph_file'], tai_file=conf_kwargs['tai_utc_file'])
                full_images = [di.SCI.data.copy()]
                full_image = full_images[0]
            else:
                # X02b data has the direct image spatially scanned
                # use the first read of the _ima before scanning as direct image
                flt_fname = data_dir + 'direct_images_ima/' + fname.split('_')[0] + '_ima.fits'
                di = data.load(flt_fname, hst_file=conf_kwargs['hst_eph_file'], tai_file=conf_kwargs['tai_utc_file'])
                full_images = [read.SCI.data.copy() for read in di.reads]
                full_images = [image[5:-5, 5:-5] for image in full_images]  # trim off reference pixels
                full_image = full_images[0]

            full_image[np.logical_not(np.isfinite(full_image))] = 0.
            full_image[full_image < 0] = 0
            print('Shape ' + str(full_image.shape))

            view(full_image, direct_image=True, cmap='binary_r', title='Full image - {}'.format(flt_fname.split('/')[-1]), vmin=0,
                 vmax=np.nanmax(full_image) / 100, show=False)
            # Find the part of the image to search, in case there are multiple sources. Area of all pixels except brightest to avoid cosmics
            areas = (np.array(
                [[np.sum(np.square(full_image[gridsize * i:gridsize * (i + 1), gridsize * j:gridsize * (j + 1)]))
                  for j in range(full_image.shape[1] / gridsize)]
                 for i in range(full_image.shape[0] / gridsize)]) -
                     np.array(
                [[np.max(np.square(full_image[gridsize * i:gridsize * (i + 1), gridsize * j:gridsize * (j + 1)]))
                  for j in range(full_image.shape[1] / gridsize)]
                 for i in range(full_image.shape[0] / gridsize)]) )
            # Sort by largest area
            _areas = areas.flatten()
            _areas, indexes = zip(*sorted(zip(_areas, range(len(_areas))), reverse=True))
            # convert indexes back into coords
            indexes = [(index % areas.shape[1], index / areas.shape[0]) for index in indexes]
            for cnt, ind in enumerate(indexes[:nstars]):
                x_ind, y_ind = ind
                x_ind *= gridsize;
                y_ind *= gridsize
                p.plot(x_ind, y_ind, marker='o', ms=gridsize * 2.5, mfc='None', color='r')
            p.show()

            for cnt, ind in enumerate(indexes[:nstars]):
                x_ind, y_ind = ind
                x_ind *= gridsize;
                y_ind *= gridsize
                size_y, size_x = full_image.shape
                if nstars == 1:
                    if size_y > size_x:
                        #Just hope that the star is within this smaller field
                        image = full_image[:size_x,:size_x]
                    elif size_y < size_x:
                        image = full_image[:size_y,:size_y]
                    else:
                        image = full_image
                else:
                    image = full_image[y_ind - gridsize:y_ind + gridsize, x_ind - gridsize:x_ind + gridsize]

                params = Parameters()
                params.add_many(('amplitude', np.nanmax(image), True, 0.), \
                                ('x_mean', x_ind + gridsize / 2, True, 0, image.shape[1]), \
                                ('y_mean', y_ind + gridsize / 2, True, 0, image.shape[0]), \
                                ('x_stddev', 10, True, 0), ('y_stddev', 10, True, 0))

                size_y, size_x = image.shape
                x = np.repeat(np.arange(0, size_x), size_y).reshape(size_y, size_x).T
                y = np.repeat(np.arange(0, size_y), size_x).reshape(size_y, size_x)

                def residuals(params, image, x, y):
                    model = Gaussian2D.evaluate(x, y, amplitude=params['amplitude'], x_mean=params['x_mean'],
                                                y_mean=params['y_mean'], \
                                                x_stddev=params['x_stddev'], y_stddev=params['y_stddev'], theta=0)
                    return (image - model).flatten()

                out = minimize(residuals, params, args=(image, x, y))
                params = out.params
                fit_params = params

                params.pretty_print()
                x_image, y_image = fit_params['x_mean'].value, fit_params['y_mean'].value
                x_std, y_std = fit_params['x_stddev'].value, fit_params['y_stddev'].value

                if nstars == 1:
                    xlim_lo = max(0, int(x_image) - gridsize)
                    xlim_hi = min(int(x_image) + gridsize, size_x)
                    ylim_lo = max(0, int(y_image) - gridsize)
                    ylim_hi = min(int(y_image) + gridsize, size_y)
                    ax = view(image[ylim_lo : ylim_hi, xlim_lo : xlim_hi],
                              direct_image=True, show=False, Return=True, cmap='binary_r', vmin=0,
                              vmax=np.nanmax(full_image) / 100,
                              title='({:.1f}, {:.1f})'.format(x_image, y_image),
                              extent=[int(x_image) - gridsize, int(x_image) + gridsize,
                                      int(y_image) - gridsize, int(y_image) + gridsize])
                else:
                    # Restore to full image reference, also accidentally labeled them all wrong
                    # x_image here refers to 0th axis, which is the spatial scan direction
                    # y_image is 1st axis, spectral direction, normally called x
                    x_image, y_image = x_image + x_ind - gridsize, y_image + y_ind - gridsize
                    ax = view(image, direct_image=True, show=False, Return=True, cmap='binary_r', vmin=0,
                              vmax=np.nanmax(full_image) / 100, title='({:.1f}, {:.1f})'.format(x_image, y_image),
                              extent=[x_ind - gridsize,x_ind + gridsize, y_ind - gridsize,y_ind + gridsize])
                ax.plot(x_image, y_image, marker='x', color='r')
                p.show()

                line = '\t'.join([str(cnt + 1), str(x_image), str(y_image), '0', '0', \
                                  str(y_std), str(x_std), '0.0', '0', '0', '0', '0.0']) + '\n'

                if cnt == 0:
                    with open(dest_file, 'w') as g:

                        g.write('# 1  NUMBER  Running object number\n# 2  X_IMAGE  Object position along x  [pixel]\n')
                        g.write(
                            '# 3  Y_IMAGE  Object position along y  [pixel]\n# 4  X_WORLD  Barycenter position along world x axis  [deg]\n')
                        g.write(
                            '# 5  Y_WORLD  Barycenter position along world y axis  [deg]\n# 6  A_IMAGE  Profile RMS along major axis  [pixel]\n')
                        g.write(
                            '# 7  B_IMAGE  Profile RMS along minor axis  [pixel]\n# 8  THETA_IMAGE  Position angle (CCW/x)  [deg]\n')
                        g.write(
                            '# 9  A_WORLD  Profile RMS along major axis (world units)  [deg]\n# 10 B_WORLD  Profile RMS along minor axis (world units)  [deg]\n')
                        g.write(
                            '# 11 THETA_WORLD  Position angle (CCW/world-x)  [deg]\n# 12 MAG_F1384  Kron-like elliptical aperture magnitude  [mag]\n')

                        g.write(line)
                else:
                    with open(dest_file, 'a') as g:
                        g.write(line)
                # catalogue create for direct image
