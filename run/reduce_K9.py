from hstscan import data, my_fns, pipeline
pipe = pipeline
import time, sys, os, shutil
from multiprocessing.pool import Pool
import logging
from progress.bar import Bar


if len(sys.argv) > 1:
    source_files = []
    for arg in sys.argv[1:]: source_files.append(arg)
else: source_files = ['eclipse.lis']; print 'Reducing default source files {}'.format(source_files)

system='WASP-12'
GO = '15820'
data_dir='/home/jacob/hstscan/data/Kelt-9/GO{}/'.format(GO)
direction='a'
red_file='/home/jacob/hstscan/data/Kelt-9/GO{}/red.conf'.format(GO)
ext_file='/home/jacob/hstscan/data/Kelt-9/GO{}/ext.conf'.format(GO)
dest_dir='/home/jacob/hstscan/data/Kelt-9/GO{}/test/'.format(GO)
log_file = dest_dir+'logs/all_terminal.log'
cores = 3

if __name__ == '__main__':

    shutil.copyfile(red_file, dest_dir+'logs/log_red.conf')    
    shutil.copyfile(ext_file, dest_dir+'logs/log_ext.conf')
    red_file = dest_dir+'logs/log_red.conf'
    ext_file = dest_dir+'logs/log_ext.conf'

    my_fns.silentremove(log_file)
    log_file = open(log_file,'a')
    sys.stdout = log_file
    sys.stderr = log_file

    for source_file in source_files:

        t0 = time.time()
        source_dir = data_dir
        with open(source_dir+source_file,'r') as g:
            lines = g.readlines()
        lines = [line.split('\t') for line in lines if not line.startswith('#')]
        if direction != 'a':
            lines = [line for line in lines if line[-1].startswith(direction)]
        files = [line[0] for line in lines if line[1].startswith('G')]

        ###############
        ### Skipping ##
        ###############
        # (optional)
        # e.g. files = files[78:]
        #files = files[420:]
        
        bar = Bar('Processing exposures', max=len(files))

        os.nice(1)
        def func(fname):
            exp = pipe.reduce_exposure(fname, conf_file=red_file)
            pipe.extract_spectra(exp, conf_file=ext_file)
            return

        def try_func(args):
            try:
                return func(args)
            except Exception:
                logging.exception("f(%r) failed" % (args,))
                raise

        pool = Pool(cores)
        r = pool.imap_unordered(try_func, files)
        for _ in r: bar.next()


        tottime = (time.time()-t0)/60.
        print 'Reduction and Extraction completed.'
        print 'Time taken: {:.2f} minutes'.format(tottime)
        log_file.close()
        sys.stdout = sys.__stdout__
        print
        print 'Reduction and Extraction completed for {}.'.format(source_files)
        print 'Time taken: {:.2f} minutes'.format(tottime)
