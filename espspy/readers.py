import os
import struct
import numpy as np
import pandas as pd
from collections import OrderedDict, namedtuple

# Keys of preamble correspond to members of the 'preamble' struct in
# esps.header.h
# Values are the corresponding data types.
preamble = OrderedDict([
    ('machine_code', 'l'),
    ('check_code', 'l'),
    ('data_offset', 'l'),
    ('record_size', 'l'),
    ('check', 'l'),
    ('edr', 'l'),            # YES if EDR_ESPS, NO if native
    ('align_pad_size', 'l'), # alignment pad needed for some SD files
    ('foreign_hd', 'l')      # pointer to foreign header, -1 if
])
EspsPreamble = namedtuple('EspsPreamble', ' '.join(preamble.keys()))
_preamble_fmt = ''.join(preamble.values())

# Keys of fixpart correspond to members of the 'fixpart' struct in
# esps/header.h.
# Values are the corresponding data types.
# Note that 'pad1' can be excluded at compile time in header.h.
fixpart = OrderedDict([
    ('type', 'h'),         # file type
    ('pad1', 'h'),
    ('check', 'l'),        # check field
    ('date', '26s'),       # file creation date
    ('hdvers', '8s'),      # header version
    ('prog', '16s'),       # program name
    ('vers', '8s'),        # prog version
    ('progdate', '26s'),   # prog compile date
    ('ndrec', 'l'),        # number of data records
    ('tag', 'h'),          # YES if data has tag
    ('nd1', 'h'),          # used for reading old files only
    ('ndouble', 'l'),      # number of doubles
    ('nfloat', 'l'),       # number of floats
    ('nlong', 'l'),        # number of longs
    ('nshort', 'l'),       # number of shorts
    ('nchar', 'l'),        # number of chars
    ('fixpartsiz', 'l'),   # fixed header size
    ('hsize', 'l'),        # total header size
    ('user', '8s'),        # user name
# headers.c appears to read spares next and get edr and machine_code from
# preamble
# (in recursive_rh). What does it do on writing header?
#    ('edr', 'h'),          # YES if EDR_ESPS, NO for native
#    ('machine_code', 'h'), # machine that produced file
    ('spares', '20s')      # spares (easiest way to read 20 bytes as a unit) 
])
EspsFixpart = namedtuple('EspsFixpart', fixpart.keys())
_fixpart_fmt = ''.join(fixpart.values())

feapart1 = OrderedDict([
    ('fea_type', 'h'),         # fea file subtype
    ('segment_labeled', 'h'),
    ('field_count', 'h'),
    ('field_order', 'h'),
    ('dummy', '36s'), # I think this contains (double, triple) pointers to sizes,
                    # ranks, dimens, types, enums, starts, derived, srcfields.
                    # The size to read is the number of bytes in nine (long)
                    # pointers.
    ('spares', '32s') # The size is number of bytes in 16 shorts.
])
EspsFeapart1 = namedtuple('EspsFeapart1', feapart1.keys())
_feapart1_fmt = ''.join(feapart1.values())

feapart2 = OrderedDict([
    ('ndouble', 'l'),
    ('nfloat', 'l'),
    ('nlong', 'l'),
    ('nshort', 'l'),
    ('nbyte', 'l'),
    ('ndcplx', 'l'),
    ('nfcplx', 'l'),
    ('nlcplx', 'l'),
    ('nscplx', 'l'),
    ('nbcplx', 'l'),
])
EspsFeapart2 = namedtuple('EspsFeapart2', feapart2.keys())
_feapart2_fmt = ''.join(feapart2.values())

# The ESPS Feature file subtype codes are defined in esps/fea.h
#define FEA_VQ	1		/* vector quant file */
#define FEA_ANA 2		/* ana file */
#define FEA_STAT 3		/* statistics file */
#define FEA_QHIST 4		/* histogram file */
#define FEA_DST	5		/* quantized distortion file */
#define FEA_2KB 6	
#define FEA_SPEC 7		/* spectrum file */
#define FEA_SD 8                /* sampled-data file */
#define FEA_FILT 9              /* filter file */
EspsFeaType = OrderedDict([
    (0, {'subtype': 'NONE', 'hdr_func': 'read_fea_header'}),
    (1, {'subtype': 'FEA_VQ'}),     # vector quant file
    (2, {'subtype': 'FEA_ANA', 'hdr_func': 'read_fea_ana_header'}), # ana file
    (3, {'subtype': 'FEA_STAT'}),   # statistics file
    (4, {'subtype': 'FEA_QHIST'}),  # histogram file
    (5, {'subtype': 'FEA_DST'}),    # quantized distortion file
    (6, {'subtype': 'FEA_2KB'}),
    (7, {'subtype': 'FEA_SPEC', 'hdr_func': 'read_fea_header'}),   # spectrum file
    (8, {'subtype': 'FEA_SD', 'hdr_func': 'read_fea_header'}), # sampled-data file
    (9, {'subtype': 'FEA_FILT'}),  # filter file
])

# The ESPS variable header codes are defined in header.h.
#define PT_ENDPAR	0		/* codes for variable items */
#define PT_SOURCE	1
#define PT_TYPTXT	2
#define PT_REFER	3
#define PT_HEADER	4
#define PT_PRE		5
#define PT_FILTER	6
#define PT_PRIOR	7
#define PT_WEIGHT	8
#define PT_PREFILTER	9
#define PT_LPF		10
#define PT_COMMENT	11
#define PT_DEEMP	12
#define PT_GENHD	13
#define PT_REFHD	14
#define PT_CWD		15
#define PT_MAX		15		/* change as we add codes */
EspsVarheadType = OrderedDict([
    (0, {'subtype': 'PT_ENDPAR', 'is_string': False}),
    (1, {'subtype': 'PT_SOURCE', 'is_string': True, 'fld': 'source'}),
    (2, {'subtype': 'PT_TYPTXT', 'is_string': True, 'fld': 'typtxt'}),
    (3, {'subtype': 'PT_REFER', 'is_string': True, 'fld': 'refer'}),
    (4, {'subtype': 'PT_HEADER', 'is_string': False}),
    (5, {'subtype': 'PT_PRE', 'is_string': False}),
    (6, {'subtype': 'PT_FILTER', 'is_string': False}),
    (7, {'subtype': 'PT_PRIOR', 'is_string': False}),
    (8, {'subtype': 'PT_WEIGHT', 'is_string': False}),
    (9, {'subtype': 'PT_PREFILTER', 'is_string': False}),
    (10, {'subtype': 'PT_LPF', 'is_string': False}),
    (11, {'subtype': 'PT_COMMENT', 'is_string': True, 'fld': 'comment'}),
    (12, {'subtype': 'PT_DEEMP', 'is_string': False}),
    (13, {'subtype': 'PT_GENHD', 'is_string': True}),
    (14, {'subtype': 'PT_REFHD', 'is_string': False}),
    (15, {'subtype': 'PT_CWD', 'is_string': True, 'fld': 'current_path'}),
])

# The ESPS datatype codes are defined in esps.h.
#define DOUBLE 1
#define FLOAT 2
#define LONG 3
#define SHORT 4
#define CHAR 5
#define UNDEF 6
#define CODED 7
#define BYTE 8
#define EFILE 9
#define AFILE 10
#define DOUBLE_CPLX 11
#define FLOAT_CPLX 12
#define LONG_CPLX 13
#define SHORT_CPLX 14
#define BYTE_CPLX 15
# TODO: What do we do about the empty types?
EspsDtype = OrderedDict([
    (1, 'd'), # DOUBLE
    (2, 'f'), # FLOAT
    (3, 'l'), # LONG
    (4, 'h'), # SHORT
    (5, 's'), # CHAR
    (6, None),  # UNDEF
    (7, 'h'),  # CODED
    (8, 's'), # BYTE
    (9, 's'), # EFILE
    (10, 's'), # AFILE
    (11, None), # DOUBLE_CPLX
    (12, None), # FLOAT_CPLX
    (13, None), # LONG_CPLX
    (14, None), # SHORT_CPLX
    (15, None), # BYTE_CPLX
])

# Fields found in .fea file data records.
EspsFeaDtype = OrderedDict([
    ('ndouble', 'd'),
    ('ndcplx', 'd'),
    ('nfloat', 'f'),
    ('nfcplx', 'f'),
    ('nlong', 'l'),
    ('nlcplx', 'l'),
    ('nshort', 'h'),
    ('nscplx', 'h'),
    ('nbyte', 's'),
    ('nbcplx', 's'),
])

class EspsHeader(object):
    def __init__(self, level=0):
        self.level = level
        self.fixpart = None
        self.genhd = EspsGenhd()
        self.variable = EspsVarhead()

class EspsVarhead(object):
    def __init__(self):
        self.source = []
        self.typtxt = None
        self.comment = None
        self.current_path = None
        self.refer = None
        self.srchead = []
        self.refhd = []

class EspsGenhd(object):
    pass

class EspsFeaReader(object):
    """A class for reading ESPS FEA files."""
    def __init__(self, infile=None, open_mode='rb',
            *args, **kwargs):
        super(EspsFeaReader, self).__init__()
        self.byte_order = None
        self.ftype = None   # The file type (we only handle FEA file type 13)
        self._fea_type = None
        self.open_mode = open_mode
        self.fea = None
        self.fea_items = None
        self._fromfile_dtype = None
        if infile != None and self.open_mode != None:
            try:
                self.fh = open(infile, self.open_mode)
            except TypeError:  # An open filehandle was passed in. 
                self.fh = infile
            except:
                raise RuntimeError('Could not open fea file.')
            self.preamble = self.read_preamble()
            self.fixpart_fmt = self.byte_order + _fixpart_fmt
            self.feapart1_fmt = self.byte_order + _feapart1_fmt
            self.feapart2_fmt = self.byte_order + _feapart2_fmt
            # TODO: recursion doesn't actually work. Only the top-level
            # header is returned.
            self.hdr = self.recursive_rh()
            # Fast-forward to data section. We do not use seek() in case we
            # are reading from a pipe since pipes are not seekable.
            # TODO: In limited testing, the data offset value in the preamble
            # was 0 for an esps utility (sgram) sending its output to stdout
            # ('-'), and this fast-forwarding doesn't know where to stop.
            # It appears the solution will be to fully implement reading of
            # the header, including recursion, which should leave the current
            # read position of the filehandle at the start of the data.
            while self.fh.tell() < self.preamble.data_offset:
                self.fh.read(1)
        self.data = None
        self._fromfile_dtype = None

    # Context manager setup
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        '''Close filehandle.'''
        self.fh.close()

# TODO: nrecs presumably will not work if self.fh is attached to
# a pipe rather than a normal file. Think about how to handle the
# situation.
    @property
    def nrecs(self):
        '''The number of data records in the file.'''
        datasize = os.stat(self.fh.name).st_size - self.preamble.data_offset
        try:
            assert(datasize % self.preamble.record_size == 0)
        except AssertionError:
            msg = 'Data area is not an integer multiple of record size.'
            raise RuntimeError(msg)
        return np.int_(datasize / self.preamble.record_size)

    @property
    def fromfile_dtype(self):
        '''To be implemented in derived class.'''
# TODO: default implementation?
        pass

    @property
    def genhd_fields(self):
        '''Return the list of fields found in the genhd property.'''
        return list(vars(self.hdr.genhd).keys())

    @property
    def fea_type(self):
        '''The FEA file subtype. The file handle cursor must be in the
        correct location the first time this property is accessed.'''
        # TODO: calls to seek() are illegal when the input file handle
        # is reading from a pipe, and this will have to be fixed to read
        # from stdout.
        if self._fea_type is None:
            cur = self.fh.tell()
            self._fea_type = struct.unpack(
                self.byte_order + 'h',
                self.fh.read(2)
            )[0]
            self.fh.seek(cur)
        return self._fea_type

    def read_preamble(self):
        '''Read and return the ESPS file preamble and preamble format.'''
        preamble = self.fh.read(32)
        magic = preamble[16:20]
        if magic == b'\x00\x00j\x1a':    # big endian
            self.byte_order = '>'
        elif magic == b'\x1aj\x00\x00':  # little endian
            self.byte_order = '<'
        preamble_fmt = self.byte_order + _preamble_fmt
        return EspsPreamble._make(
            struct.unpack(preamble_fmt, preamble)
        )

    def read_fixpart(self):
        '''Read the fixed part of the ESPS header.'''
        fixpart = EspsFixpart._make(
            struct.unpack(
                self.fixpart_fmt,
                self.fh.read(struct.calcsize(self.fixpart_fmt))
            )
        )
        stringified = {}
        for attr in ('date', 'hdvers', 'prog', 'vers', 'progdate', 'user'):
            val = getattr(fixpart, attr)
            stringified[attr] = val.replace(b'\x00', b'').decode('ascii')
        fixpart = fixpart._replace(**stringified)
        return fixpart

    def read_fea_header(self, level=0):
        '''Read header of FT_FEA with no fea subtype.'''
# TODO: combine feapart1 and feapart2
        hdr = EspsHeader(level)
        feapart1 = EspsFeapart1._make(
            struct.unpack(
                self.feapart1_fmt,
                self.fh.read(struct.calcsize(self.feapart1_fmt))
            )
        )
        hdr.feapart1 = feapart1
        if feapart1.field_order != 0:
            msg = 'Support for field_order YES not implemented.'
            raise NotImplementedError(msg)
        size = feapart1.field_count
        lfmt = '{:}{:d}l'.format(self.byte_order, size)
        hfmt = '{:}{:d}h'.format(self.byte_order, size)
# TODO: these should not be stored directly in hdr; need different names
        hdr.sizes = struct.unpack(lfmt, self.fh.read(struct.calcsize(lfmt)))
        hdr.starts = struct.unpack(lfmt, self.fh.read(struct.calcsize(lfmt)))
        hdr.ranks = struct.unpack(hfmt, self.fh.read(struct.calcsize(hfmt)))
        hdr.types = struct.unpack(hfmt, self.fh.read(struct.calcsize(hfmt)))
        hdr.names = []
        hdr.dimens = []
        hdr.srcfields = []
        feapart2 = EspsFeapart2._make(
            struct.unpack(
                self.feapart2_fmt,
                self.fh.read(struct.calcsize(self.feapart2_fmt))
            )
        )
        hdr.feapart2 = feapart2
        hdr.derived = struct.unpack(hfmt, self.fh.read(struct.calcsize(hfmt)))
        for rnk, typ in zip(hdr.ranks, hdr.types):
            slen = struct.unpack(self.byte_order + 'h', self.fh.read(2))[0]
            sfmt = '{:d}s'.format(slen)
            hdr.names.append(
                struct.unpack(
                    sfmt,
                    self.fh.read(struct.calcsize(sfmt)))[0].decode('ascii')
            )
            if rnk != 0:
                lfmt = '{:}{:d}l'.format(self.byte_order, rnk)
                hdr.dimens.append(
                    struct.unpack(lfmt, self.fh.read(struct.calcsize(lfmt)))[0]
                )
            if typ == 7: # CODED defined in esps.h
                # TODO: implement this from lines 1281-1302 in headers.c
                # Note that read_coded() is implemented already and may be
                # used to read the coded value.
                raise NotImplementedError('CODED fea type not implemented.')
            slen = struct.unpack(self.byte_order + 'h', self.fh.read(2))[0]
            sfmt = '{:d}s'.format(slen)
            flds = []
            for i in np.arange(slen):
                slen2 = struct.unpack(self.byte_order + 'h', self.fh.read(2))[0]
                sfmt2 = '{:d}s'.format(slen2)
                flds.append(
                    struct.unpack(
                        sfmt2,
                        self.fh.read(struct.calcsize(sfmt2)))[0].decode('ascii')
                )
            hdr.srcfields.append(flds)

        # read variable part
        while True:
            code = struct.unpack(self.byte_order + 'h', self.fh.read(2))[0]
            slen = struct.unpack(self.byte_order + 'h', self.fh.read(2))[0]
            if EspsVarheadType[code]['is_string'] is True:
                sfmt = '{:d}s'.format(slen * 4)
                name = struct.unpack(
                    sfmt,
                    self.fh.read(struct.calcsize(sfmt))
                )[0].replace(b'\x00', b'').decode('ascii')
            if code == 0:  # PT_ENDPAR
                break
            try:
                if EspsVarheadType[code]['subtype'] == 'PT_GENHD':
                    setattr(hdr.genhd, name, self.read_genhd())
                elif EspsVarheadType[code]['subtype'] == 'PT_HEADER':
                    break
                    # TODO: recursive header reading is untested.
                    #hdr.variable.srchead.append(self.recursive_rh(hdr.level+1))
                elif EspsVarheadType[code]['is_string'] is True: # all other string types
                    setattr(hdr.variable, EspsVarheadType[code]['fld'], name)
                else:
                # TODO: I am hoping that the genhd fields, e.g. record_freq,
                # start_time, come first, as they are the only ones we are
                # interested in. Stop processing after they are found.
                    break
            except KeyError:
                try:
                    msg = 'Reading of ESPS {:} variable header not implemented.'.format(
                        EspsVarheadType[code]['subtype']
                    )
                    raise NotImplementedError(msg)
                except KeyError:
                    msg = 'Did not recognize ESPS variable header code {:d}'.format(
                        code
                    )
                    raise RuntimeError(msg)
        return hdr

    def read_genhd(self):
        '''Read generic header.'''
        sz = struct.unpack(self.byte_order + 'i', self.fh.read(4))[0]
        typ = struct.unpack(self.byte_order + 'h', self.fh.read(2))[0]
        val = None
        if EspsDtype[typ] is None:
            # TODO: implement additional dtypes?
            msg = 'fea type {:d} not implemented in read_genhd.'.format(type)
            raise NotImplementedError(msg)
        elif typ == 7: # CODED
            val = self.read_coded()
        else:
            vfmt = '{:}{:d}{:}'.format(self.byte_order, sz, EspsDtype[typ])
            val = struct.unpack(
                vfmt,
                self.fh.read(struct.calcsize(vfmt))
            )[0]
            if EspsDtype[typ] == 's':
                val = val.replace(b'\x00', b'').decode('ascii')
        return val

    def read_coded(self):
        '''Read 'coded' header value.'''
        codes = {}
        n = 0
        lngth = struct.unpack(self.byte_order + 'h', self.fh.read(2))[0]
        while lngth != 0:
            codefmt = '{:d}s'.format(lngth)
            codes[n] = struct.unpack(
                codefmt, self.fh.read(struct.calcsize(codefmt))
            )[0].replace(b'\x00', b'').decode('ascii')
            lngth = struct.unpack('h', self.fh.read(struct.calcsize('h')))[0]
            n += 1
        key = struct.unpack('h', self.fh.read(struct.calcsize('h')))[0]
        return codes[key]

    def read_fea_ana_header(self):
        '''Read header of a FEA_ANA file.'''
        print('fea_ana type')

    # TODO: despite the name 'recursive_rh', recursion does not yet work.
    def recursive_rh(self, level=0):
        '''Read the file header.'''
        if level == 0:
            fixpart = self.read_fixpart()
            self.ftype = fixpart.type
        if self.ftype == 13:
            try:
                hdr_func = getattr(
                    self,
                    EspsFeaType[self.fea_type]['hdr_func']
                )
                hdr = hdr_func(level)
                if level == 0:
                    hdr.fixpart = fixpart
            except KeyError:
                try:
                    msg = 'Reading of ESPS {:} files not implemented'.format(
                        EspsFeaType[self.fea_type]['subtype']
                    )
                    raise RuntimeError(msg)
                except KeyError:
                    msg = 'Did not recognize ESPS FEA file subtype {:d}'.format(
                        self.fea_type
                    )
                    raise RuntimeError(msg)
        else:
            raise RuntimeError('Do not know how to read type {:}.'.format(
                self.ftype
            ))
        return hdr

    def check_data_read(self):
        '''Perform checks on whether data was read correctly.'''
        # After reading, filehandle should be at the end of the file.
        assert(self.fh.tell() == os.stat(self.fh.name).st_size)
        # TODO: additional checks against nframes or other header values
        # to see that records were read correctly.

class EspsSgramReader(EspsFeaReader):
    '''A class for reading ESPS .fspec files produced by the sgram command.'''
    def __init__(self, infile=None, open_mode='rb', *args, **kwargs):
        super(EspsSgramReader, self).__init__(infile=infile,
            open_mode=open_mode, *args, **kwargs)
        # Read the data records.
        self.data = np.fromfile(self.fh, self.fromfile_dtype)
        self.check_data_read()
        bins = np.arange(self.num_freqs + 1, dtype=float)
        self._bins = bins * self.sf / self.fft_length
        self._times = (np.arange(len(self.data)) / self.record_freq) + \
                   self.start_time
        self.set_data_view()

    @property
    def fromfile_dtype(self):
        '''The dtypes for unpacking item records using np.fromfile().'''
# TODO: Not sure this is right for all .fspec files.
        if self._fromfile_dtype is None or self._fromfile_dtype == []:
            self._fromfile_dtype = []
            flds = []
            try:
                assert(self.hdr.feapart2.nfloat == 1)
            except AssertionError:
                sys.stderr.write('Unexpected number of floats.')
                raise
# TODO: check fixpart for tag=1 before adding 'tag' to flds? Note that
# fixpart has 'tag', 'nfloat', and 'nchar' values whereas self.hdr.feapart2
# has 'nfloat' and 'nbyte' values but no 'tag'.
            flds.append(('tag', self.byte_order + 'u4'))
            flds.append(('tot_power', self.byte_order + 'f4'))
            flds.append(
                ('re_spec_val', '{:d}B'.format(self.hdr.feapart2.nbyte))
            )
            self._fromfile_dtype = np.dtype(flds)
        return self._fromfile_dtype

    @property
    def sgram(self):
        '''Return the spectrogram values of the current data view.'''
        return self.data[:]['re_spec_val'][
            self.data_view['t1idx']:self.data_view['t2idx'],
            self.data_view['hz1idx']:self.data_view['hz2idx']
        ].T

    @property
    def power(self):
        '''Return the power values of the current data view.'''
        return self.data[:]['tot_power'][
            self.data_view['t1idx']:self.data_view['t2idx']
        ]

    @property
    def tag(self):
        '''Return the tag values of the current data view.'''
# TODO: What are the 'tag' values? Should this property have a better name?
# The values seem to be related to the step size and I think indicates the
# sample in the original audio file where the record is centered.
# pplain reports this value as 'tag'.
        return self.data[:]['tag'][
            self.data_view['t1idx']:self.data_view['t2idx'],
            self.data_view['hz1idx']:self.data_view['hz2idx']
        ]

    @property
    def bin_hz(self):
        '''Return an array of the centers of the fft frequency bins in Hertz
        of the current data view.'''
        return self._bins[self.data_view['hz1idx']:self.data_view['hz2idx']]

    @property
    def times(self):
        '''Return the timepoints of each spectral slice in sgram of the
        current data view.'''
        return self._times[self.data_view['t1idx']:self.data_view['t2idx']]

    @property
    def data_view_extent(self):
        '''Return sgram slice's extents suitable for use as imshow()'s
        'extent' param for current data view.'''
        tdelta = self._times[1] - self._times[0]
        fdelta = self._bins[1] - self._bins[0]
        return [
            self._times[self.data_view['t1idx']] - tdelta/2,
            self._times[self.data_view['t2idx']] + tdelta/2,
            self._bins[self.data_view['hz1idx']] - fdelta/2,
            self._bins[self.data_view['hz2idx']] + fdelta/2
        ]

    @property
    def sf(self):
        '''Return the sf (sampling frequency) value.'''
        return self.hdr.genhd.sf

    @property
    def window_type(self):
        '''Return the window_type value.'''
        return self.hdr.genhd.window_type

    @property
    def frmlen(self):
        '''Return the frmlen value.'''
        return self.hdr.genhd.frmlen

    @property
    def frame_meth(self):
        '''Return the frame_meth value.'''
        return self.hdr.genhd.frame_meth

    @property
    def freq_format(self):
        '''Return the freq_format value.'''
        return self.hdr.genhd.freq_format

    @property
    def record_freq(self):
        '''Return the record_freq value.'''
        return self.hdr.genhd.record_freq

    @property
    def start(self):
        '''Return the start value.'''
        return self.hdr.genhd.start

    @property
    def sgram_method(self):
        '''Return the sgram_method value.'''
        return self.hdr.genhd.sgram_method

    @property
    def fft_order(self):
        '''Return the fft_order value.'''
        return self.hdr.genhd.fft_order

    @property
    def fft_length(self):
        '''Return the fft_length value.'''
        return self.hdr.genhd.fft_length

    @property
    def spec_type(self):
        '''Return the spec_type value.'''
        return self.hdr.genhd.spec_type

    @property
    def step(self):
        '''Return the step value.'''
        return self.hdr.genhd.step

    @property
    def num_freqs(self):
        '''Return the num_freqs value.'''
        return self.hdr.genhd.num_freqs

    @property
    def pre_emphasis(self):
        '''Return the pre_emphasis value.'''
        return self.hdr.genhd.pre_emphasis

    @property
    def contin(self):
        '''Return the contin value.'''
        return self.hdr.genhd.contin

    @property
    def start_time(self):
        '''Return the start_time value.'''
        return self.hdr.genhd.start_time

    def set_data_view(self, t1=0.0, t2=np.inf, hz1=0.0, hz2=np.inf):
        '''Set the time and frequency ranges that determine the values
        returned by data-related properties.'''
        t1idx = (np.abs(self._times - t1)).argmin()
        if t2 != np.inf:
            t2idx = (np.abs(self._times - t2)).argmin()
        else:
            t2idx = len(self._times) - 1
        hz1idx = (np.abs(self._bins - hz1)).argmin()
        if hz2 != np.inf:
            hz2idx = (np.abs(self._bins - hz2)).argmin()
        else:
            hz2idx = self.num_freqs
        self.data_view = {
            't1idx': t1idx, 't2idx': t2idx,
            'hz1idx': hz1idx, 'hz2idx': hz2idx
        }

class EspsFormantReader(EspsFeaReader):
    '''A class for reading ESPS .fb files produced by formant and rformant
commands.'''
    def __init__(self, infile=None, open_mode='rb', *args, **kwargs):
        super(EspsFormantReader, self).__init__(infile=infile,
            open_mode=open_mode, *args, **kwargs)
        self._times = None

    def __str__(self):
        return f'''{type(self)}
   {self.fh.name}
   start time (genhd): {self.start_time}
   record freq (genhd): {self.record_freq}
   data offset (preamble): {self.preamble.data_offset}
   record size (preamble): {self.preamble.record_size}
   field count: {self.hdr.feapart1.field_count}
   field order: {self.hdr.feapart1.field_order}
   names: {self.hdr.names}
   sizes: {self.hdr.sizes}
   ranks: {self.hdr.ranks}
   dimens: {self.hdr.dimens}
   types: {self.hdr.types}
   fromfile_dtype {self.fromfile_dtype}
   fromfile_dtype size (should be the same as record size): {self.fromfile_dtype.itemsize}
        '''

    @property
    def record_freq(self):
        return self.hdr.genhd.record_freq

    @property
    def start_time(self):
        return self.hdr.genhd.start_time
    
    @property
    def times(self):
        if self._times is None:
            frame_period = 1 / self.record_freq
            self._times = \
                (np.arange(self.nrecs) * frame_period) \
                + self.start_time
        return self._times

    @property
    def fromfile_dtype(self):
        '''The dtypes for unpacking item records using np.fromfile().'''
# TODO: according to the man page of .fea file types, the counts of the dtypes
# in the common part of the header and the feapart can differ. Figure out
# whether this implementation is correct for all situations.
# TODO: check for other dtype than ndouble and throw error (or handle) if found
        if self._fromfile_dtype is None or self._fromfile_dtype == []:
            self._fromfile_dtype = []
            flds = []
# TODO: The values in self.hdr.dimens in the .fb files I have seen so far
# have equivalent values as self.hdr.sizes, and it's not entirely clear to
# me which field should be used here. The esps docs on .fea files seem to
# say that the sizes field is correct, but I'm not certain that that is
# the correct interpretation.
            n_and_sz = zip(self.hdr.names, self.hdr.sizes)
            for sublist in [[n] * sz for n, sz in n_and_sz]:
                for i, item in enumerate(sublist):
                    flds.append(f'{item}{i+1:d}')
            self._fromfile_dtype = np.dtype(
                [(fld, self.byte_order + 'f8') for fld in flds]
            )
        return self._fromfile_dtype

    def get_df(self, t1=0.0, t2=np.Inf):
        tidx = np.nonzero((self.times >= t1) & (self.times <= t2))[0]
        offset = (tidx[0] * self.preamble.record_size) \
                 + self.preamble.data_offset
        # Read the data records.
        try:
            data = np.fromfile(
                self.fh,
                self.fromfile_dtype,
                count=len(tidx),
                offset=offset   # numpy >= 1.17.0
            )
        except TypeError:
            loc = self.fh.tell()
            self.fh.seek(offset)
            data = np.fromfile(
                self.fh,
                self.fromfile_dtype,
                count=len(tidx)
            )
            self.fh.seek(loc)
        df = pd.DataFrame.from_records(data)
        df.insert(loc=0, column='times', value=self.times[tidx])
        return df

class EspsSdReader(EspsFeaReader):
    '''Read an .sd file (FEA_SD type).

    Channel data is in self.data['ch0'], self.data['ch1'], etc.
    Sample rate is in self.record_freq.

    Examples
    --------

    # Convert a mono .sd file to .wav.
    >>> from espspy.readers import EspsSdReader
    >>> from scipy.io import wavfile
    >>> rdr = EspsSdReader('myfile.sd')
    >>> wavfile.write('myfile.wav', int(rdr.record_freq), rdr.data['ch0'])
    '''
    def __init__(self, infile=None, open_mode='rb', *args, **kwargs):
        super(EspsSdReader, self).__init__(infile=infile,
            open_mode=open_mode, *args, **kwargs)
        # Read the data records.
        self.data = np.fromfile(self.fh, self.fromfile_dtype)
        self.check_data_read()
        self._times = (np.arange(len(self.data)) / self.record_freq) + \
            self.start_time
        self.set_data_view()

    @property
    def fromfile_dtype(self):
        '''The dtypes for unpacking item records using np.fromfile().'''
# TODO: Not sure this is right for all .sd files.
        if self._fromfile_dtype is None or self._fromfile_dtype == []:
            self._fromfile_dtype = []
            flds = []
            dt = {'ndouble': 'f8', 'nfloat': 'f4', 'nlong': 'i4', 'nshort': 'i2'}
            for t,fmt in dt.items():
                for n in np.arange(getattr(self.hdr.feapart2, t)):
                    flds.append((f'ch{len(flds)}', f'{self.byte_order}{fmt}'))
            self._fromfile_dtype = np.dtype(flds)
        return self._fromfile_dtype

    @property
    def samples(self):
        '''Return the sample values of the current data view.'''
        return self.data[:]['re_spec_val'][
            self.data_view['t1idx']:self.data_view['t2idx'],
            self.data_view['hz1idx']:self.data_view['hz2idx']
        ].T

    @property
    def times(self):
        '''Return the timepoints of each spectral slice in sgram of the
        current data view.'''
        return self._times[self.data_view['t1idx']:self.data_view['t2idx']]

    @property
    def record_freq(self):
        '''Return the record_freq value.'''
        return self.hdr.genhd.record_freq

    @property
    def max_value(self):
        '''Return the max_value value.'''
        return self.hdr.genhd.max_value

    @property
    def samples(self):
        '''Return the (number of) samples value.'''
        return self.hdr.genhd.samples

    @property
    def start_time(self):
        '''Return the start_time value.'''
        return self.hdr.genhd.start_time

    @property
    def end_time(self):
        '''Return the end_time value.'''
        return self.hdr.genhd.end_time

    def set_data_view(self, t1=0.0, t2=np.inf):
        '''Set the time range that determines the values returned by data-related
        properties.'''
        t1idx = (np.abs(self._times - t1)).argmin()
        if t2 != np.inf:
            t2idx = (np.abs(self._times - t2)).argmin()
        else:
            t2idx = len(self._times) - 1
        self.data_view = {'t1idx': t1idx, 't2idx': t2idx}
