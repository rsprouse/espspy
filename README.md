# espspy
Python tools for working with ESPS data.

Currently this library only supports reading of `.fb` files created by
`formant` and `rformant` and `.fspec` files created by `sgram`.

This implementation is ALPHA and subject to change.

Currently recommended usage:

```python
from espspy.readers import EspsFormantReader

# Read data in an .fb file into a DataFrame
with EspsFormantReader('formant_meas.fb') as rdr:
    df = rdr.df

# Read an .fspec file.
with EspsSgramReader('sgram.fspec') as rdr:
    # Set time and frequency ranges for data to return.
    rdr.set_data_view(t1=1.0, t2=3.5, hz2=6000)
    sgram = rdr.sgram   # spectrogram data in data view
    times = rdr.times   # timepoints of spectral slices in data view 
    bin_hz = rdr.bin_hz # frequency bins of data view spectrogram
    extent = rdr.data_view_extent  # extent of spectogram data view
```

See the notebooks in the `doc` directory for more detailed usage notes.
