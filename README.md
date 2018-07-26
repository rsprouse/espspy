# espspy
Python tools for working with ESPS data.

Currently this library only supports reading of .fb files created by
`formant` and `rformant`.

```python
from espspy.readers import EspsFormantReader

# Read data in an .fb file into a DataFrame
with EspsFormantReader('formant_meas.fb') as rdr:
    df = rdr.data

```
