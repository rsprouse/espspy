# espspy
Python tools for working with ESPS data.

Currently this library only supports reading of .fb files created by
`formant` and `rformant`.

This implementation is ALPHA. Please do not write any code that uses
Reader attributes/methods other than the `.data` attribute.

Recommended usage:

```python
from espspy.readers import EspsFormantReader

# Read data in an .fb file into a DataFrame
with EspsFormantReader('formant_meas.fb') as rdr:
    df = rdr.data

```
