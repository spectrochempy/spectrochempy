#  =====================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
#  =====================================================================================================================
#

"""
Download data from Google Drive to the local directory


"""

from pathlib import Path
import urllib.request

testdata_url = Path('https://drive.google.com/drive/folders/1rfc9O7jK6v_SbygzJHoFEXXxYY3wIqmh?usp=sharing')

wodger = testdata_url / 'wodger.spg'

try:

    with urllib.request.urlopen(str(testdata_url)) as f:
        file = f.read().decode('utf-8')

except Exception:

    pass
    # TODO: WIP
