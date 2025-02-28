# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
import base64
import json
import pickle

import numpy as np

from spectrochempy.utils.jsonutils import json_decoder, json_encoder


def test_json_encoder_decoder(IR_dataset_2D):
    nd = IR_dataset_2D.copy()

    # make a json string to write (without encoding)

    js = json_encoder(nd, encoding=None)
    js_string = json.dumps(js, indent=2)
    print("no encoding", len(js_string))

    # load json from string
    jsd = json.loads(js_string, object_hook=json_decoder)

    assert np.all(np.array(js["data"]["tolist"]) == jsd["data"])

    # encoding  base 64
    js = json_encoder(nd, encoding="base64")
    js_string = json.dumps(js, indent=2)
    print("base64", len(js_string))

    # load json from string
    jsd = json.loads(js_string, object_hook=json_decoder)

    assert np.all(pickle.loads(base64.b64decode(js["data"]["base64"])) == jsd["data"])
