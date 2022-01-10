import json
import base64
import pickle
import numpy as np

from spectrochempy.utils import json_serialiser, json_decoder


def test_json_serialiser_decoder(IR_dataset_2D):
    nd = IR_dataset_2D.copy()

    # make a json string to write (without encoding)

    js = json_serialiser(nd, encoding=None)
    js_string = json.dumps(js, indent=2)
    print("no encoding", len(js_string))

    # load json from string
    jsd = json.loads(js_string, object_hook=json_decoder)

    assert np.all(np.array(js["data"]["tolist"]) == jsd["data"])

    # encoding  base 64
    js = json_serialiser(nd, encoding="base64")
    js_string = json.dumps(js, indent=2)
    print("base64", len(js_string))

    # load json from string
    jsd = json.loads(js_string, object_hook=json_decoder)

    assert np.all(pickle.loads(base64.b64decode(js["data"]["base64"])) == jsd["data"])
