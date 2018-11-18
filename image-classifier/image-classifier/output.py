#!/usr/bin/env python

import json
from tensorflow.python.lib.io import file_io


def save(model, params):
    basename = "model-{}.hdf5".format(params.current_date)
    model.save(basename)

    with open(basename, "r") as input:
        with file_io.FileIO(params.model_filename, 'w+') as f:
            f.write(input.read())

    file_io.delete_file(basename)
    print("model saved to GS: {}".format(params.model_filename))
