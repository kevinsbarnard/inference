# infer.py
# Runs inference on images and stores metadata

import utils
from utils import log
from keras.models import model_from_yaml
from functools import partial
from multiprocessing import Pool
import numpy as np
import configparser
import os


def infer_batch(filename_list, model, size):
    """ Perform inference on batch of images """
    # Prepare input images for processing by network
    with Pool(3) as p:
        input_imgs = p.map(partial(utils.prep_image_for_model, size=size), filename_list)
        p.close()
        p.join()

    # Creating the output vector
    output = model.predict_on_batch(np.asarray(input_imgs))

    # Associate filenames to outputs
    output_dict = {}
    for i, img_path in enumerate(filename_list):
        output_dict[img_path] = output[i]

    return output_dict


def run_inference(filenames, model_file, weights_file, batch_size: int, input_size: tuple) -> dict:
    """ Runs inference batches and constructs dictionary of outputs """
    # Read model file
    with open(model_file, 'r') as model_read:
        model = model_from_yaml(model_read.read())
        model.load_weights(weights_file)
        log('Model and weights initialized.')

    # Run batches, create global output dictionary
    global_dict = {}
    idx = 0
    while idx < len(filenames):
        # Grab batch
        if idx + batch_size > len(filenames) - 1:
            filename_list = filenames[idx:]
        else:
            filename_list = filenames[idx:idx + batch_size]
        idx += batch_size

        # Run batch
        log('Running inference on {} images'.format(len(filename_list)))
        batch_dict = infer_batch(filename_list, model, input_size)

        # Update dictionary
        global_dict.update(batch_dict)

    return global_dict


def main():
    """ Main app """
    cfg = configparser.ConfigParser()
    cfg.read('settings.ini')
    opts = utils.read_config(cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = opts['gpu_device']
    files = utils.get_files(opts['image_dir'])
    input_size = (opts['input_width'], opts['input_height'])
    outputs = run_inference(
        files,
        opts['model_file'], opts['weights_file'],
        opts['batch_size'],
        input_size
    )
    utils.write_outputs(opts['out_dir'], outputs.keys(), outputs.values())


if __name__ == '__main__':
    main()
