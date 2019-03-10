# utils.py
# Miscellaneous utility functions

import configparser
import os
from PIL import Image
import numpy as np
import glob
import json
import ntpath

level_prefix = {
    0: 'info',
    1: 'warning',
    2: 'error'
}


def log(msg, level=0):
    """ Log message to console at specified level """
    if level not in level_prefix.keys():
        raise ValueError('Bad level: {}'.format(level))
    print('[{}] {}'.format(level_prefix[level].upper(), msg))


def get_files(d: str, extension='png'):
    """ Globs all files with specified extension in directory and returns list of file paths """
    if not os.path.isdir(d):
        raise ValueError('Directory {} does not exist.'.format(d))
    glob_path = os.path.join(d, '*{}'.format(extension))
    return glob.glob(glob_path)


def read_config(config: configparser.ConfigParser) -> dict:
    """ Read config object and return dictionary of options """
    opts = {}

    tf_section = config['Tensorflow']
    file_section = config['Files']

    # Read tf section
    opts['gpu_device'] = tf_section.get('GPUDevice')
    opts['batch_size'] = tf_section.getint('BatchSize')
    opts['model_file'] = tf_section.get('ModelFile')
    opts['weights_file'] = tf_section.get('WeightsFile')
    opts['input_width'] = tf_section.getint('InputWidth')
    opts['input_height'] = tf_section.getint('InputHeight')

    # Read file section
    opts['image_dir'] = file_section.get('ImageDir')
    opts['out_dir'] = file_section.get('OutputDir')

    # Check directories, make output if doesn't exist
    if not os.path.isdir(opts['image_dir']):
        raise ValueError('{} is not a valid directory.'.format(opts['image_dir']))
    if not os.path.isdir(opts['out_dir']):
        log('OutDir {} does not exist, creating.'.format(opts['out_dir']))
        os.mkdir(opts['out_dir'])

    return opts


def prep_image_for_model(filename, size):
    """ Open image and make sure format is correct for model """
    # Open and resize image
    with Image.open(filename) as raw_img:
        img = np.array(raw_img.resize(size, Image.BICUBIC), dtype='float')

    # Check number of channels, discard alpha channel
    if img.shape[2] == 4:
        img = img[:, :, 0:3]

    return img


def write_outputs(out_dir, *data):
    """ Write output JSON files to output directory """
    log('Writing JSON outputs for {} images'.format(len(data[0])))
    for datum in zip(*data):
        json_data = make_json(datum)
        fname = ntpath.basename(datum[0])
        with open(os.path.join(out_dir, fname.split('.')[0])+'.json', 'w') as out_file:
            json.dump(json_data, out_file)


def make_json(datum) -> dict:
    """ Generate JSON data dictionary """
    # TODO Handle which data goes where
    json_data = {
        'filename': datum[0],
        'output_vector': datum[1].tolist()
    }

    return json_data
