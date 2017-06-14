"""
Command execution, input/output
"""


import argparse
import yaml
import logging
import os
from pprint import pprint

from .data_utils import load_map
from .model import SkyObservation


def load_data(config_doc):
    bands = []
    for dataset in config_doc['datasets']:
        logging.info('Loading dataset {}'.format(dataset['name']))
        path = dataset['path']

        for band in dataset['bands']:
            assert isinstance(band, basestring), 'You need to surround band names with quotes'
            v = {'band': band}
            map_filename = os.path.join(path, dataset['map_template'].format(**v))
            rms_filename = os.path.join(path, dataset['rms_template'].format(**v))
            beam_filename = os.path.join(path, dataset['beam_template'].format(**v))

            logging.info('Loading {}'.format(rms_filename))
            rms = load_map('raw', rms_filename)
            ninv_map = 1 / rms**2
            bands.append(SkyObservation(ninv_map=ninv_map))
    return bands


def load_config_file(filename):
    with open(filename) as f:
        config_doc = yaml.load(f)

    bands = load_data(config_doc)

    
    
            
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parsed_args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    load_config_file(parsed_args.config_file)

    return 0
