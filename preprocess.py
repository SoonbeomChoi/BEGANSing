import sys 
sys.path.append('utils')

import os 
import torch 
import torchaudio 
from madmom.io.midi import load_midi 
from multiprocessing import Pool 

import korean_g2p
import dsp
import file_utils
from config_parser import Config

def preprocess(basename, config):
    basestem = basename.replace('.wav', '')
    wav_filename = os.path.join(config.dataset_path, 'wav', basestem + '.wav')
    mid_filename = os.path.join(config.dataset_path, 'mid', basestem + '.mid')

def main():
    config = Config()

    

