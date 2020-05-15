import sys
sys.path.append('utils')
sys.path.append('model')

import os
import torch
from tqdm import tqdm

import dsp
from config_parser import Config
from torch_utils import set_device, load_weights

import preprocess
import dataprocess
from models import Generator

def main():
    config = Config()

    print("Processing text for \'%s\'." % (config.text_file))
    data = preprocess.preprocess(config.text_file, 'infer', config)
    dataloader = dataprocess.load_infer(data, config)

    G = Generator(config)
    G.load_state_dict(load_weights(config.checkpoint_file))
    G = set_device(G, config.device, config.use_cpu)
    G.eval()

    print("Generating spectrogram with \'%s\'." % (config.checkpoint_file))

    spec = []
    y_prev = torch.zeros(1, config.prev_length, config.fft_size//2 + 1)
    for x in tqdm(dataloader, leave=False, ascii=True):
        x, y_prev = set_device((x, y_prev), config.device, config.use_cpu)

        y_gen = G(x, y_prev)
        y_gen = y_gen.squeeze(1)
        y_prev = y_gen[:, -config.prev_length:,:]
        spec.append(y_gen.data)

    print ("Generating audio with Griffin-Lim algorithm.")
    spec = torch.cat(spec, dim=1).transpose(1, 2) # T x D -> D x T
    wave = dsp.inv_spectrogram(spec, config)

    savename = config.checkpoint_file.replace('.pt', '_') + os.path.basename(config.text_file).replace('.txt', '.wav')
    dsp.save(savename, wave, config.sample_rate)

    print("Audio saved to \'%s\'." % (savename))

if __name__ == "__main__":
    main()