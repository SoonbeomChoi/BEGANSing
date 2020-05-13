import sys 
sys.path.append('utils')
sys.path.append('g2p')

import os 
import torch 
import torchaudio 
from madmom.io.midi import load_midi 
from multiprocessing import Pool 
from functools import partial

import korean_g2p
import dsp
import file_utils
from config_parser import Config

def load_text(filename):
    text_file = open(filename)
    text = text_file.read().replace(' ', '').replace('\n', '')
    text = korean_g2p.encode(text)

    return text

def get_phoneme_duration(phone, note_duration, length_c):
    duration = []
    if note_duration < phone.num():
        length_c = 0
    elif note_duration <= phone.num()*length_c:
        length_c = max(note_duration//phone.num() - 1, 1)

    if phone.ons is not None:
        duration.append(length_c)
    if phone.nuc is not None:
        length_v = note_duration - (phone.num() - 1)*length_c
        duration.append(length_v)
    if phone.cod is not None:
        duration.append(length_c)

    return torch.tensor(duration)

def align_label(text, note, config):
    frame_rate = config.sample_rate/config.hop_size
    aligned_length = dsp.time2frame(note[-1][0] + note[-1][2], frame_rate=frame_rate)
    text_aligned = torch.zeros(aligned_length, dtype=torch.long)
    note_aligned = torch.zeros(aligned_length, dtype=torch.long)

    for i in range(len(note)):
        start = dsp.time2frame(note[i][0], frame_rate=frame_rate)
        end = dsp.time2frame(note[i][0] + note[i][2], frame_rate=frame_rate)
        duration = get_phoneme_duration(text[i], end - start, config.length_c)

        phone = torch.tensor(text[i].to_list())
        text_aligned[start:end] = phone.repeat_interleave(duration, dim=0)
        note_aligned[start:end] = note[i][1] - config.min_note

    return text_aligned, note_aligned

def preprocess(filename, set_type, config):
    basestem = os.path.basename(filename).replace('.wav', '')
    txt_filename = os.path.join(config.dataset_path, 'txt', basestem + '.txt')
    mid_filename = os.path.join(config.dataset_path, 'mid', basestem + '.mid')
    wav_filename = os.path.join(config.dataset_path, 'wav', basestem + '.wav')

    text = load_text(txt_filename)
    note = load_midi(mid_filename)
    text, note = align_label(text, note, config)

    wave = dsp.load(wav_filename, config.sample_rate)
    spec = dsp.spectrogram(wave, config).cpu().transpose(0, 1) # D x T -> T x D
    spec = torch.cat((torch.zeros(config.prev_length, config.fft_size//2 + 1), spec))

    min_length = min(note.size(0), spec.size(0))
    num_stride = (min_length - (config.spec_length + config.prev_length))//config.data_stride

    data_list = []
    for i in range(num_stride):
        text_start = i*config.data_stride
        spec_start = i*config.data_stride + config.prev_length

        t = text[text_start:text_start + config.spec_length]
        n = note[text_start:text_start + config.spec_length]
        s = spec[spec_start:spec_start + config.spec_length]
        s_prev = spec[spec_start - config.prev_length:spec_start]

        data = dict(text=t, note=n, spec_prev=s_prev, spec=s)
        data_list.append(data)

    savename = os.path.join(config.feature_path, set_type, basestem + '.pt')
    torch.save(data_list, savename)

    print(basestem)

def read_file_list(filename):
    with open(filename) as f:
        file_list = f.read().split('\n')

    return file_list

def main():
    config = Config()
    set_list = ['train', 'valid']
    file_list = {}
    for set_type in set_list:
        path = os.path.join(config.feature_path, set_type)
        file_utils.create_path(path, action='overwrite')

        list_file = set_type + '_list.txt'
        file_list[set_type] = read_file_list(os.path.join(config.dataset_path, list_file))

    if config.num_proc > 1:
        p = Pool(config.num_proc)
        for set_type in set_list:
            p.map(partial(preprocess, set_type=set_type, config=config), file_list[set_type])
    else:
        for set_type in set_list:
            [preprocess(f, set_type=set_type, config=config) for f in file_list[set_type]]

    print("Feature saved to %s" % (config.feature_path))

if __name__ == "__main__":
    main()