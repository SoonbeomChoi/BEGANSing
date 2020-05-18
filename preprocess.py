import sys 
sys.path.append('utils')
sys.path.append('g2p')

import os 
import torch 
import torchaudio 
from multiprocessing import Pool 
from functools import partial

import korean_g2p
import dsp
from config_parser import Config
from file_utils import create_path
from midi_utils import load_midi

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

def files4train(filename, config):
    basename = os.path.basename(filename)
    type_list = ['txt', 'mid', 'wav']
    file_list = []
    for t in type_list:
        if config.file_structure == 1:
            f = os.path.join(config.dataset_path, basename + '.' + t)
        elif config.file_structure == 2:
            f = os.path.join(config.dataset_path, t, basename + '.' + t)
        else:
            raise AssertionError("There is no file structure type for %s" % (config.file_structure))
        
        file_list.append(f)

    return file_list

def files4infer(filename, config):
    basename = os.path.basename(filename).replace('.txt', '')
    filepath = os.path.dirname(filename)
    if config.file_structure == 2:
        filepath = '/'.join(filepath.split('/')[:-1])

    type_list = ['txt', 'mid']
    file_list = []
    for t in type_list:
        if config.file_structure == 1:
            f = os.path.join(filepath, basename + '.' + t)
        elif config.file_structure == 2:
            f = os.path.join(filepath, t, basename + '.' + t)
        else:
            raise AssertionError("There is no file structure type for %s" % (config.file_structure))

        file_list.append(f)

    return file_list

def zero_pad(x, pad_length):
    pad = x.new_zeros(pad_length)

    return torch.cat((x, pad))

def preprocess(filename, set_type, config):
    infer = set_type is 'infer'
    if not infer:
        txt_file, mid_file, wav_file = files4train(filename, config)
    else:
        txt_file, mid_file = files4infer(filename, config)

    text = load_text(txt_file)
    note = load_midi(mid_file)
    text, note = align_label(text, note, config)

    # Zero pad to make 1 more iteration
    data_stride = config.spec_length
    if text.size(0)%data_stride != 0:
        pad_length = (text.size(0)//data_stride + 1)*data_stride - text.size(0)
        text = zero_pad(text, pad_length)
        note = zero_pad(note, pad_length)

    num_stride = text.size(0)//data_stride

    if not infer:
        wave = dsp.load(wav_file, config.sample_rate)
        spec = dsp.spectrogram(wave, config).cpu().transpose(0, 1) # D x T -> T x D
        spec = torch.cat((torch.zeros(config.prev_length, config.fft_size//2 + 1), spec))

        min_length = min(text.size(0), spec.size(0))
        data_stride = config.data_stride
        num_stride = (min_length - (config.spec_length + config.prev_length))//data_stride
    
    data_list = []
    for i in range(num_stride):
        text_start = i*data_stride
        t = text[text_start:text_start + config.spec_length]
        n = note[text_start:text_start + config.spec_length]

        data = dict(text=t, note=n)

        if not infer:
            spec_start = i*data_stride + config.prev_length
            s = spec[spec_start:spec_start + config.spec_length]
            s_prev = spec[spec_start - config.prev_length:spec_start]

            data = dict(text=t, note=n, spec_prev=s_prev, spec=s)

        data_list.append(data)

    if not infer:
        basename = os.path.basename(filename)
        savename = os.path.join(config.feature_path, set_type, basename + '.pt')
        torch.save(data_list, savename)
        print(basename)

    return data_list

def read_file_list(filename):
    with open(filename) as f:
        file_list = f.read().split('\n')

    return file_list

def make_indices(path):
    num_features = []
    for f in sorted(os.listdir(path)):
        num_feature = len(torch.load(os.path.join(path, f)))
        num_features.append(num_feature)

    num_features = torch.tensor(num_features)
    file_indices = torch.cumsum(num_features, dim=0)

    return file_indices

def main():
    config = Config()
    config_basename = os.path.basename(config.configs[0])
    print("Configuration file: \'%s\'" % (config_basename))

    set_list = ['train', 'valid']
    file_list = {}

    # Creating Path for Features
    create_path(config.feature_path, action='overwrite', verbose=False)
    for set_type in set_list:
        path = os.path.join(config.feature_path, set_type)
        create_path(path, action='overwrite')

        list_file = set_type + '_list.txt'
        file_list[set_type] = read_file_list(os.path.join(config.dataset_path, list_file))

    # Extracting Features
    if config.num_proc > 1:
        if config.use_cpu is False:
            raise AssertionError("You can not use GPU with multiprocessing.")

        p = Pool(config.num_proc)
        for set_type in set_list:
            p.map(partial(preprocess, set_type=set_type, config=config), file_list[set_type])
    else:
        for set_type in set_list:
            [preprocess(f, set_type=set_type, config=config) for f in file_list[set_type]]

    # Creating Files Indices
    for set_type in set_list:
        path = os.path.join(config.feature_path, set_type)
        file_indices = make_indices(path)
        torch.save(file_indices, os.path.join(config.feature_path, set_type + '_indices.pt'))

    print("Feature saved to \'%s\'." % (config.feature_path))

if __name__ == "__main__":
    main()