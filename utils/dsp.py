import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample, Spectrogram

from torch_utils import set_device

# Scale Methods
def time2frame(x, frame_rate):
    return int(x*frame_rate)

def frame(x, win_size, hop_size):
    if x.dim() == 1:
        num_frame = (x.size(0) - win_size)//hop_size + 1
        y = x.new_zeros(win_size, num_frame)
        for i in range(num_frame):
            y[:,i] = x[hop_size*i:hop_size*i + win_size]
    elif x.dim() == 2:
        num_frame = (x.size(1) - win_size)//hop_size + 1
        y = x.new_zeros(x.size(0), win_size, num_frame)
        for i in range(x.size(0)):
            for j in range(num_frame):
                y[i,:,j] = x[i, hop_size*j:hop_size*j + win_size]
    else:
        raise AssertionError("Input dimension should be 1 or 2")

    return y

def to_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float)

    return x 

def amp2db(x, min_level_db=None):
    x = to_tensor(x)

    x = 20.0*torch.log10(x)
    if min_level_db is not None:
        x = torch.max(x, min_level_db)

    return x 

def db2amp(x):
    x = to_tensor(x)

    return torch.pow(10.0, x*0.05)

def hz2midi(x):
    x = to_tensor(x)

    return 69.0 + 12.0*torch.log2(x/440.0)

def midi2hz(x):
    x = to_tensor(x)

    return 440.0*torch.pow(2.0, (x - 69)/12)

def preemphasis(x, preemphasis):
    return torchaudio.functional.lfilter(x, [1, -preemphasis], [1])

def inv_preemphasis(x, preemphasis):
    return torchaudio.functional.lfilter(x, [1], [1, -preemphasis])

def normalize(x, min_db, max_db, clip_val):
    x = 2.0*(x - min_db)/(max_db - min_db) - 1.0
    x = torch.clamp(clip_val*x, -clip_val, clip_val)

    return x

def denormalize(x, min_db, max_db, clip_val):
    x = x/clip_val
    x = (max_db - min_db)*(x + 1.0)/2.0 + min_db

    return x

# Wave Methods
def load(filename, sample_rate):
    y, source_rate = torchaudio.load(filename)
    if source_rate != sample_rate:
        resample = Resample(source_rate, sample_rate)
        y = resample(y)

    return y 

# Spectral Methods
def stft(y, config): 
    spec_fn = Spectrogram(n_fft=config.fft_size, 
                          win_length=config.win_size, 
                          hop_length=config.hop_size)
    y, spec_fn = set_device((y, spec_fn), config.device, config.use_cpu)
    spec = torch.sqrt(spec_fn(y))

    return spec

def istft(magnitude, phase, config):
    window = torch.hann_window(config.win_size)
    stft_matrix = torch.stack((magnitude*torch.cos(phase), magnitude*torch.sin(phase)), dim=-1)
    stft_matrix, window = set_device((stft_matrix, window), config.device, config.use_cpu)
    y = torchaudio.functional.istft(stft_matrix,
                                    n_fft=config.fft_size,
                                    hop_length=config.hop_size,
                                    win_length=config.win_size,
                                    window=window)

    return y

def spectrogram(y, config, squeeze=True):
    spec = stft(y, config)
    spec = amp2db(spec)
    spec = normalize(spec, config.min_db, config.max_db, config.clip_val)

    if squeeze:
        spec = spec.squeeze(0)
    
    return spec