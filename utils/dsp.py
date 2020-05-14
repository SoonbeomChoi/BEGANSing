import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample, Spectrogram, GriffinLim
from scipy import signal

from torch_utils import set_device

# Scale Methods
def time2frame(x, frame_rate):
    return int(x*frame_rate)

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

def preemphasis(x, filter_coefficient):
    x = x.squeeze(0).cpu().numpy()
    x = signal.lfilter([1, -filter_coefficient], [1], x)

    return torch.from_numpy(x).float().unsqueeze(0)

def deemphasis(x, filter_coefficient):
    x = x.squeeze(0).cpu().numpy()
    x = signal.lfilter([1], [1, -filter_coefficient], x)

    return torch.from_numpy(x).float().unsqueeze(0)

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

def save(filename, wave, sample_rate):
    torchaudio.save(filename, wave, sample_rate)

# Spectral Methods
def stft(wave, config): 
    spec_fn = Spectrogram(n_fft=config.fft_size, 
                          win_length=config.win_size, 
                          hop_length=config.hop_size)
    wave, spec_fn = set_device((wave, spec_fn), config.device, config.use_cpu)
    spec = torch.sqrt(spec_fn(wave))

    return spec

def spectrogram(wave, config, squeeze=True):
    wave = preemphasis(wave, config.preemphasis)
    spec = stft(wave, config)
    spec = amp2db(spec)
    spec = normalize(spec, config.min_db, config.max_db, config.clip_val)

    if squeeze:
        spec = spec.squeeze(0)
    
    return spec

def inv_spectrogram(spec, config):
    griffin_lim = GriffinLim(n_fft=config.fft_size,
                             win_length=config.win_size,
                             hop_length=config.hop_size,
                             n_iter=60)

    spec, griffin_lim = set_device((spec, griffin_lim), config.device, config.use_cpu)

    spec = db2amp(denormalize(spec, config.min_db, config.max_db, config.clip_val))
    wave = griffin_lim(spec**config.spec_power)
    wave = deemphasis(wave, config.preemphasis)

    return wave
