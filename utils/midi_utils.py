"""
This file is for loading MIDI and it replicates madmom.io.midi just to reduce dependency.
"""

import numpy as np
import mido
import warnings

DEFAULT_TEMPO = 500000
DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_TIME_SIGNATURE = (4, 4)

def tick2second(tick, ticks_per_beat=DEFAULT_TICKS_PER_BEAT,
                tempo=DEFAULT_TEMPO):
    # Note: both tempo (microseconds) and ticks are per quarter note
    #       thus the time signature is irrelevant
    scale = tempo*1e-6/ticks_per_beat

    return tick*scale

def tick2beat(tick, ticks_per_beat=DEFAULT_TICKS_PER_BEAT,
              time_signature=DEFAULT_TIME_SIGNATURE):
    return tick/(4.0*ticks_per_beat/time_signature[1])

def note_hash(channel, pitch):
    return channel*128 + pitch

class MIDIFile(mido.MidiFile):
    def __init__(self, filename=None, file_format=0, ticks_per_beat=DEFAULT_TICKS_PER_BEAT, 
                 unit='seconds', timing='absolute', **kwargs):
        super(MIDIFile, self).__init__(filename=filename, type=file_format,
                                       ticks_per_beat=ticks_per_beat, **kwargs)
        self.unit = unit
        self.timing = timing

    def __iter__(self):
        if self.type == 2:
            raise TypeError("can't merge tracks in type 2 (asynchronous) file")

        tempo = DEFAULT_TEMPO
        time_signature = DEFAULT_TIME_SIGNATURE
        cum_delta = 0
        for msg in mido.merge_tracks(self.tracks):
            # Convert relative message time to desired unit
            if msg.time > 0:
                if self.unit.lower() in ('t', 'ticks'):
                    delta = msg.time
                elif self.unit.lower() in ('s', 'sec', 'seconds'):
                    delta = tick2second(msg.time, self.ticks_per_beat, tempo)
                elif self.unit.lower() in ('b', 'beats'):
                    delta = tick2beat(msg.time, self.ticks_per_beat,
                                      time_signature)
                else:
                    raise ValueError("`unit` must be either 'ticks', 't', "
                                     "'seconds', 's', 'beats', 'b', not %s." %
                                     self.unit)
            else:
                delta = 0
            # Convert relative time to absolute values if needed
            if self.timing.lower() in ('a', 'abs', 'absolute'):
                cum_delta += delta
            elif self.timing.lower() in ('r', 'rel', 'relative'):
                cum_delta = delta
            else:
                raise ValueError("`timing` must be either 'relative', 'rel', "
                                 "'r', or 'absolute', 'abs', 'a', not %s." %
                                 self.timing)

            yield msg.copy(time=cum_delta)

            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'time_signature':
                time_signature = (msg.numerator, msg.denominator)

def load_midi(filename, file_format=0, ticks_per_beat=DEFAULT_TICKS_PER_BEAT, unit='seconds', timing='absolute'):
    midi = MIDIFile(filename=filename, file_format=file_format, ticks_per_beat=ticks_per_beat, unit=unit, timing=timing)

    notes = []
    sounding_notes = {}

    for msg in midi:
        note_on = msg.type == 'note_on'
        note_off = msg.type == 'note_off'
        if not (note_on or note_off):
            continue

        note = note_hash(msg.channel, msg.note)

        if note_on and msg.velocity > 0:
            sounding_notes[note] = (msg.time, msg.velocity)

        elif note_off or (note_on and msg.velocity == 0):
            if note not in sounding_notes:
                warnings.warn('ignoring MIDI message %s' % msg)
                continue

            notes.append((sounding_notes[note][0], msg.note,
                          msg.time - sounding_notes[note][0],
                          sounding_notes[note][1], msg.channel))

            del sounding_notes[note]

    return np.asarray(sorted(notes), dtype=np.float)