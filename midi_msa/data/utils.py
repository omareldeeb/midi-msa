import bisect
import collections
import os.path
import statistics
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import mido
import numpy as np
from scipy.ndimage import maximum_filter1d
import scipy.fftpack
import torch
import torch.nn.functional as F

from . import midisong as ms


@dataclass
class PatchData:
    piano_rolls: List[torch.Tensor]
    patch_metadata: Dict
    sslm_near_patches: Optional[List[torch.Tensor]] = None
    sslm_far_patches: Optional[List[torch.Tensor]] = None


def parse_midi(file_path: Union[Path, str]):
    """
    Parse a MIDI file into a list of bar segments per track.
    A bar segment is defined as a list of MIDI messages encoded as tuples that fit into a single bar.
    A tuple is defined as (time, note, velocity, duration, channel, program)
    Also extracts time signature information from the MIDI file.
    """
    midi = mido.MidiFile(file_path, clip=True)

    track_data = {
        (track.name if track.name else f"track_{idx}"): []
        for idx, track in enumerate(midi.tracks)
    }

    channel_volumes = {
        i: 127
        for i in range(16)
    }
    channel_expressions = {
        i: 127
        for i in range(16)
    }
    channel_instruments = {
        i: 0
        for i in range(16)
    }

    # Store time signatures with their tick positions
    # Default to 4/4 if no time signature is found
    time_signatures = [(0, 4, 4)]  # (tick_position, numerator, denominator)

    for idx, track in enumerate(midi.tracks):
        track_name = track.name if track.name else f"track_{idx}"
        current_ticks = 0
        for msg in track:
            current_ticks += msg.time
            if msg.type == "time_signature":
                # Store time signature change with current tick position
                time_signatures.append((current_ticks, msg.numerator, msg.denominator))
            elif msg.type == "control_change":
                if msg.control == 7:
                    channel_volumes[msg.channel] = msg.value
                elif msg.control == 11:
                    channel_expressions[msg.channel] = msg.value
            elif msg.type == "program_change":
                channel_instruments[msg.channel] = msg.program
            elif msg.type == "marker":
                pass
            elif msg.type == "note_on" and msg.velocity > 0:
                velocity = msg.velocity * (channel_volumes[msg.channel] / 127.) * (
                        channel_expressions[msg.channel] / 127.)
                program = channel_instruments[msg.channel]
                track_data[track_name].append({
                    "time": current_ticks,
                    "note": msg.note,
                    "velocity": velocity,
                    "duration": -1,
                    "channel": msg.channel,
                    "program": program
                })
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                for note in track_data[track_name]:
                    if note["duration"] == -1 and note["note"] == msg.note and note["channel"] == msg.channel:
                        note["duration"] = current_ticks - note["time"]
                        break

    return track_data, midi.ticks_per_beat, time_signatures


def parse_markers(markers_qn_path: Union[Path, str], file_id: str, ticks_per_beat: int) -> List[int]:
    """
    Parse markers from a JSON file and convert them to ticks based on the MIDI ticks per beat.
    The markers are expected to be in the format:
    {
        "file_id": [[qn1, label1], [qn2, label2], ...]
    }
    where qn is a quarter note position.
    The output is a list of unique tick positions sorted in ascending order.
    """
    # file_id = os.path.basename(file_path).split('.mid')[0]
    with open(markers_qn_path, 'r') as f:
        markers = json.load(f)

    marker_qns = markers[file_id]
    markers_ticks = [int(round(x[0] * ticks_per_beat)) for x in marker_qns if not x[1].startswith('Fadeout')]

    markers_ticks = list(set(markers_ticks))
    markers_ticks.sort()

    return markers_ticks


def instrument_overtone_intensities(program, num_harmonics=3, max_harmonic=5):
    """
    Generate a set of harmonics and their intensities for a given instrument program.
    The harmonics are random but fixed for a given program.
    """
    original_seed = np.random.get_state()[1][0]  # Save the original random seed

    np.random.seed(hash(str(program)) % 2 ** 32)

    harmonics = np.sort(np.random.choice(max_harmonic, num_harmonics, replace=False) + 2)
    intensities = np.sort(np.random.rand(num_harmonics))[::-1]

    # Return to original seed
    np.random.seed(original_seed)

    return harmonics, intensities


def hz_to_midi(frequency):
    if frequency <= 0:
        raise ValueError("Frequency must be greater than 0 Hz.")
    return 69 + 12 * np.log2(frequency / 440.0)


def midi_to_hz(midi_note):
    return 440.0 * 2 ** ((midi_note - 69) / 12)


def normalize_channel(C: np.array):
    """C a tensor of shape (H, W)"""
    C_min = C.min()
    C_max = C.max()
    denom = C_max - C_min

    if denom == 0:
        return C

    return (C - C_min) / denom


def create_piano_roll_fast(path_to_midi_file,
                           chroma=False,
                           target_ticks_per_beat=4,
                           max_target_ticks=40960,
                           clip=True,
                           normalize=False,
                           compute_measure_ticks=True):
    def get_avg_cc_7_and_cc_11_by_click(tr: ms.Track) -> dict[int, dict[int, float]]:
        to_avg = {7: collections.defaultdict(list),
                  11: collections.defaultdict(list)}

        for cc in tr.ccs:
            if cc.cc in (7, 11):
                to_avg[cc.cc][cc.click].append(cc.val)

        res = {7: {},
               11: {}}

        for cc in [7, 11]:
            for click, L in to_avg[cc].items():
                res[cc][click] = statistics.mean(L)
            if 0 not in res[cc]:
                res[cc][0] = 127
            # then map to [0.0, 1.0]
            for click, val in res[cc].items():
                res[cc][click] = val / 127

        return res

    S = ms.MidiSong.from_midi_file(path_to_file=path_to_midi_file, clean_up_time_signatures=False)
    S.apply_pedals_to_extend_note_lengths()
    # S.remove_pedals()

    # S = ms.MidiSongByMeasure.from_MidiSong(S, consume_calling_song=True)
    # S.quantize_notes_by_measure(q=(target_ticks_per_beat,))
    # S.change_cpq(to_cpq=target_ticks_per_beat)

    S.quantize_notes(q=(target_ticks_per_beat, ))
    S.change_cpq(to_cpq=target_ticks_per_beat)

    num_pitches = 12 if chroma else 128

    # could add another 4 * target_ticks_per_beat for safety
    # duration_ticks = S.get_measure_endpoints(make_copy=False)[-1]
    # measure_ticks = S.get_measure_endpoints(make_copy=True)

    duration_ticks = 1
    for t in S.tracks:
        for n in t.notes:
            if n.click == n.end:
                duration_ticks = max(duration_ticks, n.end + 1)
            else:
                duration_ticks = max(duration_ticks, n.end)

    # print('here', S.cpq, S.max_click_on_init, S.get_max_click(), duration_ticks)

    if duration_ticks > max_target_ticks:
        raise OverflowError(f"File {path_to_midi_file} has {duration_ticks} ticks "
                            f"(at {target_ticks_per_beat} ticks per beat), which is too large. "
                            f"Increase max_target_ticks to avoid this error.")

    # S = ms.MidiSong.from_MidiSongByMeasure(S, consume_calling_song=True, clean_up_time_signatures=False)
    time_sigs = [(t.click, t.num, t.denom) for t in S.time_signatures]

    # ch 0: raw piano roll, ch 1: piano roll w/cc modifications, ch 2: overtones based on ch 1, ch 3: drums
    piano_roll = np.zeros((4, num_pitches, duration_ticks))

    for i, t in enumerate(S.tracks):
        cc_7_and_11_mults_by_click = get_avg_cc_7_and_cc_11_by_click(tr=t)
        cc_clicks = {7: list(cc_7_and_11_mults_by_click[7].keys()),
                     11: list(cc_7_and_11_mults_by_click[11].keys())}
        cc_clicks[7].sort()
        cc_clicks[11].sort()
        for n in t.notes:
            pitch = n.pitch
            vel = n.vel / 127
            click = n.click
            if t.is_drum:
                end = n.click + 1
            else:
                end = n.end if n.end != n.click else n.click + 1

            cc_vel_mult = 1.0
            for cc in cc_clicks:
                click_i = bisect.bisect_right(cc_clicks[cc], click) - 1
                cc_vel_mult *= cc_7_and_11_mults_by_click[cc][cc_clicks[cc][click_i]]

            if t.is_drum:
                piano_roll[3, pitch % num_pitches, click] = vel * cc_vel_mult
            else:
                piano_roll[0, pitch % num_pitches, click: end] += vel
                piano_roll[1, pitch % num_pitches, click: end] += vel * cc_vel_mult
                harmonics, intensities = instrument_overtone_intensities_fast(program=t.inst)
                max_intensity = intensities[0]
                for harmonic, intensity in zip(harmonics, intensities):
                    overtone_pitch = pitch + harmonic
                    if overtone_pitch <= 127 or chroma or True:  # just let the overtones wrap around
                        decay = np.linspace(1.0, 0.1, end - click) * intensity / max_intensity
                        piano_roll[2, overtone_pitch % num_pitches, click: end] += vel * cc_vel_mult * decay

    # now clip and/or normalize all channels
    if clip:
        piano_roll = np.clip(piano_roll, 0.0, 1.0)
    if normalize:
        for i in range(piano_roll.shape[0]):
            piano_roll[i] = normalize_channel(piano_roll[i])

    if compute_measure_ticks:
        S.remove_pedals()
        S = ms.MidiSongByMeasure.from_MidiSong(S, consume_calling_song=True)
        measure_ticks = S.get_measure_endpoints(make_copy=False)
    else:
        measure_ticks = None

    # now consolidate our 4 channels to form our output
    # Always return 3 channels: non-drums, overtones, drums
    res = {'piano_roll': np.stack([piano_roll[1],
                                   piano_roll[2],
                                   piano_roll[3]]),
           'time_signatures': time_sigs,
           'measure_ticks': measure_ticks
           }

    return res


OVERTONES = {
    0: ([4, 5, 6], np.array([0.46164491, 0.28045567, 0.0701171])),
    1: ([2, 5, 6], np.array([0.5360616, 0.51915406, 0.27246444])),
    2: ([2, 5, 6], np.array([0.59407944, 0.16482702, 0.07510001])),
    3: ([2, 5, 6], np.array([0.42268757, 0.28799927, 0.10433312])),
    4: ([2, 3, 6], np.array([0.56370442, 0.53071473, 0.34982111])),
    5: ([3, 4, 6], np.array([0.93670124, 0.63314628, 0.45107436])),
    6: ([3, 4, 6], np.array([0.79209567, 0.55634835, 0.54239875])),
    7: ([2, 3, 6], np.array([0.48518155, 0.43796504, 0.22272311])),
    8: ([2, 5, 6], np.array([0.73324229, 0.53235516, 0.20888515])),
    9: ([4, 5, 6], np.array([0.91070995, 0.44879788, 0.38392218])),
    10: ([2, 3, 5], np.array([0.46424573, 0.38643153, 0.31335952])),
    11: ([2, 3, 4], np.array([0.25880494, 0.17710375, 0.00621956])),
    12: ([2, 3, 6], np.array([0.60351133, 0.25328097, 0.03207099])),
    13: ([2, 4, 6], np.array([0.78438012, 0.63085452, 0.01354107])),
    14: ([2, 4, 5], np.array([0.83866864, 0.48730866, 0.09652502])),
    15: ([2, 3, 5], np.array([0.88933973, 0.30377682, 0.28402545])),
    16: ([2, 5, 6], np.array([0.80784898, 0.66218991, 0.52689169])),
    17: ([2, 3, 5], np.array([0.65254347, 0.40694587, 0.40514385])),
    18: ([2, 3, 6], np.array([0.85110388, 0.52192566, 0.51563116])),
    19: ([2, 4, 5], np.array([0.73327607, 0.12879316, 0.02455808])),
    20: ([2, 3, 5], np.array([0.9170311, 0.260518, 0.17624925])),
    21: ([3, 4, 5], np.array([0.94045792, 0.5678866, 0.40389296])),
    22: ([2, 5, 6], np.array([0.83253324, 0.36225289, 0.3348712])),
    23: ([2, 4, 5], np.array([0.77491917, 0.10814883, 0.01873242])),
    24: ([2, 3, 5], np.array([0.99248003, 0.25006502, 0.06380248])),
    25: ([2, 4, 6], np.array([0.85634731, 0.69133266, 0.66679565])),
    26: ([2, 3, 6], np.array([0.94494218, 0.21341968, 0.0799861])),
    27: ([3, 4, 5], np.array([0.43548817, 0.41182802, 0.31285364])),
    28: ([3, 4, 5], np.array([0.97285518, 0.63609732, 0.53861494])),
    29: ([3, 5, 6], np.array([0.66219386, 0.51658791, 0.25434071])),
    30: ([4, 5, 6], np.array([0.23360433, 0.16932574, 0.12126467])),
    31: ([4, 5, 6], np.array([0.61581971, 0.30751058, 0.23698217])),
    32: ([2, 5, 6], np.array([0.76227044, 0.45444219, 0.15125884])),
    33: ([2, 3, 6], np.array([0.46159215, 0.36442946, 0.23730151])),
    34: ([2, 3, 4], np.array([0.80711012, 0.14154419, 0.03367026])),
    35: ([2, 4, 5], np.array([0.85007268, 0.70231566, 0.25870182])),
    36: ([2, 5, 6], np.array([0.76041836, 0.31754424, 0.0873672])),
    37: ([2, 4, 6], np.array([0.82298704, 0.10561976, 0.01588093])),
    38: ([4, 5, 6], np.array([0.68295955, 0.16562422, 0.01373179])),
    39: ([3, 4, 6], np.array([0.77265098, 0.58177045, 0.30735203])),
    40: ([2, 3, 4], np.array([0.96971145, 0.92151616, 0.30636454])),
    41: ([2, 3, 4], np.array([0.3148555, 0.10672426, 0.06051311])),
    42: ([4, 5, 6], np.array([0.95385309, 0.45539437, 0.0530052])),
    43: ([2, 3, 5], np.array([0.81805836, 0.54606386, 0.26567235])),
    44: ([3, 5, 6], np.array([0.93798322, 0.7191844, 0.24256421])),
    45: ([4, 5, 6], np.array([0.48174257, 0.06529793, 0.01904464])),
    46: ([3, 5, 6], np.array([0.71770108, 0.46667047, 0.35810488])),
    47: ([2, 3, 5], np.array([0.71498345, 0.5763956, 0.2628946])),
    48: ([3, 5, 6], np.array([0.95302712, 0.83247846, 0.72707413])),
    49: ([3, 4, 6], np.array([0.87360863, 0.5966814, 0.1488981])),
    50: ([2, 5, 6], np.array([0.42568672, 0.28072509, 0.12897304])),
    51: ([2, 4, 6], np.array([0.84030862, 0.56278435, 0.50507829])),
    52: ([2, 4, 6], np.array([0.70228594, 0.41955147, 0.00750306])),
    53: ([2, 4, 5], np.array([0.80450214, 0.34014752, 0.1390039])),
    54: ([2, 5, 6], np.array([0.63494215, 0.62976952, 0.27102382])),
    55: ([3, 5, 6], np.array([0.89451759, 0.58507109, 0.43512873])),
    56: ([4, 5, 6], np.array([0.9075779, 0.7272653, 0.4872161])),
    57: ([2, 4, 5], np.array([0.96299157, 0.44701006, 0.08684755])),
    58: ([2, 4, 5], np.array([0.93944345, 0.68986451, 0.24678895])),
    59: ([2, 4, 6], np.array([0.97092787, 0.96581408, 0.18029256])),
    60: ([4, 5, 6], np.array([0.56399969, 0.4410507, 0.22521578])),
    61: ([4, 5, 6], np.array([0.24223235, 0.13051106, 0.06596591])),
    62: ([2, 3, 4], np.array([0.81340061, 0.61367299, 0.02009184])),
    63: ([3, 4, 6], np.array([0.72220041, 0.63448419, 0.00391132])),
    64: ([3, 4, 5], np.array([0.63248872, 0.41237143, 0.18752739])),
    65: ([2, 4, 5], np.array([0.6852431, 0.41204937, 0.12641334])),
    66: ([2, 3, 5], np.array([0.98855623, 0.85317148, 0.07977359])),
    67: ([3, 4, 5], np.array([0.93709134, 0.81039548, 0.52975649])),
    68: ([2, 3, 5], np.array([0.80742839, 0.2593828, 0.25664508])),
    69: ([2, 4, 6], np.array([0.84197294, 0.71467849, 0.54253114])),
    70: ([3, 5, 6], np.array([0.87877068, 0.72535176, 0.20880191])),
    71: ([2, 3, 6], np.array([0.93968964, 0.74876745, 0.51348945])),
    72: ([2, 3, 6], np.array([0.97144782, 0.9624782, 0.26084589])),
    73: ([4, 5, 6], np.array([0.68313948, 0.51315304, 0.44797168])),
    74: ([2, 4, 6], np.array([0.40823519, 0.17894515, 0.02850354])),
    75: ([2, 5, 6], np.array([0.54786214, 0.54446204, 0.39649967])),
    76: ([2, 4, 5], np.array([0.99274053, 0.86763765, 0.59237642])),
    77: ([2, 4, 5], np.array([0.73447376, 0.48039706, 0.43327863])),
    78: ([2, 4, 5], np.array([0.98568638, 0.07212553, 0.06881383])),
    79: ([2, 3, 5], np.array([0.63417606, 0.5246093, 0.07370876])),
    80: ([3, 4, 6], np.array([0.66996697, 0.4470458, 0.36837374])),
    81: ([2, 5, 6], np.array([0.6403987, 0.25872259, 0.10879074])),
    82: ([3, 4, 6], np.array([0.97059377, 0.80185875, 0.20964758])),
    83: ([2, 4, 5], np.array([0.67426879, 0.58652335, 0.06964229])),
    84: ([3, 5, 6], np.array([0.41275249, 0.34394568, 0.05800959])),
    85: ([2, 3, 5], np.array([0.84627893, 0.70370688, 0.45180918])),
    86: ([3, 4, 5], np.array([0.96321656, 0.21321243, 0.20753586])),
    87: ([4, 5, 6], np.array([0.74295075, 0.66853995, 0.56719275])),
    88: ([3, 5, 6], np.array([0.8760791, 0.46678926, 0.18722006])),
    89: ([2, 3, 5], np.array([0.67723627, 0.15492652, 0.02428154])),
    90: ([2, 3, 6], np.array([0.34544903, 0.23269052, 0.09441297])),
    91: ([2, 4, 6], np.array([0.81213914, 0.46944902, 0.36396809])),
    92: ([4, 5, 6], np.array([0.81654826, 0.06138655, 0.03278733])),
    93: ([2, 3, 4], np.array([0.5562459, 0.08924231, 0.04953602])),
    94: ([2, 5, 6], np.array([0.935277, 0.8576009, 0.65039414])),
    95: ([2, 3, 6], np.array([0.28376186, 0.21814151, 0.06929687])),
    96: ([3, 4, 5], np.array([0.55837265, 0.18508284, 0.10349277])),
    97: ([2, 3, 4], np.array([0.43990863, 0.38193666, 0.38119709])),
    98: ([3, 5, 6], np.array([0.59582113, 0.56275964, 0.48060071])),
    99: ([2, 4, 6], np.array([0.67692235, 0.60091396, 0.10518461])),
    100: ([2, 3, 4], np.array([0.5581758, 0.0906917, 0.05456114])),
    101: ([2, 4, 6], np.array([0.98603251, 0.42668028, 0.06374854])),
    102: ([2, 5, 6], np.array([0.81291559, 0.57806389, 0.08398249])),
    103: ([2, 4, 6], np.array([0.74229284, 0.70190148, 0.63568075])),
    104: ([3, 4, 5], np.array([0.6470684, 0.63956385, 0.20652468])),
    105: ([4, 5, 6], np.array([0.80517874, 0.70794593, 0.47264146])),
    106: ([2, 4, 6], np.array([0.7557115, 0.32837889, 0.14945491])),
    107: ([2, 3, 4], np.array([0.76493337, 0.36969734, 0.03682478])),
    108: ([2, 4, 6], np.array([0.84922732, 0.06943161, 0.05556929])),
    109: ([4, 5, 6], np.array([0.7237797, 0.4311242, 0.04336195])),
    110: ([2, 4, 6], np.array([0.69615066, 0.67212766, 0.30426581])),
    111: ([2, 3, 5], np.array([0.5036139, 0.22544196, 0.20261521])),
    112: ([2, 3, 4], np.array([0.65181733, 0.26209308, 0.1319334])),
    113: ([3, 4, 5], np.array([0.92666817, 0.51805361, 0.14598683])),
    114: ([3, 4, 6], np.array([0.47158003, 0.07272669, 0.02524788])),
    115: ([3, 4, 5], np.array([0.76647342, 0.43636838, 0.09054291])),
    116: ([3, 5, 6], np.array([0.72943731, 0.18238347, 0.14812741])),
    117: ([4, 5, 6], np.array([0.60920755, 0.48991884, 0.19798412])),
    118: ([2, 4, 6], np.array([0.62571052, 0.13134759, 0.05888562])),
    119: ([4, 5, 6], np.array([0.72678395, 0.65544518, 0.16758307])),
    120: ([3, 4, 6], np.array([0.68567085, 0.65214429, 0.13009329])),
    121: ([3, 4, 6], np.array([0.94102286, 0.3763907, 0.06391996])),
    122: ([3, 4, 6], np.array([0.84405982, 0.20058271, 0.07756991])),
    123: ([2, 3, 6], np.array([0.93353152, 0.59748944, 0.35284541])),
    124: ([4, 5, 6], np.array([0.61335888, 0.39821512, 0.21141188])),
    125: ([3, 5, 6], np.array([0.4544098, 0.16355136, 0.13217224])),
    126: ([2, 3, 6], np.array([0.87757151, 0.67448489, 0.35511471])),
    127: ([2, 3, 4], np.array([0.77414319, 0.53975739, 0.17858585])),
}


# OVERTONES = {
# 0: ([3, 4, 5, 6, 7], np.array([0.9988225 , 0.75635805, 0.71735146, 0.67229691, 0.53719098])),
# 1: ([2, 4, 5, 6, 9], np.array([0.71286682, 0.58540022, 0.58340341, 0.49390077, 0.24068324])),
# 2: ([2, 3, 5, 6, 9], np.array([0.82252254, 0.74600941, 0.38571488, 0.26712816, 0.02900321])),
# 3: ([3, 4, 5, 6, 8], np.array([0.88097097, 0.8530615 , 0.8379902 , 0.3825824 , 0.30791384])),
# 4: ([3, 4, 5, 7, 8], np.array([0.9162284 , 0.48858071, 0.46935549, 0.43575551, 0.16662058])),
# 5: ([3, 4, 6, 7, 8], np.array([0.72536647, 0.58033773, 0.16390617, 0.04217063, 0.04007141])),
# 6: ([3, 4, 5, 7, 8], np.array([0.8776628 , 0.30485642, 0.26566153, 0.24928454, 0.16772504])),
# 7: ([2, 3, 6, 7, 9], np.array([0.8934998 , 0.74362099, 0.24910444, 0.13436623, 0.01062664])),
# 8: ([2, 5, 6, 7, 9], np.array([0.96518214, 0.58615911, 0.57868364, 0.08455658, 0.07947959])),
# 9: ([3, 4, 5, 7, 8], np.array([0.70646089, 0.59995652, 0.21445069, 0.19923653, 0.14769565])),
# 10: ([3, 5, 7, 8, 9], np.array([0.95528686, 0.83399663, 0.56405545, 0.51720095, 0.38331732])),
# 11: ([2, 4, 6, 7, 8], np.array([0.48098504, 0.3905987 , 0.38566617, 0.24515973, 0.1817787 ])),
# 12: ([3, 4, 5, 7, 8], np.array([0.99906408, 0.9073596 , 0.65557435, 0.54649163, 0.04381705])),
# 13: ([2, 4, 7, 8, 9], np.array([0.76917609, 0.69028221, 0.67414169, 0.61112733, 0.53804388])),
# 14: ([3, 4, 6, 7, 9], np.array([0.83598371, 0.59706831, 0.47364296, 0.44061721, 0.30929599])),
# 15: ([2, 3, 4, 5, 6], np.array([0.88437268, 0.7034198 , 0.5917996 , 0.37731204, 0.24031939])),
# 16: ([2, 3, 4, 5, 7], np.array([0.98947992, 0.95241275, 0.46837851, 0.29370254, 0.22970823])),
# 17: ([3, 4, 5, 6, 9], np.array([0.97569868, 0.96129481, 0.93027778, 0.54798387, 0.42307222])),
# 18: ([4, 5, 6, 8, 9], np.array([0.44441583, 0.29138838, 0.26344347, 0.25691877, 0.13701839])),
# 19: ([2, 3, 5, 7, 8], np.array([0.52539173, 0.36152544, 0.30516057, 0.16392915, 0.12413527])),
# 20: ([2, 4, 5, 6, 9], np.array([0.93276749, 0.73757226, 0.61455884, 0.4703222 , 0.09991918])),
# 21: ([2, 4, 5, 8, 9], np.array([0.6569177 , 0.5082943 , 0.31569435, 0.16603091, 0.06994039])),
# 22: ([2, 3, 4, 5, 9], np.array([0.8233649 , 0.79901244, 0.56369847, 0.33506833, 0.21719684])),
# 23: ([2, 3, 5, 8, 9], np.array([0.52891712, 0.44433244, 0.38086559, 0.33088399, 0.22443885])),
# 24: ([4, 5, 6, 7, 8], np.array([0.49841825, 0.40551638, 0.23598785, 0.18992575, 0.15009096])),
# 25: ([2, 3, 4, 6, 7], np.array([0.90693851, 0.84436887, 0.49687134, 0.3901606 , 0.07703592])),
# 26: ([3, 4, 5, 6, 7], np.array([0.59153959, 0.5781782 , 0.44706457, 0.1708079 , 0.147079  ])),
# 27: ([3, 5, 7, 8, 9], np.array([0.78967813, 0.56364493, 0.54335617, 0.50613216, 0.04769554])),
# 28: ([2, 3, 4, 5, 6], np.array([0.78277096, 0.62294664, 0.52525227, 0.37265565, 0.2443791 ])),
# 29: ([3, 4, 6, 8, 9], np.array([0.8256041 , 0.57652978, 0.34178336, 0.33776494, 0.1015175 ])),
# 30: ([2, 5, 6, 7, 9], np.array([0.86789472, 0.85783631, 0.57011493, 0.53919679, 0.38570429])),
# 31: ([2, 3, 5, 7, 9], np.array([0.94834603, 0.84551692, 0.79938254, 0.41249523, 0.0342176 ])),
# 32: ([2, 3, 6, 7, 9], np.array([0.8654294 , 0.73042501, 0.46279224, 0.32296306, 0.15666705])),
# 33: ([2, 3, 5, 7, 8], np.array([0.9659027 , 0.92464426, 0.39875098, 0.15689595, 0.08674838])),
# 34: ([2, 3, 6, 8, 9], np.array([0.99010314, 0.85904274, 0.82929373, 0.64837627, 0.16231015])),
# 35: ([2, 3, 4, 6, 7], np.array([0.94753097, 0.50346317, 0.29445549, 0.13541565, 0.07356258])),
# 36: ([3, 5, 6, 8, 9], np.array([0.81111823, 0.77614196, 0.67076237, 0.66191714, 0.34348002])),
# 37: ([2, 6, 7, 8, 9], np.array([0.90291457, 0.71734958, 0.4474729 , 0.24823298, 0.18732664])),
# 38: ([2, 5, 6, 7, 9], np.array([0.79558297, 0.73655419, 0.51152106, 0.35933093, 0.33516836])),
# 39: ([2, 3, 4, 6, 8], np.array([0.58572619, 0.49652528, 0.37011881, 0.16574627, 0.14353484])),
# 40: ([5, 6, 7, 8, 9], np.array([0.7677898 , 0.66020309, 0.41354766, 0.35029731, 0.16488982])),
# 41: ([4, 6, 7, 8, 9], np.array([0.94436646, 0.83328282, 0.42529121, 0.18655182, 0.16747267])),
# 42: ([3, 4, 6, 7, 8], np.array([0.94246166, 0.81844493, 0.53628979, 0.35888172, 0.06852767])),
# 43: ([2, 3, 5, 7, 8], np.array([0.67464341, 0.66101166, 0.58243932, 0.45511813, 0.06174512])),
# 44: ([2, 3, 4, 5, 6], np.array([0.76841019, 0.65698221, 0.41569267, 0.26470403, 0.00254715])),
# 45: ([2, 5, 6, 8, 9], np.array([0.83229906, 0.75978273, 0.4160644 , 0.07690486, 0.01485571])),
# 46: ([2, 3, 4, 5, 6], np.array([0.81957948, 0.7593818 , 0.42768245, 0.39112814, 0.33101081])),
# 47: ([2, 5, 6, 8, 9], np.array([0.98067786, 0.8926789 , 0.69297921, 0.66646662, 0.26204858])),
# 48: ([2, 4, 6, 8, 9], np.array([0.8801299 , 0.44145142, 0.08463001, 0.04150815, 0.01143287])),
# 49: ([3, 4, 6, 7, 9], np.array([9.85367769e-01, 7.24568796e-01, 6.82338755e-01, 5.60494694e-01, 2.28542294e-04])),
# 50: ([2, 3, 4, 6, 8], np.array([0.83004684, 0.43450523, 0.35661957, 0.31318129, 0.01247182])),
# 51: ([3, 4, 5, 7, 8], np.array([0.73434539, 0.54444808, 0.49001922, 0.27598052, 0.11636825])),
# 52: ([2, 4, 6, 7, 8], np.array([0.91740254, 0.8711881 , 0.82569894, 0.75443903, 0.49569388])),
# 53: ([2, 4, 5, 7, 8], np.array([0.95337282, 0.78887079, 0.54222466, 0.23594493, 0.14703368])),
# 54: ([4, 5, 6, 7, 8], np.array([0.33160526, 0.29628006, 0.2727447 , 0.1934616 , 0.15346125])),
# 55: ([2, 3, 4, 5, 6], np.array([0.98768006, 0.76973568, 0.59297767, 0.27259352, 0.06768353])),
# 56: ([2, 5, 6, 7, 9], np.array([0.40386256, 0.23016529, 0.17856119, 0.0820965 , 0.03478254])),
# 57: ([5, 6, 7, 8, 9], np.array([0.7981112 , 0.53667863, 0.24166267, 0.19902655, 0.14364553])),
# 58: ([2, 3, 6, 8, 9], np.array([0.84072866, 0.77647279, 0.7551837 , 0.67641471, 0.15218512])),
# 59: ([3, 4, 6, 7, 8], np.array([0.75660068, 0.63390869, 0.55787854, 0.33019963, 0.30533939])),
# 60: ([2, 3, 5, 8, 9], np.array([0.92847708, 0.63415063, 0.52482391, 0.46287829, 0.0107331 ])),
# 61: ([2, 3, 5, 6, 7], np.array([0.95625087, 0.91138569, 0.18125187, 0.08668818, 0.00712487])),
# 62: ([2, 4, 5, 6, 8], np.array([0.91021472, 0.51943947, 0.2302685 , 0.22237842, 0.18793702])),
# 63: ([2, 3, 4, 6, 7], np.array([0.99340503, 0.74400418, 0.72721546, 0.72680186, 0.53327053])),
# 64: ([4, 5, 6, 7, 9], np.array([0.72160116, 0.57807896, 0.43315551, 0.18197893, 0.06750179])),
# 65: ([2, 5, 7, 8, 9], np.array([0.79327866, 0.64667449, 0.37774587, 0.3509799 , 0.182348  ])),
# 66: ([3, 5, 6, 7, 8], np.array([0.91997079, 0.7238136 , 0.3827395 , 0.02806501, 0.00783945])),
# 67: ([2, 3, 5, 8, 9], np.array([0.91563385, 0.54678885, 0.52750362, 0.33877272, 0.1199536 ])),
# 68: ([2, 3, 4, 5, 7], np.array([0.96016175, 0.62837078, 0.48609328, 0.37359688, 0.25993966])),
# 69: ([3, 6, 7, 8, 9], np.array([0.99635261, 0.92645841, 0.63393449, 0.40647122, 0.30446327])),
# 70: ([2, 3, 4, 7, 9], np.array([0.87720973, 0.60734774, 0.32319237, 0.29975513, 0.25720222])),
# 71: ([3, 5, 6, 8, 9], np.array([0.49241594, 0.47670076, 0.31071765, 0.20681898, 0.15845903])),
# 72: ([2, 4, 6, 8, 9], np.array([0.95631339, 0.786748  , 0.59394362, 0.3251305 , 0.26053879])),
# 73: ([2, 3, 5, 8, 9], np.array([0.93272521, 0.33881858, 0.26962314, 0.26141198, 0.21309319])),
# 74: ([4, 5, 6, 7, 9], np.array([0.92667544, 0.30645408, 0.10088954, 0.08387282, 0.04860337])),
# 75: ([2, 3, 5, 7, 9], np.array([0.98762442, 0.92433113, 0.77261943, 0.14089615, 0.0841763 ])),
# 76: ([2, 3, 4, 6, 9], np.array([0.79130746, 0.4808795 , 0.29910746, 0.07440602, 0.02267408])),
# 77: ([2, 3, 4, 5, 9], np.array([0.71249497, 0.62573159, 0.53836104, 0.46757915, 0.37527692])),
# 78: ([2, 4, 5, 6, 8], np.array([0.87583757, 0.756837  , 0.60862272, 0.0672499 , 0.04354583])),
# 79: ([4, 5, 6, 7, 8], np.array([0.72571623, 0.67621953, 0.52284302, 0.47039681, 0.14682657])),
# 80: ([4, 5, 6, 8, 9], np.array([0.93441777, 0.76134035, 0.75381551, 0.25788748, 0.23469616])),
# 81: ([2, 3, 5, 6, 8], np.array([0.71175524, 0.67352365, 0.33919104, 0.1606721 , 0.00732928])),
# 82: ([2, 3, 7, 8, 9], np.array([0.81151384, 0.78568546, 0.50368304, 0.27448512, 0.07250199])),
# 83: ([3, 5, 7, 8, 9], np.array([0.87755857, 0.87509418, 0.35725346, 0.27103844, 0.07149341])),
# 84: ([4, 6, 7, 8, 9], np.array([0.97114241, 0.87021848, 0.59881841, 0.27163665, 0.13705172])),
# 85: ([2, 3, 5, 7, 8], np.array([0.99059981, 0.6383284 , 0.61583174, 0.41345497, 0.10168845])),
# 86: ([2, 3, 6, 7, 9], np.array([0.91244757, 0.88520214, 0.61760762, 0.50755701, 0.07826332])),
# 87: ([2, 3, 5, 6, 9], np.array([0.90225332, 0.60988864, 0.38059215, 0.15636669, 0.05795571])),
# 88: ([2, 5, 7, 8, 9], np.array([0.93732137, 0.83522093, 0.61311794, 0.54186031, 0.17909946])),
# 89: ([4, 5, 6, 8, 9], np.array([0.87568489, 0.68478658, 0.58538628, 0.48529047, 0.46755462])),
# 90: ([3, 4, 6, 8, 9], np.array([0.92721403, 0.90306927, 0.5467413 , 0.15979312, 0.01545425])),
# 91: ([4, 5, 6, 8, 9], np.array([0.8793442 , 0.6200017 , 0.32445318, 0.2620042 , 0.249513  ])),
# 92: ([2, 4, 5, 6, 8], np.array([0.73036776, 0.66224186, 0.59236516, 0.53034293, 0.44985581])),
# 93: ([2, 3, 7, 8, 9], np.array([0.93742074, 0.82100869, 0.31600983, 0.11794851, 0.07302239])),
# 94: ([5, 6, 7, 8, 9], np.array([0.93241176, 0.62483175, 0.38235014, 0.07409163, 0.03709725])),
# 95: ([2, 5, 6, 8, 9], np.array([0.97526675, 0.95181663, 0.83854006, 0.82923832, 0.32439033])),
# 96: ([2, 3, 5, 7, 8], np.array([0.82233857, 0.67701979, 0.50130094, 0.32101274, 0.09657641])),
# 97: ([2, 4, 5, 6, 7], np.array([0.90917909, 0.7408135 , 0.63484151, 0.30766431, 0.24309234])),
# 98: ([3, 5, 7, 8, 9], np.array([0.88245709, 0.79872817, 0.56162035, 0.47331895, 0.10016166])),
# 99: ([2, 3, 5, 6, 7], np.array([0.99802991, 0.85414865, 0.84326223, 0.51518968, 0.36732736])),
# 100: ([3, 4, 5, 6, 8], np.array([0.76351633, 0.31906417, 0.1242052 , 0.09509108, 0.08508076])),
# 101: ([5, 6, 7, 8, 9], np.array([0.74304343, 0.61930077, 0.45545011, 0.28492524, 0.09940438])),
# 102: ([4, 5, 7, 8, 9], np.array([0.96516855, 0.57893071, 0.38869652, 0.34467906, 0.03367131])),
# 103: ([2, 3, 5, 7, 9], np.array([0.83699042, 0.75423696, 0.58070512, 0.45310998, 0.3961646 ])),
# 104: ([4, 5, 6, 7, 8], np.array([0.88381488, 0.85281614, 0.73571742, 0.14813664, 0.05808287])),
# 105: ([2, 3, 4, 5, 6], np.array([0.54639176, 0.40480834, 0.29524149, 0.23561655, 0.14524882])),
# 106: ([3, 4, 6, 7, 8], np.array([0.98321001, 0.50023397, 0.49982486, 0.23266039, 0.05213168])),
# 107: ([3, 4, 5, 7, 8], np.array([0.97487627, 0.63904551, 0.57870767, 0.29396594, 0.08871121])),
# 108: ([2, 3, 4, 5, 8], np.array([0.9947858 , 0.49746357, 0.38968633, 0.37321027, 0.1056264 ])),
# 109: ([3, 4, 5, 6, 8], np.array([0.84744794, 0.5715397 , 0.28339918, 0.11031789, 0.0628728 ])),
# 110: ([2, 4, 5, 6, 7], np.array([0.80151357, 0.7248113 , 0.51109649, 0.41230284, 0.32072238])),
# 111: ([3, 4, 5, 6, 9], np.array([0.98712311, 0.42001599, 0.38197256, 0.12958581, 0.12732684])),
# 112: ([2, 5, 6, 8, 9], np.array([0.96606639, 0.80230998, 0.67089654, 0.40140795, 0.22195468])),
# 113: ([2, 4, 5, 8, 9], np.array([0.84173736, 0.3560192 , 0.24119335, 0.10532208, 0.02250545])),
# 114: ([5, 6, 7, 8, 9], np.array([0.98475848, 0.63123092, 0.51020609, 0.06633853, 0.0042955 ])),
# 115: ([2, 3, 4, 8, 9], np.array([0.88218738, 0.68185754, 0.6609161 , 0.23637783, 0.0597526 ])),
# 116: ([2, 3, 5, 8, 9], np.array([0.92182512, 0.70685246, 0.6094279 , 0.29974093, 0.28812471])),
# 117: ([2, 3, 4, 5, 9], np.array([0.41050035, 0.39595735, 0.39227146, 0.11175635, 0.02009435])),
# 118: ([2, 5, 6, 8, 9], np.array([0.69890472, 0.56315245, 0.54725174, 0.14263982, 0.08375886])),
# 119: ([3, 6, 7, 8, 9], np.array([0.84400567, 0.79795212, 0.56596349, 0.06378745, 0.05854682])),
# 120: ([2, 5, 6, 7, 8], np.array([0.90618372, 0.54489313, 0.41598824, 0.38630647, 0.37494688])),
# 121: ([2, 5, 6, 7, 8], np.array([0.48342567, 0.3248571 , 0.22300546, 0.18525548, 0.03804161])),
# 122: ([4, 6, 7, 8, 9], np.array([0.54436261, 0.50926434, 0.29628139, 0.19148006, 0.1902753 ])),
# 123: ([2, 4, 7, 8, 9], np.array([0.74087782, 0.40886929, 0.22653127, 0.1268814 , 0.0735844 ])),
# 124: ([2, 3, 4, 8, 9], np.array([0.99955536, 0.73384007, 0.61138493, 0.29733387, 0.15067294])),
# 125: ([2, 3, 5, 6, 7], np.array([0.90713024, 0.81968599, 0.73828465, 0.6890157 , 0.29406535])),
# 126: ([4, 5, 6, 7, 9], np.array([0.99436547, 0.83948604, 0.82576203, 0.21402503, 0.10808613])),
# 127: ([2, 4, 7, 8, 9], np.array([0.76113219, 0.75682709, 0.40017985, 0.24486619, 0.03358239])),
# }


def instrument_overtone_intensities_fast(program: int):
    mult_to_sum = {2: 12,
                   3: 19,
                   4: 24,
                   5: 28,
                   6: 31,
                   7: 34,
                   8: 36,
                   9: 38,
                   10: 40
                   }
    harmonics, intensities = OVERTONES[program]
    harmonics = [mult_to_sum[h] for h in harmonics]
    return harmonics, intensities


def create_piano_roll(
        note_data,
        ticks_per_beat,
        chroma=False,
        target_ticks_per_beat=4,
        instrument_overtones=False,
        separate_drums=False
):
    if len(note_data) == 0:
        return None
    num_notes = 12 if chroma else 128
    duration_ticks = note_data[-1]["time"] + note_data[-1]["duration"]
    piano_roll = np.zeros((3, num_notes, duration_ticks))

    for note in note_data:
        # fixed duration for drum tracks since we only need the onsets
        drum_track = note["channel"] == 9
        duration = 1 if drum_track else note["duration"]

        start = note["time"]
        end = min(start + duration, duration_ticks)
        if end - start <= 0:
            continue

        pitch_class = note["note"] % 12 if chroma else note["note"]

        velocity = note["velocity"]
        piano_roll_channel = 2 if drum_track and separate_drums else 0
        piano_roll[piano_roll_channel, pitch_class, start:end] = velocity
        if not instrument_overtones:
            piano_roll[1, pitch_class, start:end] = velocity

        if drum_track and not separate_drums:
            piano_roll[0, pitch_class, start:end] = velocity

        # Add overtones
        if instrument_overtones and not drum_track:
            program = note["program"]
            harmonics, intensities = instrument_overtone_intensities(program)
            pitch = midi_to_hz(note["note"])
            max_intensity = intensities[0]
            for harmonic, intensity in zip(harmonics, intensities):
                overtone_pitch = pitch * harmonic
                overtone_midi = hz_to_midi(overtone_pitch)
                overtone_pitch_class = overtone_midi % 12 if chroma else overtone_midi
                overtone_pitch_class = int(np.round(overtone_pitch_class))
                if overtone_pitch_class <= 127:
                    decay = np.linspace(1.0, 0.0, end - start) * intensity / max_intensity
                    piano_roll[1, overtone_pitch_class, start:end] = velocity * intensity * decay

    # Calculate the exact target length to maintain alignment with tick markers
    resample_ratio = target_ticks_per_beat / ticks_per_beat
    target_length = int(duration_ticks * resample_ratio)

    try:
        piano_roll_tensor = torch.tensor(piano_roll)
        # Use adaptive max pooling to preserve maximum velocities while achieving exact target length
        piano_roll = F.adaptive_max_pool1d(
            piano_roll_tensor,
            output_size=target_length
        ).numpy()
    except Exception as e:
        print(e)
        print(piano_roll.shape)
        return None

    return piano_roll


def random_take(one_in_n: int) -> bool:
    return (torch.randint(0, one_in_n, ()) < 1).bool().item()  # type: ignore


def get_piano_roll_cache_path(file_id: str, piano_roll_dir: Optional[Path],
                              target_ticks_per_beat: int) -> Optional[Path]:
    """Get the cache path for a piano roll file."""
    if not piano_roll_dir:
        return None

    # Create a unique filename based on file_id and piano roll parameters
    cache_filename = (
        f"{file_id}_tpb{target_ticks_per_beat}.pt"
    )
    return piano_roll_dir / cache_filename


def get_sslm_cache_path(file_id: str, sslm_dir: Optional[Path], target_ticks_per_beat: int) -> Optional[Path]:
    """Get the cache path for an SSLM file."""
    if not sslm_dir:
        return None

    cache_filename = f"{file_id}_sslm_tp{target_ticks_per_beat}.pt"
    return sslm_dir / cache_filename


# def create_lakh_dataset(
#         lakh_midi_dir: Union[Path, str],
#         data_dir: Union[Path, str],
#         files_dict: Dict[str, List[str]],
#         markers_qn_path: Optional[Union[Path, str]] = None,
#         annotation_dir: Optional[Union[Path, str]] = None,
#         target_ticks_per_beat: int = 4,
#         instrument_overtones: bool = True,
#         separate_drums: bool = True,
#         compute_sslm_near: bool = False,
#         compute_sslm_far: bool = False,
#         measures_qn_path: Optional[Union[Path, str]] = None,
# ):
#     """
#     Loads MIDI files from the Lakh MIDI dataset, processes them into piano rolls,
#     and saves them in a structured directory format for training, validation, and testing.
#
#     Supports two data formats:
#
#     OLD FORMAT (markers_qn_path provided):
#         - Boundaries from markers_qn.json
#         - Optional labels from annotation_dir
#
#     NEW FORMAT (annotation_dir without markers_qn_path):
#         - Boundaries AND labels from {FILE_ID}_labels_coarse_qn.json
#
#     Args:
#         lakh_midi_dir: Directory containing MIDI files
#         data_dir: Output directory for processed data
#         files_dict: Dict mapping split names to file IDs
#         markers_qn_path: Path to markers_qn.json (old format, optional)
#         annotation_dir: Directory with per-file annotation JSONs (new format)
#         measures_qn_path: Path to measures_qn.json (optional, for beat/downbeat)
#     """
#     if isinstance(data_dir, str):
#         data_dir = Path(data_dir)
#     if isinstance(lakh_midi_dir, str):
#         lakh_midi_dir = Path(lakh_midi_dir)
#     if annotation_dir and isinstance(annotation_dir, str):
#         annotation_dir = Path(annotation_dir)
#
#     # Determine format: old (with tubb/non_tubb) or new (simple split)
#     use_old_format = markers_qn_path is not None
#
#     if not markers_qn_path and not annotation_dir:
#         raise ValueError("Must provide either markers_qn_path (old format) or annotation_dir (new format)")
#
#     if not data_dir.exists():
#         data_dir.mkdir(parents=True)
#
#     for split in files_dict.keys():
#         (data_dir / split).mkdir(exist_ok=True)
#         (data_dir / split / Path("metadata")).mkdir(exist_ok=True)
#         (data_dir / split / Path("piano_rolls")).mkdir(exist_ok=True)
#         if compute_sslm_near or compute_sslm_far:
#             (data_dir / split / Path("sslms")).mkdir(exist_ok=True)
#
#     # Load measure data if available
#     measure_qns_all = None
#     if measures_qn_path:
#         with open(measures_qn_path, "r") as f:
#             measure_qns_all = json.load(f)
#     for key in files_dict:
#         print(f"Processing files: {key}")
#         for test_example in tqdm(files_dict[key], desc="Loading examples"):
#             metadata_path = data_dir / Path(key) / Path("metadata") / Path(f"{test_example}.pt")
#             piano_roll_path = get_piano_roll_cache_path(test_example, data_dir / Path(key) / Path("piano_rolls"),
#                                                         target_ticks_per_beat)
#             if metadata_path.exists() and piano_roll_path and piano_roll_path.exists():
#                 # Skip already processed files
#                 continue
#
#             midi_path = lakh_midi_dir / Path(f"{test_example[0]}") / Path(test_example + ".mid")
#             if not midi_path.exists():
#                 print(f"Missing MIDI file: {midi_path}")
#                 continue
#
#             # MIDI
#             try:
#                 midi = mido.MidiFile(midi_path, clip=True)
#                 ticks_per_beat = midi.ticks_per_beat
#             except Exception as e:
#                 print(f"Error loading MIDI file: {midi_path}")
#                 print(e)
#                 continue
#
#             # Load boundaries and labels based on format
#             if use_old_format:
#                 # Old format: boundaries from markers_qn.json
#                 try:
#                     markers_ticks = parse_markers(
#                         markers_qn_path=markers_qn_path,
#                         file_id=test_example,
#                         ticks_per_beat=ticks_per_beat
#                     )
#                 except Exception as e:
#                     print(f"Error parsing markers for {test_example}: {e}")
#                     continue
#
#                 # Convert to target resolution
#                 markers_ticks = [int(round(marker * target_ticks_per_beat / ticks_per_beat)) for marker in
#                                  markers_ticks]
#                 segment_labels = None
#             else:
#                 # New format: boundaries and labels from annotation file
#                 assert annotation_dir is not None, "annotation_dir required for new format"
#                 annotation_path = annotation_dir / Path(f"{test_example}_labels_coarse_qn.json")
#                 if not annotation_path.exists():
#                     print(f"Missing annotation file: {annotation_path}")
#                     continue
#
#                 try:
#                     with open(annotation_path, "r") as f:
#                         annotations = json.load(f)
#                     processed_annotations = preprocess_labels(annotations)
#
#                     # Extract boundaries (in quarter notes) and labels together
#                     segment_qns = [ann[0] for ann in processed_annotations]
#                     segment_labels = [ann[1] for ann in processed_annotations]
#
#                     # Convert quarter notes to ticks
#                     markers_ticks = [int(round(qn * target_ticks_per_beat)) for qn in segment_qns]
#                 except Exception as e:
#                     print(f"Error processing annotations for {test_example}: {e}")
#                     continue
#
#             # Load measure boundaries if available
#             measure_ticks = None
#             if measure_qns_all and test_example in measure_qns_all:
#                 measure_qns = measure_qns_all[test_example]
#                 measure_ticks = [int(round(qn * target_ticks_per_beat)) for qn in measure_qns]
#             else:
#                 print(f"No measure data for {test_example}")
#
#             piano_roll = create_piano_roll_fast(
#                 path_to_midi_file=midi_path,
#                 chroma=False,
#                 target_ticks_per_beat=target_ticks_per_beat,
#             )
#             piano_roll = torch.tensor(piano_roll, dtype=torch.float32)
#             torch.save(piano_roll, str(piano_roll_path))
#
#             data = {
#                 "segment_boundaries": markers_ticks,
#             }
#
#             # Add measure ticks if available
#             if measure_ticks is not None:
#                 data["measure_ticks"] = measure_ticks
#
#             # Add segment labels if available
#             if segment_labels is not None:
#                 data["segment_labels"] = segment_labels
#             elif use_old_format and annotation_dir:
#                 # Old format with annotation_dir: optionally add labels from annotations
#                 # WARNING: boundaries from markers_qn.json may not match annotation boundaries!
#                 annotation_path = annotation_dir / f"{test_example}_labels_coarse_qn.json"
#                 if annotation_path.exists():
#                     with open(annotation_path, "r") as f:
#                         annotations = json.load(f)
#                     processed_annotations = preprocess_labels(annotations)
#                     data["segment_labels"] = [label for _, label in processed_annotations]
#
#             sslm_data = dict()
#             if compute_sslm_near:
#                 sslm_piano_roll = piano_roll.sum(dim=0)
#                 sslm_near, _ = compute_sslms(sslm_piano_roll, L=int((90 / 0.5) * target_ticks_per_beat))
#                 sslm_data["sslm_near"] = sslm_near
#
#             if compute_sslm_far:
#                 sslm_piano_roll = piano_roll.sum(dim=0)
#                 _, sslm_far = compute_sslms(sslm_piano_roll, L=int((90 / 0.5) * target_ticks_per_beat))
#                 sslm_data["sslm_far"] = sslm_far
#
#             if sslm_data:
#                 sslm_path = get_sslm_cache_path(test_example, data_dir / Path(key) / Path("sslms"),
#                                                 target_ticks_per_beat)
#                 torch.save(sslm_data, str(sslm_path))
#
#             torch.save(data, str(metadata_path))
#
#
# def create_piano_roll_patch_data(
#         midi_dir: Union[Path, str],
#         files_dict: Dict[str, List[str]],
#         markers_qn_path: Optional[Union[Path, str]] = None,
#         measures_qn_path: Optional[Union[Path, str]] = None,
#         annotation_dir: Optional[Union[Path, str]] = None,
#         piano_roll_dir: Optional[Union[Path, str]] = None,
#         sslm_dir: Optional[Union[Path, str]] = None,
#         window_half_ticks: int = 256,
#         target_ticks_per_beat: int = 4,
#         instrument_overtones: bool = True,
#         separate_drums: bool = True,
#         pad_boundary_patches: bool = True,
#         positive_oversampling_factor: int = 2,
#         negative_undersampling_factor: int = 1,
#         return_sslm_near: bool = False,
#         return_sslm_far: bool = False,
# ) -> PatchData:
#     if isinstance(midi_dir, str):
#         midi_dir = Path(midi_dir)
#     if markers_qn_path and isinstance(markers_qn_path, str):
#         markers_qn_path = Path(markers_qn_path)
#     if annotation_dir and isinstance(annotation_dir, str):
#         annotation_dir = Path(annotation_dir)
#     if piano_roll_dir and isinstance(piano_roll_dir, str):
#         piano_roll_dir = Path(piano_roll_dir)
#     if sslm_dir and isinstance(sslm_dir, str):
#         sslm_dir = Path(sslm_dir)
#
#     # Create cache paths
#     if piano_roll_dir and not piano_roll_dir.exists():
#         piano_roll_dir.mkdir(parents=True)
#     if sslm_dir and not sslm_dir.exists():
#         sslm_dir.mkdir(parents=True)
#
#     use_old_format = annotation_dir is None
#
#     measure_qns_all = None
#     if measures_qn_path:
#         with open(measures_qn_path, "r") as f:
#             measure_qns_all = json.load(f)
#
#     # Full piano rolls and sslms will be cached in memory
#     # Otherwise we would need to load them here and again in __getitem__ for every patch
#     piano_roll_idx = 0
#     piano_rolls = []
#     sslm_nears = []
#     sslm_fars = []
#
#     sample_idx = 0
#     patch_data = {}
#
#     for split in files_dict.keys():
#         for test_example in tqdm(files_dict[split], desc=f"Loading examples: {split}"):
#             midi_path = midi_dir / Path(f"{test_example[0]}") / Path(test_example + ".mid")
#             if not midi_path.exists():
#                 print(f"Missing MIDI file: {midi_path}")
#                 continue
#
#             # MIDI
#             try:
#                 midi = mido.MidiFile(midi_path, clip=True)
#                 ticks_per_beat = midi.ticks_per_beat
#             except Exception as e:
#                 print(f"Error loading MIDI file: {midi_path}")
#                 print(e)
#                 continue
#
#             if measure_qns_all and test_example in measure_qns_all:
#                 measure_qns = measure_qns_all[test_example]
#                 measure_ticks = [int(round(qn * target_ticks_per_beat)) for qn in measure_qns]
#             else:
#                 print(f"No measure data for {test_example}")
#                 continue
#
#             # Old format: boundaries from markers_qn.json
#             if use_old_format:
#                 try:
#                     markers_ticks = parse_markers(
#                         markers_qn_path=markers_qn_path,
#                         file_id=test_example,
#                         ticks_per_beat=ticks_per_beat
#                     )
#                 except Exception as e:
#                     print(f"Error parsing markers for {test_example}: {e}")
#                     continue
#
#                 # Convert to target resolution
#                 markers_ticks = [int(round(marker * target_ticks_per_beat / ticks_per_beat)) for marker in
#                                  markers_ticks]
#                 segment_labels = None
#             else:
#                 # New format: boundaries and labels from annotation file
#                 assert annotation_dir is not None, "annotation_dir required for new format"
#                 annotation_path = annotation_dir / Path(f"{test_example}_labels_coarse_qn.json")
#                 if not annotation_path.exists():
#                     print(f"Missing annotation file: {annotation_path}")
#                     continue
#
#                 try:
#                     with open(annotation_path, "r") as f:
#                         annotations = json.load(f)
#                     processed_annotations = preprocess_labels(annotations)
#
#                     # Extract boundaries (in quarter notes) and labels together
#                     segment_qns = [ann[0] for ann in processed_annotations]
#                     segment_labels = [ann[1] for ann in processed_annotations]
#
#                     # Convert quarter notes to ticks
#                     markers_ticks = [int(round(qn * target_ticks_per_beat)) for qn in segment_qns]
#                 except Exception as e:
#                     print(f"Error processing annotations for {test_example}: {e}")
#                     continue
#
#             piano_roll_path = get_piano_roll_cache_path(test_example, piano_roll_dir, target_ticks_per_beat)
#             sslm_path = get_sslm_cache_path(test_example, sslm_dir, target_ticks_per_beat)
#
#             if piano_roll_path and piano_roll_path.exists():
#                 piano_roll = torch.load(piano_roll_path)
#             else:
#                 try:
#                     piano_roll = create_piano_roll_fast(
#                         path_to_midi_file=midi_path,
#                         chroma=False,
#                         target_ticks_per_beat=target_ticks_per_beat,
#                     )
#                 except Exception as e:
#                     print(f"Error creating piano roll for {test_example}: {e}")
#                     continue
#             piano_roll = torch.tensor(piano_roll, dtype=torch.float32)
#             if piano_roll_path and not piano_roll_path.exists():
#                 torch.save(piano_roll, str(piano_roll_path))
#
#             # Compute first and last nonzero columns of the first channel (first and last onset, respectively)
#             if piano_roll.dim() == 4:
#                 batch_mask = piano_roll[0]  # Select the first batch
#             else:
#                 batch_mask = piano_roll
#             channel_mask = batch_mask[0]  # Select the first channel
#
#             # Find nonzero column indices
#             nonzero_indices = channel_mask.nonzero(as_tuple=True)
#             if nonzero_indices[1].numel() > 0:
#                 first_nonzero_column = nonzero_indices[1].min().item()
#                 last_nonzero_column = nonzero_indices[1].max().item()
#             else:
#                 continue
#
#             # Throw out markers before first onset or after last onset
#             markers_ticks = torch.tensor(markers_ticks, dtype=torch.float32, device=piano_roll.device)
#             measure_ticks = torch.tensor(measure_ticks, dtype=torch.float32, device=piano_roll.device)
#             markers_ticks = markers_ticks[markers_ticks > first_nonzero_column]
#             markers_ticks = markers_ticks[markers_ticks < last_nonzero_column]
#             measure_ticks = measure_ticks[measure_ticks > first_nonzero_column]
#             measure_ticks = measure_ticks[measure_ticks < last_nonzero_column]
#
#             # Add first and last nonzero column to the segment boundaries
#             markers_ticks = torch.cat([
#                 torch.tensor([first_nonzero_column], dtype=torch.float32, device=piano_roll.device),
#                 markers_ticks,
#                 torch.tensor([last_nonzero_column], dtype=torch.float32, device=piano_roll.device)
#             ])
#             measure_ticks = torch.cat([
#                 torch.tensor([first_nonzero_column], dtype=torch.float32, device=piano_roll.device),
#                 measure_ticks,
#                 torch.tensor([last_nonzero_column], dtype=torch.float32, device=piano_roll.device)
#             ])
#
#             # Crop piano roll to the first and last onset
#             piano_roll = piano_roll[..., first_nonzero_column:last_nonzero_column + 1]
#             # Adjust segment boundaries to the cropped piano roll
#             markers_ticks -= first_nonzero_column
#             measure_ticks -= first_nonzero_column
#
#             # Pad piano roll to the left and right for boundary segment extraction
#             padding = window_half_ticks
#             if pad_boundary_patches:
#                 piano_roll = F.pad(piano_roll, (padding, padding), mode='constant', value=0)
#                 markers_ticks += padding
#                 measure_ticks += padding
#
#             piano_rolls.append(piano_roll)
#
#             if return_sslm_near or return_sslm_far:
#                 sslm_data = dict()
#                 if sslm_path and sslm_path.exists():
#                     sslm_data = torch.load(sslm_path)
#                 else:
#                     sslm_near, sslm_far = compute_sslms_from_midi_path(p=midi_path,
#                                                                        target_ticks_per_beat=target_ticks_per_beat)
#                     sslm_near = sslm_near[..., first_nonzero_column:last_nonzero_column + 1]
#                     sslm_far = sslm_far[..., first_nonzero_column:last_nonzero_column + 1]
#                     if pad_boundary_patches:
#                         sslm_near = F.pad(sslm_near, (padding, padding), mode='constant', value=0)
#                         sslm_far = F.pad(sslm_far, (padding, padding), mode='constant', value=0)
#
#                     if return_sslm_near:
#                         sslm_data["sslm_near"] = sslm_near
#                     if return_sslm_far:
#                         sslm_data["sslm_far"] = sslm_far
#                     if sslm_path and not sslm_path.exists():
#                         torch.save(sslm_data, str(sslm_path))
#
#                 if return_sslm_near:
#                     sslm_nears.append(sslm_data.get("sslm_near", None))
#                 if return_sslm_far:
#                     sslm_fars.append(sslm_data.get("sslm_far", None))
#
#             for i in measure_ticks:
#                 is_segment_boundary = (markers_ticks == i).any().item()
#                 nearest_segment_boundary = markers_ticks[torch.argmin(torch.abs(markers_ticks - i))].item()
#
#                 patch_info = {
#                     "file_id": test_example,
#                     "midi_path": str(midi_path),
#                     "piano_roll_path": str(piano_roll_path) if piano_roll_path else None,
#                     "sslm_path": str(sslm_path) if sslm_path else None,
#
#                     "from": i - window_half_ticks,  # todo: rename to from_tick
#                     "to": i + window_half_ticks,  # todo: rename to to_tick
#                     "is_segment_boundary": is_segment_boundary,
#                     "nearest_segment_boundary": nearest_segment_boundary,
#                     "key": split,  # todo: rename to split
#                     "piano_roll_idx": piano_roll_idx,
#                     "sample_idx": sample_idx,
#                     "patch_idx": i,
#                 }
#                 # Add SSLM patch index if we're returning SSLMs
#                 if return_sslm_near:
#                     patch_info["sslm_near_patch_idx"] = piano_roll_idx
#                 if return_sslm_far:
#                     patch_info["sslm_far_patch_idx"] = piano_roll_idx
#
#                 # Add segment label immediately to the right of patch center
#                 # This is the label of the segment containing position i (or i+epsilon)
#                 if segment_labels is not None:
#                     # Find the largest boundary <= i
#                     boundaries_at_or_before = markers_ticks[markers_ticks <= i]
#                     if len(boundaries_at_or_before) > 0:
#                         # Find index of the most recent boundary
#                         current_boundary = boundaries_at_or_before[-1]
#                         boundary_idx = torch.where(markers_ticks == current_boundary)[0][0].item()
#                         patch_info["segment_label"] = segment_labels[boundary_idx]
#                     else:
#                         # Shouldn't happen if boundaries include first onset
#                         patch_info["segment_label"] = "Start"
#
#                 repetitions = positive_oversampling_factor if is_segment_boundary == 1. else int(
#                     random_take(one_in_n=negative_undersampling_factor))
#                 for _ in range(repetitions):
#                     patch_data[sample_idx] = patch_info
#                     sample_idx += 1
#
#             piano_roll_idx += 1
#
#     return PatchData(
#         piano_rolls=piano_rolls,
#         patch_metadata=patch_data,
#         sslm_near_patches=sslm_nears if return_sslm_near else None,
#         sslm_far_patches=sslm_fars if return_sslm_far else None
#     )
#
#
# _sslms_near = dict()
# _sslms_far = dict()
#
#
# def get_piano_roll_patches(
#         data_dir: Union[Path, str],
#         window_half_ticks: int = 256,
#         positive_oversampling_factor: int = 2,
#         negative_undersampling_factor: int = 1,
#         pad_boundary_patches: bool = True,
#         return_sslm_near: bool = False,
#         return_sslm_far: bool = False
# ) -> PatchData:
#     """
#     Load piano rolls from the specified paths, process them, and return a PatchData object
#     containing piano rolls, patch metadata, and optionally SSLM patches.
#
#     Supports both old format (tubb/non_tubb structure) and new format (simple train/val/test).
#     """
#     if isinstance(data_dir, str):
#         data_dir = Path(data_dir)
#
#     # Detect directory structure
#     piano_roll_paths = []
#     metadata_paths = dict()
#     sslm_paths = dict()
#     for subdir in data_dir.iterdir():
#         if subdir.is_dir():
#             piano_roll_dir = subdir / "piano_rolls"
#             if piano_roll_dir.exists():
#                 piano_roll_paths.extend([
#                     path for path in piano_roll_dir.iterdir()
#                     if path.suffix == ".pt" and not path.name.startswith(".")
#                 ])
#             else:
#                 # Assume simple structure
#                 piano_roll_paths.extend([
#                     path for path in subdir.iterdir()
#                     if path.suffix == ".pt" and not path.name.startswith(".")
#                 ])
#
#             metadata_dir = subdir / "metadata"
#             if metadata_dir.exists():
#                 # add metadata_path.stem: metadata_path
#                 for path in metadata_dir.iterdir():
#                     if path.suffix == ".pt" and not path.name.startswith("."):
#                         metadata_paths[path.stem] = path
#
#             sslm_dir = subdir / "sslms"
#             if sslm_dir.exists():
#                 for path in sslm_dir.iterdir():
#                     if path.suffix == ".pt" and not path.name.startswith("."):
#                         file_id = path.stem.split("_")[0]
#                         sslm_paths[file_id] = path
#
#     padding = window_half_ticks
#
#     piano_rolls = []
#     sslms_near = []
#     sslms_far = []
#     patch_data = {}
#     sample_idx = 0
#     piano_roll_idx = 0
#
#     for piano_roll_path in tqdm(piano_roll_paths, desc="Loading inputs and labels"):
#         file_id = str(piano_roll_path.stem).split("_")[0]
#         metadata_path = metadata_paths.get(file_id, None)
#         if metadata_path is None:
#             print(f"Missing metadata for {piano_roll_path.stem}, skipping.")
#             continue
#         metadata = torch.load(metadata_path)
#
#         try:
#             piano_roll = torch.load(piano_roll_path)
#         except RuntimeError:
#             print(f"Error loading {piano_roll_path}")
#             continue
#
#         sslm_path = sslm_paths.get(file_id, None)
#         if sslm_path:
#             sslm_data = torch.load(sslm_path)
#             if return_sslm_near and "sslm_near" in sslm_data:
#                 _sslms_near[piano_roll_idx] = sslm_data["sslm_near"]
#             if return_sslm_far and "sslm_far" in sslm_data:
#                 _sslms_far[piano_roll_idx] = sslm_data["sslm_far"]
#
#         # Don't oversample/undersample in validation/test sets
#         positive_oversampling_factor = positive_oversampling_factor if 'train' in str(piano_roll_path) else 1
#         negative_undersampling_factor = negative_undersampling_factor if 'train' in str(piano_roll_path) else 1
#
#         segment_boundaries = torch.tensor(metadata["segment_boundaries"])
#         measure_boundaries = torch.tensor(metadata["measure_ticks"])
#         segment_labels = metadata.get("segment_labels", None)
#
#         # Compute first and last nonzero columns of the first channel (first and last onset, respectively)
#         if piano_roll.dim() == 4:
#             batch_mask = piano_roll[0]  # Select the first batch
#         else:
#             batch_mask = piano_roll
#         channel_mask = batch_mask[0]  # Select the first channel
#
#         # Find nonzero column indices
#         nonzero_indices = channel_mask.nonzero(as_tuple=True)
#         if nonzero_indices[1].numel() > 0:
#             first_nonzero_column = nonzero_indices[1].min().item()
#             last_nonzero_column = nonzero_indices[1].max().item()
#         else:
#             continue
#
#         # Throw out markers before first onset or after last onset
#         segment_boundaries = segment_boundaries[segment_boundaries > first_nonzero_column]
#         segment_boundaries = segment_boundaries[segment_boundaries < last_nonzero_column]
#         measure_boundaries = measure_boundaries[measure_boundaries > first_nonzero_column]
#         measure_boundaries = measure_boundaries[measure_boundaries < last_nonzero_column]
#
#         # Add first and last nonzero column to the segment boundaries
#         segment_boundaries = torch.cat([
#             torch.tensor([first_nonzero_column], dtype=torch.float32, device=piano_roll.device),
#             segment_boundaries,
#             torch.tensor([last_nonzero_column], dtype=torch.float32, device=piano_roll.device)
#         ])
#         measure_boundaries = torch.cat([
#             torch.tensor([first_nonzero_column], dtype=torch.float32, device=piano_roll.device),
#             measure_boundaries,
#             torch.tensor([last_nonzero_column], dtype=torch.float32, device=piano_roll.device)
#         ])
#
#         # Crop piano roll to the first and last onset
#         piano_roll = piano_roll[..., first_nonzero_column:last_nonzero_column + 1]
#         # Adjust segment boundaries to the cropped piano roll
#         segment_boundaries -= first_nonzero_column
#         measure_boundaries -= first_nonzero_column
#
#         # Pad piano roll to the left and right for boundary segment extraction
#         if pad_boundary_patches:
#             piano_roll = F.pad(piano_roll, (padding, padding), mode='constant', value=0)
#             segment_boundaries += padding
#             measure_boundaries += padding
#
#         piano_rolls.append(piano_roll)
#
#         # TODO: SSLMs should be precomputed and saved like the piano rolls
#         if return_sslm_near:
#             # Check if precomputed, otherwise load from file or compute
#             if piano_roll_idx in _sslms_near:
#                 sslm_near = _sslms_near[piano_roll_idx]
#             else:
#                 sslm_piano_roll = piano_roll.sum(dim=0, keepdim=False)
#                 sslm_near, sslm_far = compute_sslms(sslm_piano_roll, L=704)  # 88s <=> L=704
#                 _sslms_near[piano_roll_idx] = sslm_near
#                 _sslms_far[piano_roll_idx] = sslm_far
#
#             sslm_near = sslm_near[..., first_nonzero_column:last_nonzero_column + 1]
#             if pad_boundary_patches:
#                 sslm_near = F.pad(sslm_near, (padding, padding), mode='constant', value=0)
#             sslms_near.append(sslm_near)
#
#         if return_sslm_far:
#             if piano_roll_idx in _sslms_far:
#                 sslm_far = _sslms_far[piano_roll_idx]
#             else:
#                 sslm_piano_roll = piano_roll.sum(dim=0, keepdim=False)
#                 _, sslm_far = compute_sslms(sslm_piano_roll, L=704)  # 88s <=> L=704
#                 _sslms_far[piano_roll_idx] = sslm_far
#
#             sslm_far = sslm_far[..., first_nonzero_column:last_nonzero_column + 1]
#             if pad_boundary_patches:
#                 sslm_far = F.pad(sslm_far, (padding, padding), mode='constant', value=0)
#             sslms_far.append(sslm_far)
#
#         for i in measure_boundaries:
#             if not pad_boundary_patches and (i - padding <= 0 or i + padding >= piano_roll.shape[-1]):
#                 continue
#
#             is_segment_boundary = (segment_boundaries == i).any().item()
#             repetitions = positive_oversampling_factor if is_segment_boundary == 1. else int(
#                 random_take(one_in_n=negative_undersampling_factor))
#
#             nearest_segment_boundary = segment_boundaries[torch.argmin(torch.abs(segment_boundaries - i))].item()
#
#             sample = {
#                 # Metadata
#                 "filename": piano_roll_path.stem,
#                 "from": i - padding,
#                 "to": i + padding,
#                 # Data
#                 "piano_roll_idx": piano_roll_idx,
#                 "patch_idx": i,
#                 "is_segment_boundary": is_segment_boundary,
#                 "key": piano_roll_path.parent.parent.stem,  # non_tubb_train, non_tubb_val, tubb_train, tubb_val
#
#                 # New: nearest segment boundary
#                 "nearest_segment_boundary": nearest_segment_boundary
#             }
#
#             # Add segment label immediately to the right of patch center
#             # This is the label of the segment containing position i (or i+epsilon)
#             if segment_labels is not None:
#                 # Find the largest boundary <= i
#                 boundaries_at_or_before = segment_boundaries[segment_boundaries <= i]
#                 if len(boundaries_at_or_before) > 0:
#                     # Find index of the most recent boundary
#                     current_boundary = boundaries_at_or_before[-1]
#                     boundary_idx = torch.where(segment_boundaries == current_boundary)[0][0].item()
#                     sample["segment_label"] = segment_labels[boundary_idx]
#                 else:
#                     # Shouldn't happen if boundaries include first onset
#                     sample["segment_label"] = "Start"
#
#             # Add SSLM patch index if we're returning SSLMs
#             if return_sslm_near:
#                 sample["sslm_near_patch_idx"] = piano_roll_idx
#             if return_sslm_far:
#                 sample["sslm_far_patch_idx"] = piano_roll_idx
#
#             for _ in range(repetitions):
#                 patch_data[sample_idx] = sample
#                 sample_idx += 1
#
#         piano_roll_idx += 1
#
#     return PatchData(
#         piano_rolls=piano_rolls,
#         patch_metadata=patch_data,
#         sslm_near_patches=sslms_near if return_sslm_near else None,
#         sslm_far_patches=sslms_far if return_sslm_far else None
#     )


def concatenate_time_frames_torch(tensor, m=2):
    freq, time = tensor.shape[-2], tensor.shape[-1]
    # Append m-1 frames to the end of the tensor (circulant)
    tensor = torch.cat([tensor, tensor[..., :m]], dim=-1)
    stacked_frames = torch.cat([tensor[..., :, i:time - m + 2 + i] for i in range(m + 1)], dim=-2)

    return stacked_frames


def compute_sslms_from_piano_roll(piano_roll: torch.Tensor, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
    sslm_piano_roll = piano_roll[0] + piano_roll[1]
    sslm_piano_roll = torch.vstack([sslm_piano_roll, piano_roll[2]])
    # L = int((90 / 0.5) * target_ticks_per_beat)  # 90 seconds worth of lags

    sslm_near, sslm_far = compute_sslms(
        sslm_piano_roll,
        L=L
    )
    return sslm_near, sslm_far


def compute_sslms_from_midi_path(p, target_ticks_per_beat):
    piano_roll = create_piano_roll_fast(path_to_midi_file=p,
                                        target_ticks_per_beat=12,
                                        max_target_ticks=9999999999999999999)['piano_roll']
    piano_roll_2 = create_piano_roll_fast(path_to_midi_file=p,
                                          target_ticks_per_beat=target_ticks_per_beat,
                                          max_target_ticks=9999999999999999999)['piano_roll']

    piano_roll_pt = torch.tensor(piano_roll)
    L = 12 * int((90 / 0.5))
    sslm_near, sslm_far = compute_sslms_from_piano_roll(piano_roll=piano_roll_pt, L=L)
    target_temporal_ticks = piano_roll_2.shape[-1]
    sslm_near = torch.nn.functional.interpolate(sslm_near.unsqueeze(0).unsqueeze(0),
                                                size=(sslm_near.shape[0], target_temporal_ticks),
                                                mode='area').squeeze(0).squeeze(0)
    sslm_far = torch.nn.functional.interpolate(sslm_far.unsqueeze(0).unsqueeze(0),
                                               size=(sslm_far.shape[0], target_temporal_ticks),
                                               mode='area').squeeze(0).squeeze(0)
    return sslm_near, sslm_far


def compute_sslms(piano_roll: torch.Tensor, L: int = 720) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns R \in R^{T' x L}: time x lag relationship matrix with adaptive normalization,
    sigmoid smoothing, and value quantization to `n_levels` levels in [0,1].
    """

    X = torch.as_tensor(piano_roll.squeeze(), dtype=torch.float32)  # expected shape [F, T]
    assert X.dim() == 2, f"input_spectrogram must be [F, T], got {X.shape}"

    p = 1  # no downsampling for now
    C = 2  # context frames on each side
    kappa = 0.1  # quantile for adaptive normalization
    num_features = 128

    # ---- 1) Downsample time by max-pooling (factor p) ----
    # treat frequency bins as channels for 1D pooling over time
    X_pool = torch.nn.functional.max_pool1d(X.unsqueeze(0), kernel_size=p, stride=p)  # [1, F, T']
    X_pool = X_pool.squeeze(0)  # [F, T'] = x'
    # T_prime = X_pool.shape[-1]

    # ---- 2) Per-frame DCT-II (drop DC bin) → MFCC-like ----
    # scipy works on CPU; move to CPU temporarily if needed
    X_cpu = X_pool.detach().to("cpu").numpy()
    mfcc_np = scipy.fftpack.dct(X_cpu, type=2, axis=0, norm=None)[1:, :]  # drop 0th coeff → [F_dct-1, T']
    mfcc = torch.from_numpy(mfcc_np).to(dtype=torch.float32)  # [F', T']

    # ---- 3) Context stacking: frames within ±C  (total 2C+1) ----
    # Expect: concatenate_time_frames_torch(mfcc, m=C) → [F'*(2C+1), T'] (aligned centers)
    bagged = concatenate_time_frames_torch(mfcc, m=C)  # user-provided helper
    # L2-normalize each time slice
    X_norm = torch.nn.functional.normalize(bagged, dim=0, eps=1e-8)  # [F_total, T']

    # ---- 4) Full cosine distance and convert to lag matrix ----
    sims = X_norm.T @ X_norm  # [T', T']
    dists = 1.0 - sims  # [T', T']
    T = dists.shape[0]
    max_lag = min(L, T)  # guard if L > T'
    lags = torch.arange(max_lag)  # [L]
    t_idx = torch.arange(T).unsqueeze(0)  # [1, T]
    t_minus_l = (t_idx - lags.unsqueeze(1)) % T  # [L, T]
    # Build D_{t,l} = dcos(x_t, x_{t-l}) with time-circular wrap
    D = dists[t_idx.expand_as(t_minus_l), t_minus_l]  # [L, T]
    D = D.T.contiguous()  # [T, L]  (time x lag)

    # ---- 5) Adaptive normalization with row-wise quantiles (eq. 2) ----
    # q[t] = Q_kappa(D[t, 1..L])
    q = torch.quantile(D, q=kappa, dim=1, keepdim=True)  # [T, 1]
    # epsilon_{t,l} = 0.5 * ( q[t] + q[(t-l) mod T] )
    # Reuse index grid we used earlier but in [L, T] order; rebuild in [T, L]
    l_idx = torch.arange(max_lag).unsqueeze(0).expand(T, -1)  # [T, L]
    t_grid = torch.arange(T).unsqueeze(1).expand(-1, max_lag)  # [T, L]
    t_minus_l_grid = (t_grid - l_idx) % T  # [T, L]
    eps = 0.5 * (q[t_grid] + q[t_minus_l_grid])  # [T, L, 1]
    eps = eps.squeeze(-1)  # [T, L]
    eps = torch.clamp(eps, min=1e-6)

    # ---- 6) Sigmoid smoothing (eq. 3) → relationship matrix R ----
    R = torch.sigmoid(1.0 - (D / eps))  # [T, L], values in (0,1)

    # Subsample features by maxpooling: 6 for sslm_near, 20 for sslm_far
    sslm_near = torch.nn.functional.max_pool1d(R.unsqueeze(0), kernel_size=4, stride=1).squeeze(0)
    sslm_far = torch.nn.functional.max_pool1d(R.unsqueeze(0), kernel_size=16, stride=2).squeeze(0)

    # Limit feature dimension to most similar 100 lags
    # most_similar_near = torch.topk(sslm_near, k=100, dim=1).indices
    # most_similar_far  = torch.topk(sslm_far,  k=100, dim=1).indices
    # sslm_near = torch.gather(sslm_near, 1, most_similar_near)
    # sslm_far  = torch.gather(sslm_far, 1, most_similar_far)
    sslm_near = sslm_near[:, :num_features]
    sslm_far = sslm_far[:, :num_features]

    # Upsample by factor 4 in time to match T' (from p=4 maxpool earlier)
    target_size = (piano_roll.shape[-1], num_features)
    sslm_near = torch.nn.functional.interpolate(sslm_near.unsqueeze(0).unsqueeze(0), size=target_size,
                                                mode='bicubic').squeeze(0).squeeze(0)
    sslm_far = torch.nn.functional.interpolate(sslm_far.unsqueeze(0).unsqueeze(0), size=target_size,
                                               mode='bicubic').squeeze(0).squeeze(0)

    # Transpose to get [num_lags, T]
    sslm_near = sslm_near.T
    sslm_far = sslm_far.T
    return sslm_near, sslm_far


def create_target_activation(
        times: List[float],
        fps: int,
        length: int
) -> torch.Tensor:
    activation = torch.zeros(length)
    active_frames = torch.unique(torch.tensor([int(round(timestamp * fps)) for timestamp in times]))
    active_frames = active_frames[(active_frames >= 0) & (active_frames < length)]
    activation[active_frames] = 1.

    return activation


def widen_temporal_events(events, num_neighbors=2):
    """Widen temporal events by a given number of neighbors."""
    widen_events = events
    for i in range(num_neighbors):
        widen_events = maximum_filter1d(widen_events, size=3)
        neighbor_indices = np.flatnonzero((events != 1) & (widen_events > 0))
        widen_events[neighbor_indices] *= 0.5

    return widen_events


def extract_peaks(events, threshold=0.5):
    device = events.device
    padded = torch.cat((torch.tensor([-float('inf')], device=device),
                        events,
                        torch.tensor([-float('inf')], device=device)))
    b_exceeds_threshold = events > threshold
    b_exceeds_left_neighbor = events > padded[:-2]
    b_exceeds_right_neighbor = events > padded[2:]
    return torch.where(b_exceeds_threshold & b_exceeds_left_neighbor & b_exceeds_right_neighbor)[0]


def load_annotation(p: str):
    """p a path to an annotation txt file"""
    res = []
    with open(p) as infile:
        lines = infile.readlines()
    for line in lines:
        line = line.split('\t')
        line[1] = line[1].rstrip('\n')
        res.append([float(line[0]), str(line[1])])
    return res


def get_midi_path(file_id: str, midi_dirs: list[Path]):
    for folder in midi_dirs:
        path = folder / f"{file_id[0]}" / f"{file_id}.mid"
        if os.path.exists(path):
            return path
        path = folder / f"{file_id}.mid"
        if os.path.exists(path):
            return path
    raise ValueError(f'No path to midi file {file_id} found')


def generic_precision(numerator, n_retrieved):
    if n_retrieved == 0:
        return 0
    else:
        return numerator/n_retrieved


def generic_recall(numerator, n_relevant):
    if n_relevant == 0:
        return None
    else:
        return numerator / n_relevant


def generic_F1(numerator, n_relevant, n_retrieved):
    recall = generic_recall(numerator=numerator, n_relevant=n_relevant)
    precision = generic_precision(numerator=numerator, n_retrieved=n_retrieved)

    if recall is None:  # if there are no relevant documents, F1 is None (undefined)
        return None

    denom = recall + precision
    if denom == 0:
        return 0

    return 2 * precision * recall / denom




