
from enum import Enum
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import mido
import numpy as np
from scipy.ndimage import maximum_filter1d
import scipy.fftpack
import torch
import torch.nn.functional as F
from tqdm import tqdm


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
    markers_ticks = [int(round(x[0] * ticks_per_beat)) for x in marker_qns]

    markers_ticks = list(set(markers_ticks))
    markers_ticks.sort()

    return markers_ticks


def instrument_overtone_intensities(program, num_harmonics=3, max_harmonic=5):
    """
    Generate a set of harmonics and their intensities for a given instrument program.
    The harmonics are random but fixed for a given program.
    """
    original_seed = np.random.get_state()[1][0]  # Save the original random seed

    np.random.seed(hash(str(program)) % 2**32)

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
    return 440.0 * 2**((midi_note - 69) / 12)


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
    return (torch.randint(0, one_in_n, ()) < 1).bool().item() # type: ignore


def create_lakh_dataset(
    lakh_midi_dir: Union[Path, str],
    data_dir: Union[Path, str],
    files_dict: Dict[str, List[str]],
    markers_qn_path: Union[Path, str],
    target_ticks_per_beat: int = 4,
    instrument_overtones: bool = True,
    separate_drums: bool = True,
    compute_sslm_near: bool = False,
    compute_sslm_far: bool = False
):
    """
    Loads MIDI files from the Lakh MIDI dataset, processes them into piano rolls,
    and saves them in a structured directory format for training, validation, and testing.
    The dataset is split into training, validation, and test sets based on the provided good files.
    The processed data is saved in PyTorch tensor format.
    The directory structure is as follows:
    - DATA_DIR/
        - tubb_train/
        - non_tubb_train/
        - tubb_val/
        - non_tubb_val/
        - tubb_test/
        - non_tubb_test/
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    if isinstance(lakh_midi_dir, str):
        lakh_midi_dir = Path(lakh_midi_dir)

    if not data_dir.exists():
        data_dir.mkdir()

    Path(data_dir / "tubb_train").mkdir(exist_ok=True)
    Path(data_dir / "non_tubb_train").mkdir(exist_ok=True)
    Path(data_dir / "tubb_val").mkdir(exist_ok=True)
    Path(data_dir / "non_tubb_val").mkdir(exist_ok=True)
    Path(data_dir / "tubb_test").mkdir(exist_ok=True)
    Path(data_dir / "non_tubb_test").mkdir(exist_ok=True)

    measure_qns_all = json.load(open("../data/measures_qn.json"))
    for key in files_dict:
        print(f"Processing files: {key}")
        for test_example in tqdm(files_dict[key], desc="Loading test examples"):
            save_path = data_dir / Path(key) / Path(f"{test_example}.pt")
            if save_path.exists():
                continue

            measure_qns = measure_qns_all[test_example]
            midi_path = lakh_midi_dir / Path(f"{test_example[0]}") / Path(test_example + ".mid")
            if not midi_path.exists():
                print(f"Missing MIDI file: {midi_path}")
                continue

            # MIDI
            try:
                track_data, ticks_per_beat, time_signatures = parse_midi(midi_path)
                markers_ticks = parse_markers(
                    markers_qn_path=markers_qn_path,
                    file_id=test_example,
                    ticks_per_beat=ticks_per_beat
                )
            except Exception as e:
                print(f"Error loading MIDI file: {midi_path}")
                print(e)
                continue

            # Annotation
            if ticks_per_beat > target_ticks_per_beat:
                markers_ticks = [int(round(marker * target_ticks_per_beat / ticks_per_beat)) for marker in markers_ticks]
                measure_ticks = [int(round(qn * target_ticks_per_beat)) for qn in measure_qns]
            else:
                print(f"Skipping {test_example} due to downsample factor")
                continue

            piano_rolls = []
            # drum_piano_roll = None  # Separate channel
            for track_name, note_data in track_data.items():
                piano_roll = create_piano_roll(
                    note_data,
                    ticks_per_beat,
                    chroma=False,
                    target_ticks_per_beat=target_ticks_per_beat,
                    instrument_overtones=instrument_overtones,
                    separate_drums=separate_drums
                )
                # Some tracks are empty
                if piano_roll is None:
                    continue
                piano_rolls.append(piano_roll)

            if len(piano_rolls) == 0:
                print(f"Skipping {test_example} due to empty piano rolls")
                continue

            actual_length = max(piano_roll.shape[-1] for piano_roll in piano_rolls)
            for i, piano_roll in enumerate(piano_rolls):
                piano_rolls[i] = torch.nn.functional.pad(torch.tensor(piano_roll), (0, actual_length - piano_roll.shape[-1]))

            piano_roll = torch.stack(piano_rolls)
            # Merge channels
            piano_roll = piano_roll.sum(dim=0).clamp(0, 127)

            data = {
                "piano_roll": piano_roll,
                "segment_boundaries": markers_ticks,
                "measure_ticks": measure_ticks
            }

            if compute_sslm_near:
                sslm_near, _ = compute_sslms(piano_roll)
                data["sslm_near"] = sslm_near
            
            if compute_sslm_far:
                _, sslm_far = compute_sslms(piano_roll)
                data["sslm_far"] = sslm_far

            torch.save(data, save_path)


_sslms_near = dict()
_sslms_far = dict()
def get_piano_roll_patches(
    data_dir: Union[Path, str],
    window_half_ticks: int = 256,
    positive_oversampling_factor: int = 2,
    negative_undersampling_factor: int = 1,
    pad_boundary_patches: bool = True,
    return_sslm_near: bool = False,
    return_sslm_far: bool = False
) -> PatchData:
    """
    Load piano rolls from the specified paths, process them, and return a PatchData object
    containing piano rolls, patch metadata, and optionally SSLM patches.
    Handles positive oversampling and negative undersampling and pads the piano rolls if specified.
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    # Paths defined above in create_lakh_dataset()
    piano_roll_paths = \
        [path for path in (data_dir / "tubb_train").iterdir() if path.suffix == ".pt" and not path.name.startswith(".")] + \
        [path for path in (data_dir / "non_tubb_train").iterdir() if path.suffix == ".pt" and not path.name.startswith(".")] + \
        [path for path in (data_dir / "tubb_val").iterdir() if path.suffix == ".pt" and not path.name.startswith(".")] + \
        [path for path in (data_dir / "non_tubb_val").iterdir() if path.suffix == ".pt" and not path.name.startswith(".")] + \
        [path for path in (data_dir / "tubb_test").iterdir() if path.suffix == ".pt" and not path.name.startswith(".")] + \
        [path for path in (data_dir / "non_tubb_test").iterdir() if path.suffix == ".pt" and not path.name.startswith(".")]

    padding = window_half_ticks

    piano_rolls = []
    sslms_near = []
    sslms_far = []
    patch_data = {}
    sample_idx = 0
    piano_roll_idx = 0

    for piano_roll_path in tqdm(piano_roll_paths, desc="Loading inputs and labels"):
        try:
            data = torch.load(piano_roll_path)
        except RuntimeError:
            print(f"Error loading {piano_roll_path}")
            continue
        
        # Don't oversample/undersample in validation/test sets
        positive_oversampling_factor = positive_oversampling_factor if 'train' in str(piano_roll_path) else 1
        negative_undersampling_factor = negative_undersampling_factor if 'train' in str(piano_roll_path) else 1

        piano_roll = data["piano_roll"]
        segment_boundaries = data["segment_boundaries"]
        measure_boundaries = data["measure_ticks"]

        # Compute first and last nonzero columns of the first channel (first and last onset, respectively)
        if piano_roll.dim() == 4:
            batch_mask = piano_roll[0]  # Select the first batch 
        else:
            batch_mask = piano_roll
        channel_mask = batch_mask[0]  # Select the first channel

        # Find nonzero column indices
        nonzero_indices = channel_mask.nonzero(as_tuple=True)
        if nonzero_indices[1].numel() > 0:
            first_nonzero_column = nonzero_indices[1].min().item()
            last_nonzero_column = nonzero_indices[1].max().item()
        else:
            continue

        # Throw out markers before first onset or after last onset
        segment_boundaries = segment_boundaries[segment_boundaries > first_nonzero_column]
        segment_boundaries = segment_boundaries[segment_boundaries < last_nonzero_column]
        measure_boundaries = measure_boundaries[measure_boundaries > first_nonzero_column]
        measure_boundaries = measure_boundaries[measure_boundaries < last_nonzero_column]

        # Add first and last nonzero column to the segment boundaries
        segment_boundaries = torch.cat([
            torch.tensor([first_nonzero_column], dtype=torch.float32, device=piano_roll.device),
            segment_boundaries,
            torch.tensor([last_nonzero_column], dtype=torch.float32, device=piano_roll.device)
        ])
        measure_boundaries = torch.cat([
            torch.tensor([first_nonzero_column], dtype=torch.float32, device=piano_roll.device),
            measure_boundaries,
            torch.tensor([last_nonzero_column], dtype=torch.float32, device=piano_roll.device)
        ])

        # Crop piano roll to the first and last onset
        piano_roll = piano_roll[..., first_nonzero_column:last_nonzero_column + 1]
        # Adjust segment boundaries to the cropped piano roll
        segment_boundaries -= first_nonzero_column
        measure_boundaries -= first_nonzero_column

        # Pad piano roll to the left and right for boundary segment extraction
        if pad_boundary_patches:
            piano_roll = F.pad(piano_roll, (padding, padding), mode='constant', value=0)
            segment_boundaries += padding
            measure_boundaries += padding

        piano_rolls.append(piano_roll)

        # TODO: SSLMs should be precomputed and saved like the piano rolls
        if return_sslm_near:
            # Check if precomputed, otherwise load from file or compute
            data = torch.load(piano_roll_path)
            if "sslm_near" in data:
                sslm_near = data["sslm_near"]
            elif "sslm" in data:  # Backwards compatibility with old format
                sslm_near = data["sslm"]
            elif piano_roll_idx in _sslms_near:
                sslm_near = _sslms_near[piano_roll_idx]
            else:
                sslm_near = compute_sslm(piano_roll, L=112)  # 14s <=> L=112
                _sslms_near[piano_roll_idx] = sslm_near
            sslms_near.append(sslm_near)
        
        if return_sslm_far:
            # Check if precomputed, otherwise load from file or compute
            data = torch.load(piano_roll_path)
            if "sslm_far" in data:
                sslm_far = data["sslm_far"]
            elif piano_roll_idx in _sslms_far:
                sslm_far = _sslms_far[piano_roll_idx]
            else:
                sslm_far = compute_sslm(piano_roll, L=704)  # 88s <=> L=704
                _sslms_far[piano_roll_idx] = sslm_far
            sslms_far.append(sslm_far)

        for i in measure_boundaries:
            if not pad_boundary_patches and (i - padding <= 0 or i + padding >= piano_roll.shape[-1]):
                continue

            is_segment_boundary = (segment_boundaries == i).any().item()
            repetitions = positive_oversampling_factor if is_segment_boundary == 1. else int(random_take(one_in_n=negative_undersampling_factor))

            nearest_segment_boundary = segment_boundaries[torch.argmin(torch.abs(segment_boundaries - i))].item()

            sample = {
                # Metadata
                "filename": piano_roll_path.stem,
                "from": i - padding,
                "to": i + padding,
                # Data
                "piano_roll_idx": piano_roll_idx,
                "patch_idx": i,
                "is_segment_boundary": is_segment_boundary,
                "key": piano_roll_path.parent.stem, # non_tubb_train, non_tubb_val, tubb_train, tubb_val

                # New: nearest segment boundary
                "nearest_segment_boundary": nearest_segment_boundary
            }
            
            # Add SSLM patch index if we're returning SSLMs
            if return_sslm_near:
                sample["sslm_near_patch_idx"] = piano_roll_idx
            if return_sslm_far:
                sample["sslm_far_patch_idx"] = piano_roll_idx

            for _ in range(repetitions):
                patch_data[sample_idx] = sample
                sample_idx += 1

        piano_roll_idx += 1
    
    return PatchData(
        piano_rolls=piano_rolls,
        patch_metadata=patch_data,
        sslm_near_patches=sslms_near if return_sslm_near else None,
        sslm_far_patches=sslms_far if return_sslm_far else None
    )


def concatenate_time_frames_torch(tensor, m=2):
    freq, time = tensor.shape[-2], tensor.shape[-1]
    # Append m-1 frames to the end of the tensor (circulant)
    tensor = torch.cat([tensor, tensor[..., :m]], dim=-1)
    stacked_frames = torch.cat([tensor[..., :, i:time-m+2+i] for i in range(m+1)], dim=-2)

    return stacked_frames


def compute_sslms(piano_roll: torch.Tensor, L: int = 720) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns R \in R^{T' x L}: time x lag relationship matrix with adaptive normalization,
    sigmoid smoothing, and value quantization to `n_levels` levels in [0,1].
    """

    X = torch.as_tensor(piano_roll.squeeze(), dtype=torch.float32)  # expected shape [F, T]
    assert X.dim() == 2, "input_spectrogram must be [F, T]"

    p = 4 # no downsampling for now
    C = 2 # context frames on each side
    kappa = 0.1 # quantile for adaptive normalization
    num_features = 128

    # ---- 1) Downsample time by max-pooling (factor p) ----
    # treat frequency bins as channels for 1D pooling over time
    X_pool = torch.nn.functional.max_pool1d(X.unsqueeze(0), kernel_size=p, stride=p)  # [1, F, T']
    X_pool = X_pool.squeeze(0)  # [F, T'] = x'
    T_prime = X_pool.shape[-1]

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
    sims = X_norm.T @ X_norm                         # [T', T']
    dists = 1.0 - sims                               # [T', T']
    T = dists.shape[0]
    max_lag = min(L, T)                              # guard if L > T'
    lags = torch.arange(max_lag)      # [L]
    t_idx = torch.arange(T).unsqueeze(0)     # [1, T]
    t_minus_l = (t_idx - lags.unsqueeze(1)) % T              # [L, T]
    # Build D_{t,l} = dcos(x_t, x_{t-l}) with time-circular wrap
    D = dists[t_idx.expand_as(t_minus_l), t_minus_l]         # [L, T]
    D = D.T.contiguous()                                     # [T, L]  (time x lag)

    # ---- 5) Adaptive normalization with row-wise quantiles (eq. 2) ----
    # q[t] = Q_kappa(D[t, 1..L])
    q = torch.quantile(D, q=kappa, dim=1, keepdim=True)      # [T, 1]
    # epsilon_{t,l} = 0.5 * ( q[t] + q[(t-l) mod T] )
    # Reuse index grid we used earlier but in [L, T] order; rebuild in [T, L]
    l_idx = torch.arange(max_lag).unsqueeze(0).expand(T, -1)  # [T, L]
    t_grid = torch.arange(T).unsqueeze(1).expand(-1, max_lag) # [T, L]
    t_minus_l_grid = (t_grid - l_idx) % T                                     # [T, L]
    eps = 0.5 * (q[t_grid] + q[t_minus_l_grid])  # [T, L, 1]
    eps = eps.squeeze(-1)                        # [T, L]
    eps = torch.clamp(eps, min=1e-6)

    # ---- 6) Sigmoid smoothing (eq. 3) → relationship matrix R ----
    R = torch.sigmoid(1.0 - (D / eps))                                        # [T, L], values in (0,1)

    # Subsample features by maxpooling: 6 for sslm_near, 20 for sslm_far
    sslm_near = torch.nn.functional.max_pool1d(R.unsqueeze(0), kernel_size=6, stride=6).squeeze(0)
    sslm_far  = torch.nn.functional.max_pool1d(R.unsqueeze(0), kernel_size=20, stride=20).squeeze(0)

    # Limit feature dimension to most similar 100 lags
    # most_similar_near = torch.topk(sslm_near, k=100, dim=1).indices
    # most_similar_far  = torch.topk(sslm_far,  k=100, dim=1).indices
    # sslm_near = torch.gather(sslm_near, 1, most_similar_near)
    # sslm_far  = torch.gather(sslm_far, 1, most_similar_far)
    sslm_near = sslm_near[:, :num_features]
    sslm_far  = sslm_far[:, :num_features]

    # Upsample by factor 4 in time to match T' (from p=4 maxpool earlier)
    target_size = (piano_roll.shape[-1], num_features)
    sslm_near = torch.nn.functional.interpolate(sslm_near.unsqueeze(0).unsqueeze(0), size=target_size, mode='bicubic').squeeze(0).squeeze(0)
    sslm_far  = torch.nn.functional.interpolate(sslm_far.unsqueeze(0).unsqueeze(0), size=target_size, mode='bicubic').squeeze(0).squeeze(0)

    # Transpose to get [num_lags, T]
    sslm_near = sslm_near.T
    sslm_far  = sslm_far.T
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

class SegmentFunction(str, Enum):
    # TODO
    pass

# Mapping from raw labels to consolidated enum categories
LABEL_MAP = {
    # TODO
}

def consolidate_segment_labels(segment_labels: List[str]) -> List[SegmentFunction]:
    def convert(label: str) -> SegmentFunction:
        key = label.lower()
        if key == "unknown":
            print("consolidate_segment_labels: warning: converting 'unknown' label to SILENCE")
        return LABEL_MAP.get(key, SegmentFunction.OTHER)

    return [convert(label) for label in segment_labels]