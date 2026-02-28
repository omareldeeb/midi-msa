import collections
import copy
import os
import json

LABEL_MAP = {
    "Intro": "Intro",
    "Pre-Verse": "Intro",

    "Outro": "Intro",
    "Coda": "Outro",

    "Verse": "Verse",
    "Pre-Chorus": "Verse",

    "Chorus": "Chorus",
    "Post-Chorus": "Chorus",

    "Bridge": "Bridge",
    "Transition": "Bridge",

    "Solo": "Instrumental",
    "Instrumental": "Instrumental",
    "Interlude": "Instrumental",

    "End": "Start",
    "Start": "Start",
}

LABEL_MAP_TRAIN = {
    "Intro": "Intro",
    "Pre-Verse": "Intro",
    "Outro": "Intro",
    "Coda": "Intro",

    "Verse": "Verse",
    "Pre-Chorus": "Verse",

    "Chorus": "Chorus",
    "Post-Chorus": "Chorus",

    "Bridge": "Bridge",
    "Transition": "Bridge",

    "Solo": "Instrumental",
    "Instrumental": "Instrumental",
    "Interlude": "Instrumental",

    "End": "Start",
    "Start": "Start",
}

LABEL_MAP_VAL = {
    "Intro": "Intro",
    "Pre-Verse": "Intro",

    "Outro": "Outro",
    "Coda": "Outro",

    "Verse": "Verse",
    "Pre-Chorus": "Verse",

    "Chorus": "Chorus",
    "Post-Chorus": "Chorus",

    "Bridge": "Bridge",
    "Transition": "Bridge",

    "Solo": "Instrumental",
    "Instrumental": "Instrumental",
    "Interlude": "Instrumental",

    "End": "Start",
    "Start": "Start",
}

IDENTITY_LABEL_MAP = {
    x: x for x in LABEL_MAP_TRAIN
}

LABEL_MAP_TRAIN['Bridge'] = 'Instrumental'
LABEL_MAP_TRAIN['Transition'] = 'Instrumental'
LABEL_MAP_VAL['Bridge'] = 'Instrumental'
LABEL_MAP_VAL['Transition'] = 'Instrumental'


def _process_pre_verses_in_intro(L: list[tuple[int, float, str]]) -> list[tuple[int, float, str]]:
    res = []
    in_intro = True
    for i, t, label in L:
        if label != 'Intro' and label != 'Pre-Verse':
            in_intro = False
        if in_intro and label == 'Pre-Verse':
            label = 'Intro'
        res.append((i, t, label))
    return res


def _process_remove_consecutive_identical_labels(L: list[tuple[int, float, str]]) -> list[tuple[int, float, str]]:
    res = []
    cur_label = ''
    for i, t, label in L:
        if label != cur_label:
            res.append((i, t, label))
        cur_label = label
    return res


def _process_bridge_to_chorus(L: list[tuple[int, float, str]]) -> list[tuple[int, float, str]]:
    res = []
    counter = collections.Counter()
    for i, t, label in L:
        counter[label] += 1
    if counter['Verse'] >= 1 and counter['Bridge'] >= 1 and counter['Pre-Chorus'] == 0 and counter['Chorus'] == 0 and \
            counter['Post-Chorus'] == 0:
        for i, t, label in L:
            if label == 'Bridge':
                label = 'Chorus'
            res.append((i, t, label))
    else:
        res = L
    return res


def has_overlapping_labels(L: list[list[float | str]], threshold: float = 0.01):
    seen = set()
    for tup in L:
        t, label = tup
        if any(abs(t-x) < threshold for x in seen):
            return True
        seen.add(t)
    return False


def preprocess_labels(L: list[list[float | str]], label_map: dict[str, str],
                      remove_consecutive_identical=False) -> list[list[float | str]]:
    # discard Fadeouts. Don't apply the simplifying label map yet
    working_labels = []
    for i, tup in enumerate(L):
        t, label = tup
        t: float
        label: str
        label = label.split(';')[0].split(' + ')[0]
        if label != 'Fadeout':
            if label != "End":  # discard End label too!
                working_labels.append((i, t, label))

    # Change pre-verse to intro if intro is the only label seen so far (reading left to right).
    # (This is why we don't apply the simplifying label map yet.)
    working_labels = _process_pre_verses_in_intro(working_labels)

    # Apply label simplifying map.
    if any(label not in label_map for (i, t, label) in working_labels):
        print(f"Warning: found unknown labels: {[label for (i, t, label) in working_labels if label not in label_map]}")
    working_labels = [(i, t, label_map.get(label, "Instrumental")) for (i, t, label) in working_labels]

    # change bridge to chorus in certain songs:
    working_labels = _process_bridge_to_chorus(working_labels)

    if remove_consecutive_identical:
        working_labels = _process_remove_consecutive_identical_labels(working_labels)

    final_labels = []
    for tup in working_labels:
        i, t, label = tup
        final_labels.append([t, label])

    return final_labels


def postprocess_labels(L: list[list[float | str]], label_map_out: dict[str, str]):
    res = copy.deepcopy(L)

    def process_outro():  # in place operation
        res.reverse()
        for i, T in enumerate(res):
            t, old_label = T
            t: float
            old_label: str
            if old_label not in ('Coda', 'Outro', 'Intro'):
                res.reverse()
                return
            else:
                if old_label == 'Intro':
                    res[i] = [t, label_map_out['Outro']]
        res.reverse()
        return

    # in place operation - replace "Intro"'s in the Outro with Outro
    process_outro()

    end_functions = ('Outro', 'Coda', 'End')

    def find_middle():
        in_intro = True
        lo_, hi_ = None, None
        for i, T in enumerate(res):
            t, label = T
            if in_intro and label not in ('Intro', 'Start'):
                in_intro = False
                lo_ = i
                continue
            if not in_intro and all(x in end_functions for x in res[i:]):
                hi_ = i + 1
                break
        return lo_, hi_

    lo, hi = find_middle()
    lo = 0 if lo is None else lo
    hi = len(res) if hi is None else hi

    def process_middle(lo_, hi_):
        for i in range(lo_, hi_):
            t, label = res[i]
            if label == 'Intro':
                if i + 1 < len(res) and res[i + 1][1] == 'Verse':
                    res[i] = [t, label_map_out['Pre-Verse']]
                else:
                    res[i] = [t, label_map_out['Instrumental']]

    # in place operation - replace "Intro"'s in the middle with appropriate functions
    process_middle(lo, hi)

    return res


if __name__ == '__main__' or False:
    label_counter = collections.Counter()
    for folder, _, fnames in os.walk(r'SLMS\v1.1.1'):
        for fname in fnames:
            if 'labels_coarse' in fname:
                print(fname)
                labels = json.load(open(os.path.join(folder, fname)))
                assert not has_overlapping_labels(labels)

                L = preprocess_labels(labels, remove_consecutive_identical=False)
                print([label for t, label in L])
                for t, label in L:
                    label_counter[label] += 0.5

                L = preprocess_labels(labels, remove_consecutive_identical=True)
                print([label for t, label in L])

    print('done')