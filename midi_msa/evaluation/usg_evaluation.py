from typing import Dict
import mir_eval.segment
import mir_eval.util
import numpy as np
import torch
from torcheval.metrics.functional import multiclass_f1_score
from tqdm import tqdm
from collections import defaultdict

from ..data import utils
from ..data import label_preprocessor


def _compute_predicted_boundary_ticks(L, logit_threshold=0.0):
    res = [0]
    for record in L:
        if record['boundary_logit'].item() > logit_threshold:
            t = record['center_tick']
            if t not in res:
                res.append(t)

    final_tick = 0
    for record in L:
        final_tick = record['n_ticks_in_file']
        break

    if final_tick not in res:
        res.append(final_tick)

    return res


def _compute_gt_boundary_ticks(L):
    res = [0]
    for record in L:
        if record['is_boundary']:
            t = record['center_tick']
            if t not in res:
                res.append(t)

    final_tick = 0
    for record in L:
        final_tick = record['n_ticks_in_file']
        break

    if final_tick not in res:
        res.append(final_tick)

    return res


def _compute_predicted_labels(L, boundary_ticks, segment_vocab, device, method='vote'):
    if method == 'vote':
        return _compute_predicted_labels_via_vote(L=L, boundary_ticks=boundary_ticks,
                                                  segment_vocab=segment_vocab, device=device)
    elif method == 'single_sample':
        return _compute_predicted_labels_via_boundary_samples(L=L, boundary_ticks=boundary_ticks,
                                                              segment_vocab=segment_vocab, device=device)
    else:
        raise ValueError(f'unknown method: {method}')


def _compute_predicted_labels_via_vote(L, boundary_ticks, segment_vocab, device):
    """Outputs use the model's underlying segment label vocab. No postprocessing of label names is done.'"""
    res = []

    cur_boundary_tick = 0
    for i, record in enumerate(L):
        cur_tick = record['center_tick']
        if i == 0:
            assert cur_tick == 0, 'first record must have center tick = 0'
        if cur_tick in boundary_ticks:
            if i > 0:
                label_idx = torch.argmax(label_votes).item()
                res.append(segment_vocab[label_idx])
            cur_boundary_tick = cur_tick
            label_votes = torch.zeros(size=(len(segment_vocab),), device=device)

        label_votes += record['label_logits'].softmax(dim=-1)

    if cur_boundary_tick < boundary_ticks[-1]:
        label_idx = torch.argmax(label_votes).item()
        res.append(segment_vocab[label_idx])

    assert len(res) == len(boundary_ticks) - 1
    return res


def _compute_predicted_labels_via_boundary_samples(L, boundary_ticks, segment_vocab, device):
    res = []

    cur_boundary_tick = 0
    label = None

    for i, record in enumerate(L):
        cur_tick = record['center_tick']
        if i == 0:
            assert cur_tick == 0, 'first record must have center tick = 0'
        if cur_tick in boundary_ticks or cur_tick == 0:
            if i > 0:
                res.append(label)
            label = torch.argmax(record['label_logits']).item()
            label = segment_vocab[label]

    if cur_boundary_tick < boundary_ticks[-1]:
        res.append(label)

    return res


def _compute_gt_labels(L, boundary_ticks):
    res = []

    cur_boundary_tick = 0
    label = None

    for i, record in enumerate(L):
        cur_tick = record['center_tick']
        if i == 0:
            assert cur_tick == 0, 'first record must have center tick = 0'
        if cur_tick in boundary_ticks or cur_tick == 0:
            if i > 0:
                res.append(label)
            label = record['gt_label']

    if cur_boundary_tick < boundary_ticks[-1]:
        res.append(label)

    return res


def _labels_by_tick(boundary_ticks, labels):
    res = []
    label = 'Start'
    for tick in range(boundary_ticks[-1]):
        if tick in boundary_ticks:
            tick_i = boundary_ticks.index(tick)
            label = labels[tick_i]
        res.append(label)
    return res


def _predicted_labels_by_patch_index(L, t_label_tuples):
    res = []

    ticks = []
    labels = []
    for t, lab in t_label_tuples:
        ticks.append(t)
        labels.append(lab)

    cur_lab = 'Start'
    for i, record in enumerate(L):
        cur_tick = record['center_tick']
        if cur_tick in ticks:
            cur_lab = labels[ticks.index(cur_tick)]
        res.append(cur_lab)
    return res


def validate_usg_model(model, val_loader, label_map_train, segment_vocab_train, label_map_val, segment_vocab_val,
                       device, boundary_criterion, segment_criterion, segment_label_loss_weight,
                       boundary_f1_discard_first_and_last=True,
                       ) -> Dict[str, float]:
    model.eval()

    outputs_by_file_id = defaultdict(list)
    evaluate_labels = False

    total_loss = 0.0
    total_boundary_prec = 0.0
    total_boundary_recall = 0.0
    total_boundary_f1 = 0.0
    total_nce_f1 = 0.0
    total_pairwise_prec = 0.0
    total_pairwise_recall = 0.0
    total_pairwise_f1 = 0.0
    total_label_f1 = {label: 0.0 for label in label_map_val}
    total_label_accuracy = 0.0
    total_label_accuracy_after_segment_picking = 0.0
    accuracy_global_numerator = 0.0
    accuracy_global_denominator = 0.0

    est_boundaries_global = set()
    gt_boundaries_global = set()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch_idx, batch in enumerate(pbar):

            batch = {
                k: v.to(torch.float32).to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                # if v not in ['patch_metadata']
            }

            output = model(
                batch["piano_roll_patch"],
                batch.get("sslm_near_patch"),
                batch.get("sslm_far_patch"),
            )

            boundary_loss = boundary_criterion(
                output["boundary_logits"], batch["targets"].float()
            )
            loss = boundary_loss

            if "segment_label_logits" in output and "segment_label_target" in batch and label_map_val == label_map_train:
                segment_loss = segment_criterion(
                    output["segment_label_logits"], batch["segment_label_target"].long()
                )
                loss = loss + segment_label_loss_weight * segment_loss

            total_loss += loss.item()

            patch_metadata = batch['patch_metadata']
            for i, file_id in enumerate(patch_metadata['file_id']):
                record = {
                    'start_tick': patch_metadata['start_tick'][i].item(),
                    'end_tick': patch_metadata['end_tick'][i].item(),
                    'center_tick': patch_metadata['center_tick'][i].item(),
                    'is_boundary': patch_metadata['is_boundary'][i].item(),
                    'gt_label': patch_metadata['label'][i],
                    'n_ticks_in_file': patch_metadata['n_ticks_in_file'][i].item(),
                    'boundary_logit': output['boundary_logits'][i]  # .to('cpu')
                }
                if 'segment_label_logits' in output:
                    evaluate_labels = True
                    record['label_logits'] = output['segment_label_logits'][i]  # .to('cpu')

                outputs_by_file_id[file_id].append(record)

            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                "batch_loss": loss.item(),
                "avg_loss": avg_loss
            })

        avg_loss = total_loss / len(val_loader)

        for file_id, L in outputs_by_file_id.items():
            L.sort(key=lambda x: x['start_tick'])

            # setup
            predicted_boundary_ticks = _compute_predicted_boundary_ticks(L)  # includes 0 and n_piano_roll_ticks
            gt_boundary_ticks = _compute_gt_boundary_ticks(L)  # includes 0 and n_piano_roll_ticks
            if evaluate_labels:
                predicted_labels = _compute_predicted_labels(L, predicted_boundary_ticks, segment_vocab_train, device,
                                                             method='vote')
                L_ = [[t, label] for t, label in zip(predicted_boundary_ticks, predicted_labels)]
                L_postprocessed = label_preprocessor.postprocess_labels(L=L_, label_map_out=label_map_val)
                predicted_labels = [label for t, label in L_postprocessed]
                predicted_labels_by_patch_index = _predicted_labels_by_patch_index(L=L, t_label_tuples=L_postprocessed)

                gt_labels = _compute_gt_labels(L, gt_boundary_ticks)
                gt_labels_by_patch_index = [record['gt_label'] for record in L]

            # compute boundary metrics
            for i, x in enumerate(gt_boundary_ticks):
                if i == 0 or i == len(gt_boundary_ticks) - 1:
                    if not boundary_f1_discard_first_and_last:
                        gt_boundaries_global.add((x, file_id))
                else:
                    gt_boundaries_global.add((x, file_id))
            for i, x in enumerate(predicted_boundary_ticks):
                if i == 0 or i == len(predicted_boundary_ticks) - 1:
                    if not boundary_f1_discard_first_and_last:
                        est_boundaries_global.add((x, file_id))
                else:
                    est_boundaries_global.add((x, file_id))

            reference_intervals = np.column_stack(
                (gt_boundary_ticks[:-1], gt_boundary_ticks[1:])
            )
            estimated_intervals = np.column_stack(
                (predicted_boundary_ticks[:-1], predicted_boundary_ticks[1:])
            )

            boundary_prec, boundary_recall, boundary_f1 = (
                mir_eval.segment.detection(
                    reference_intervals=reference_intervals,
                    estimated_intervals=estimated_intervals,
                    trim=boundary_f1_discard_first_and_last
                )
            )
            total_boundary_prec += boundary_prec
            total_boundary_recall += boundary_recall
            total_boundary_f1 += boundary_f1

            # compute label-related metrics
            if evaluate_labels:

                # PW F1
                pairwise_prec, pairwise_recall, pairwise_f1 = (
                    mir_eval.segment.pairwise(
                        reference_intervals=reference_intervals,
                        reference_labels=gt_labels,
                        estimated_intervals=estimated_intervals,
                        estimated_labels=predicted_labels,
                        frame_size=(
                            1.0
                        ),
                    )
                )

                # NCE
                # in mir_eval.segment._contingency_matrix, replace dtype=np.int with dtype=int
                score_over, score_under, nce_f_measure = mir_eval.segment.nce(
                    reference_intervals=reference_intervals,
                    reference_labels=gt_labels,
                    estimated_intervals=estimated_intervals,
                    estimated_labels=predicted_labels
                )
                total_nce_f1 += nce_f_measure
                total_pairwise_prec += pairwise_prec
                total_pairwise_recall += pairwise_recall
                total_pairwise_f1 += pairwise_f1

                # setup
                predicted_labels_by_tick = _labels_by_tick(predicted_boundary_ticks, predicted_labels)
                gt_labels_by_tick = _labels_by_tick(gt_boundary_ticks, gt_labels)

                # Patch-wise accuracy (closest analog to tick-wise accuracy)
                accuracy_numerator = sum([1 if a == b else 0 for a, b in zip(gt_labels_by_patch_index, predicted_labels_by_patch_index)])
                accuracy_denominator = len(L)  # number of patches
                accuracy = accuracy_numerator / accuracy_denominator
                total_label_accuracy += accuracy
                accuracy_global_numerator += accuracy_numerator
                accuracy_global_denominator += accuracy_denominator

                # Tick-wise accuracy after segment picking
                a_numerator = sum([1 if a == b else 0 for a, b in zip(gt_labels_by_tick, predicted_labels_by_tick)])
                a_denominator = gt_boundary_ticks[-1]  # number of ticks
                label_accuracy_after_segment_picking = a_numerator / a_denominator
                total_label_accuracy_after_segment_picking += label_accuracy_after_segment_picking

                # Multiclass F1 score
                f1 = multiclass_f1_score(
                    torch.tensor([segment_vocab_val.index(x) for x in predicted_labels_by_tick]),
                    torch.tensor([segment_vocab_val.index(x) for x in gt_labels_by_tick]),
                    num_classes=len(segment_vocab_val),
                    average=None,
                )

                true_labels_set = set(gt_labels_by_tick)
                for label_idx, label in enumerate(segment_vocab_val):
                    if label in true_labels_set:
                        label_f1 = f1[label_idx].item()
                        total_label_f1[label] += label_f1

    metrics = {'loss': avg_loss}
    num_files = len(outputs_by_file_id)
    if num_files > 0:
        boundary_f1_global = utils.generic_F1(numerator=len(est_boundaries_global.intersection(gt_boundaries_global)),
                                              n_relevant=len(gt_boundaries_global),
                                              n_retrieved=len(est_boundaries_global))

        metrics["boundary_precision"] = total_boundary_prec / num_files
        metrics["boundary_recall"] = total_boundary_recall / num_files
        metrics["boundary_f1"] = total_boundary_f1 / num_files
        metrics["boundary_f1_global"] = boundary_f1_global

        if evaluate_labels:
            metrics["pairwise_precision"] = total_pairwise_prec / num_files
            metrics["pairwise_recall"] = total_pairwise_recall / num_files
            metrics["pairwise_f1"] = total_pairwise_f1 / num_files
            metrics["nce"] = total_nce_f1 / num_files

            # Label F1
            for label in segment_vocab_val:
                metrics[f"f1_{label}"] = total_label_f1[label] / num_files

            # Average label F1
            # metrics["average_label_f1"] = np.mean([total_label_f1[label] / num_boundary_batches for label in label_map])

            metrics['label_accuracy'] = total_label_accuracy / num_files
            metrics['label_accuracy_global'] = accuracy_global_numerator / accuracy_global_denominator if accuracy_global_denominator != 0 else 0
            metrics['label_accuracy_after_segment_picking'] = total_label_accuracy_after_segment_picking / num_files

        if evaluate_labels:
            metrics['primary_optimization_metric'] = (metrics['boundary_f1'] + metrics['label_accuracy_after_segment_picking']) / 2
        else:
            metrics['primary_optimization_metric'] = metrics['boundary_f1']

    return metrics

