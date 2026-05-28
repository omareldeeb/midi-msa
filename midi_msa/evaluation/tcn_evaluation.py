from typing import Dict
import mir_eval.segment
import mir_eval.util
import numpy as np
import torch
from torcheval.metrics.functional import multiclass_f1_score
from tqdm import tqdm

from ..data import utils
from ..data import label_preprocessor


def validate_tcn_model(model, val_loader, label_map_train, segment_vocab_train, label_map_val, segment_vocab_val,
                       device, loss_fn,
                       boundary_f1_discard_first_and_last: bool,
                       function_activation: str = "softmax") -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    num_batches = 0

    # Metrics accumulators
    total_beat_f1 = 0.0
    total_downbeat_f1 = 0.0
    total_boundary_prec = 0.0
    total_boundary_recall = 0.0
    total_boundary_f1 = 0.0
    total_pairwise_prec = 0.0
    total_pairwise_recall = 0.0
    total_pairwise_f1 = 0.0
    total_label_f1 = {label: 0.0 for label in label_map_val}
    total_nce_f1 = 0.0
    total_label_accuracy = 0.0
    total_label_accuracy_after_segment_picking = 0.0
    num_boundary_batches = 0
    est_boundaries_global = set()
    gt_boundaries_global = set()
    est_beats_global = set()
    gt_beats_global = set()
    est_downbeats_global = set()
    gt_downbeats_global = set()
    accuracy_global_numerator = 0.0
    accuracy_global_denominator = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            piano_rolls = batch["piano_roll"].to(torch.float32).to(device)
            sslm_near = batch.get("sslm_near")
            sslm_far = batch.get("sslm_far")
            measure_ticks = batch.get("measure_ticks")
            file_ids = batch.get('file_id')
            assert len(file_ids) == 1, 'tcn validation requires batch size == 1'
            file_id = file_ids[0]

            if sslm_near is not None:
                sslm_near = sslm_near.to(torch.float32).to(device)
            if sslm_far is not None:
                sslm_far = sslm_far.to(torch.float32).to(device)

            targets = {
                k: v.to(device)
                for k, v in batch.items()
                if k not in ["piano_roll", "sslm_near", "sslm_far", "measure_ticks", 'file_id',
                             'segment_ticks_in_piano_roll', 'segment_labels_in_piano_roll']
            }

            outputs = model(
                piano_rolls, sslm_near=sslm_near, sslm_far=sslm_far
            )

            # loss only makes sense if label_map_train == label_map_val
            if label_map_train == label_map_val:
                losses = loss_fn(outputs, targets, batch)
                total_loss += losses["total_loss"].item()

            num_batches += 1
            t_max = piano_rolls.shape[-1] - 1

            # Compute beat f1
            # Only compute for single samples (batch_size=1)
            if 'beat_activation' in targets and targets['beat_activation'].shape[0] == 1:
                true_beats = torch.where(targets['beat_activation'].squeeze() == 1.0)[0]
                predicted_beats = utils.extract_peaks(outputs.beat_output.squeeze())
                relevant = set(x.item() for x in true_beats)
                retrieved = set(x.item() for x in predicted_beats)
                beat_f1 = utils.generic_F1(numerator=len(relevant.intersection(retrieved)),
                                           n_relevant=len(relevant),
                                           n_retrieved=len(retrieved))
                total_beat_f1 += beat_f1

                for x in relevant:
                    gt_beats_global.add((x, file_id))
                for x in retrieved:
                    est_beats_global.add((x, file_id))

            # Compute downbeat f1
            # Only compute for single samples (batch_size=1)
            if 'downbeat_activation' in targets and targets['downbeat_activation'].shape[0] == 1:
                true_downbeats = torch.where(targets['downbeat_activation'].squeeze() == 1.0)[0]
                predicted_downbeats = utils.extract_peaks(outputs.downbeat_output.squeeze())
                relevant = set(x.item() for x in true_downbeats)
                retrieved = set(x.item() for x in predicted_downbeats)
                downbeat_f1 = utils.generic_F1(numerator=len(relevant.intersection(retrieved)),
                                               n_relevant=len(relevant),
                                               n_retrieved=len(retrieved))
                total_downbeat_f1 += downbeat_f1

                for x in relevant:
                    gt_downbeats_global.add((x, file_id))
                for x in retrieved:
                    est_downbeats_global.add((x, file_id))

            # Compute boundary and pairwise metrics
            if measure_ticks is not None and "segment_activation" in targets:
                predicted_boundary_ticks, predicted_label_indices = (
                    model.compute_predictions(
                        output=outputs,
                        measure_ticks=measure_ticks,
                        function_activation=function_activation
                    )
                )

                # add one tick beyond the end for stacking purposes
                predicted_boundary_ticks = [int(x) for x in predicted_boundary_ticks]
                if predicted_boundary_ticks and predicted_boundary_ticks[-1] != t_max + 1:
                    predicted_boundary_ticks.append(t_max + 1)

                gt_boundary_ticks = [int(x) for x in batch.get('segment_ticks_in_piano_roll')]
                assert 0 in gt_boundary_ticks, 'gt_boundary_ticks lacks tick 0'
                # add one tick beyond the end for stacking purposes
                if gt_boundary_ticks and gt_boundary_ticks[-1] != t_max + 1:
                    gt_boundary_ticks.append(t_max + 1)

                if len(gt_boundary_ticks) < 2:
                    print('Fewer than 2 ground truth boundary ticks for this batch item. '
                          'Skipping boundary and label computations for this batch item.')
                    continue

                reference_intervals = np.column_stack(
                    (gt_boundary_ticks[:-1], gt_boundary_ticks[1:])
                )

                # if boundary_f1_discard_first_and_last then:
                # always throw out the first tick (whether "Start" or "Verse" or whatever), and
                # always throw out the last tick, whether "End" or not
                for i, x in enumerate(predicted_boundary_ticks):
                    if i == 0 or i == len(predicted_boundary_ticks) - 1:
                        if not boundary_f1_discard_first_and_last:
                            est_boundaries_global.add((x, file_id))
                    else:
                        est_boundaries_global.add((x, file_id))

                for i, x in enumerate(gt_boundary_ticks):
                    if i == 0 or i == len(gt_boundary_ticks) - 1:
                        if not boundary_f1_discard_first_and_last:
                            gt_boundaries_global.add((x, file_id))
                    else:
                        gt_boundaries_global.add((x, file_id))

                estimated_intervals = np.column_stack(
                    (
                        predicted_boundary_ticks[:-1],
                        predicted_boundary_ticks[1:],
                    )
                )

                if len(estimated_intervals) == 0:
                    print('No intervals predicted. Skipping boundary and label computations for this batch item.')
                    continue

                try:
                    # Boundary detection metrics

                    # Will trigger a warning when the model doesn't predict any boundaries other than first and
                    # last tick (happens early in training) - in this case, precision, recall, and f1 are all
                    # returned as 0.0.
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
                    num_boundary_batches += 1

                except Exception as e:
                    print(f'Exception computing boundary metrics, {e}')
                    print('estimated intervals:', estimated_intervals)
                    print('ref int:', reference_intervals)
                    print('file', batch['file_id'])

                # Pairwise metrics (requires labels)
                if (
                    "segment_label_activations" in targets
                    and len(gt_boundary_ticks) > 1
                ):

                    gt_labels = [str(x[0]) for x in batch.get('segment_labels_in_piano_roll')]

                    # Shouldn't need to do this
                    reference_intervals_adj, reference_labels = (
                        mir_eval.util.adjust_intervals(
                            reference_intervals, gt_labels, t_min=0, t_max=t_max + 1
                        )
                    )

                    predicted_labels = [
                        segment_vocab_train[idx] for idx in predicted_label_indices
                    ]

                    L = [[t, label] for t, label in zip(predicted_boundary_ticks, predicted_labels)]
                    L_postprocessed = label_preprocessor.postprocess_labels(L=L, label_map_out=label_map_val)
                    predicted_labels = [label for t, label in L_postprocessed]

                    estimated_intervals_adj, predicted_labels = (
                        mir_eval.util.adjust_intervals(
                            estimated_intervals,
                            predicted_labels,
                            t_min=0,
                            t_max=t_max + 1,
                        )
                    )

                    assert reference_intervals_adj.shape == reference_intervals.shape
                    assert (reference_intervals_adj == reference_intervals).all()

                    assert estimated_intervals_adj.shape == estimated_intervals.shape
                    assert (estimated_intervals_adj == estimated_intervals).all()

                    assert len(reference_intervals_adj) == len(reference_labels)
                    assert len(estimated_intervals_adj) == len(predicted_labels)

                    try:
                        pairwise_prec, pairwise_recall, pairwise_f1 = (
                            mir_eval.segment.pairwise(
                                reference_intervals=reference_intervals_adj,
                                reference_labels=reference_labels,
                                estimated_intervals=estimated_intervals_adj,
                                estimated_labels=predicted_labels,
                                frame_size=(
                                    # (0.1 / 0.5) * self.cfg.target_ticks_per_beat
                                    1.0
                                ),
                            )
                        )

                        # in mir_eval.segment._contingency_matrix, replace dtype=np.int with dtype=int
                        score_over, score_under, nce_f_measure = mir_eval.segment.nce(
                            reference_intervals=reference_intervals_adj,
                            reference_labels=reference_labels,
                            estimated_intervals=estimated_intervals_adj,
                            estimated_labels=predicted_labels
                        )
                        total_nce_f1 += nce_f_measure

                        total_pairwise_prec += pairwise_prec
                        total_pairwise_recall += pairwise_recall
                        total_pairwise_f1 += pairwise_f1
                    except ValueError as err:
                        print(f'Warning: Error in mir_eval.segment computation: {err}')

                    # compute tick-wise label accuracy
                    true_segment_label_idxs = targets['segment_label_activations'][0]
                    if label_map_train == label_map_val:
                        predicted_label_idxs = torch.argmax(outputs.function_outputs[0], dim=0)
                        accuracy_numerator = sum(true_segment_label_idxs == predicted_label_idxs).item()
                        accuracy_denominator = piano_rolls.shape[-1]
                        accuracy = accuracy_numerator / accuracy_denominator
                        total_label_accuracy += accuracy
                        accuracy_global_numerator += accuracy_numerator
                        accuracy_global_denominator += accuracy_denominator

                    # compute tick-wise label accuracy after segment labeling
                    true_labels_by_tick = [segment_vocab_val[x] for x in true_segment_label_idxs]
                    predicted_labels_by_tick = []
                    label = 'Start'
                    for tick in range(piano_rolls.shape[-1]):
                        if tick in predicted_boundary_ticks:
                            tick_i = predicted_boundary_ticks.index(tick)
                            label = predicted_labels[tick_i]
                        predicted_labels_by_tick.append(label)
                    a_numerator = sum([1 if a == b else 0 for a, b in zip(true_labels_by_tick, predicted_labels_by_tick)])
                    a_denominator = piano_rolls.shape[-1]
                    label_accuracy_after_segment_picking = a_numerator/a_denominator
                    total_label_accuracy_after_segment_picking += label_accuracy_after_segment_picking

                    if 0:
                        # Compute F1 for function labels
                        f1 = multiclass_f1_score(
                            torch.tensor([segment_vocab_val.index(x) for x in predicted_labels_by_tick]),
                            torch.tensor([segment_vocab_val.index(x) for x in true_labels_by_tick]),
                            num_classes=len(segment_vocab_val),
                            average=None,
                        )
                        true_labels_set = set(true_labels_by_tick)
                        for label_idx, label in enumerate(segment_vocab_val):
                            if label in true_labels_set:
                                label_f1 = f1[label_idx].item()
                                total_label_f1[label] += label_f1

            if label_map_val == label_map_train:
                pbar.set_postfix({
                    "batch_loss": losses["total_loss"].item(),
                    "avg_loss": total_loss / num_batches,
                })

    metrics = {"loss": total_loss / num_batches} if label_map_val == label_map_train else {}

    if num_boundary_batches > 0:
        boundary_f1_global = utils.generic_F1(numerator=len(est_boundaries_global.intersection(gt_boundaries_global)),
                                              n_relevant=len(gt_boundaries_global),
                                              n_retrieved=len(est_boundaries_global))
        beat_f1_global = utils.generic_F1(numerator=len(est_beats_global.intersection(gt_beats_global)),
                                          n_relevant=len(gt_beats_global),
                                          n_retrieved=len(est_beats_global))
        downbeat_f1_global = utils.generic_F1(numerator=len(est_downbeats_global.intersection(gt_downbeats_global)),
                                              n_relevant=len(gt_downbeats_global),
                                              n_retrieved=len(est_downbeats_global))

        metrics['beat_f1'] = total_beat_f1 / num_boundary_batches
        metrics['beat_f1_global'] = beat_f1_global
        metrics['downbeat_f1'] = total_downbeat_f1 / num_boundary_batches
        metrics['downbeat_f1_global'] = downbeat_f1_global
        metrics["boundary_precision"] = total_boundary_prec / num_boundary_batches
        metrics["boundary_recall"] = total_boundary_recall / num_boundary_batches
        metrics["boundary_f1"] = total_boundary_f1 / num_boundary_batches
        metrics["boundary_f1_global"] = boundary_f1_global
        metrics["pairwise_precision"] = total_pairwise_prec / num_boundary_batches
        metrics["pairwise_recall"] = total_pairwise_recall / num_boundary_batches
        metrics["pairwise_f1"] = total_pairwise_f1 / num_boundary_batches
        metrics["nce"] = total_nce_f1 / num_boundary_batches

        # Label F1
        for label in segment_vocab_val:
            metrics[f"f1_{label}"] = total_label_f1[label] / num_boundary_batches

        # Average label F1
        # metrics["average_label_f1"] = np.mean([total_label_f1[label] / num_boundary_batches for label in label_map])

        metrics['label_accuracy'] = total_label_accuracy / num_boundary_batches
        metrics['label_accuracy_global'] = accuracy_global_numerator / accuracy_global_denominator if accuracy_global_denominator != 0 else 0
        metrics['label_accuracy_after_segment_picking'] = total_label_accuracy_after_segment_picking / num_boundary_batches
        metrics['primary_optimization_metric'] = (metrics['boundary_f1'] + metrics['label_accuracy_after_segment_picking']) / 2
        # TEMPORARY OVERRIDE FOR SWEEP
        # metrics['primary_optimization_metric'] = metrics['boundary_f1']

    return metrics
