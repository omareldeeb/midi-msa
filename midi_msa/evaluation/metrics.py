import torch


def acc_prec_recall(input: torch.Tensor, target: torch.Tensor):
    tp_count = ((input > 0) & (target == 1)).sum().item()
    fp_count = ((input > 0) & (target == 0)).sum().item()
    tn_count = ((input <= 0) & (target == 0)).sum().item()
    fn_count = ((input <= 0) & (target == 1)).sum().item()

    accuracy = (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count) if tp_count + tn_count + fp_count + fn_count > 0 else 0
    precision = tp_count / (tp_count + fp_count) if tp_count + fp_count > 0 else 0
    recall = tp_count / (tp_count + fn_count) if tp_count + fn_count > 0 else 0

    return accuracy, precision, recall


def compute_metrics(input, target):
    results = {}
    for i in range(input.size(-1)):
        acc, prec, recall = acc_prec_recall(input[..., i], target[..., i])
        results["accuracy_" + str(i)] = acc
        results["precision_" + str(i)] = prec
        results["recall_" + str(i)] = recall
    return results