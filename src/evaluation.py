def compute_accuracy(preds, gts):
    assert(len(preds) == len(gts))

    tp = 0
    fp = 0

    for pred, gt in zip(preds, gts):
        if pred == gt:
            tp += 1
        else:
            fp += 1

    accuracy = float(tp)/(tp+fp)

    return accuracy
