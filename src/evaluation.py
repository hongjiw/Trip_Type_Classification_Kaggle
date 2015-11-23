from sklearn.cross_validation import KFold
import numpy

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

def compute_score(p, y):
    assert(p.shape == y.shape)
    p = numpy.maximum(numpy.minimum(p, 1 - 1e-10), 1e-10)
    score = -(numpy.log(p) * y).sum() / len(p)
    return score

def k_fold_validate(clf, X, y_digit, y_vector, k):
    kf = KFold(len(X), n_folds = k)
    score = []
    for train_idx, valid_idx in kf:
        clf = clf.fit(X[train_idx], y_digit[train_idx])
        valid_p = clf.predict_proba(X[valid_idx])
        score.append(compute_score(valid_p, y_vector[valid_idx]))
    return numpy.asarray(score)
        