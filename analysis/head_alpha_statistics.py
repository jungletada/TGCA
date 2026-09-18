"""Image-cluster paired bootstrap and prespecified two-fold alpha selection."""
import hashlib
import numpy as np

from tools.evaluate_cam_threshold_grid import confusion_metrics


def stratified_hash_folds(ids, labels):
    """Deterministic multilabel stratification; SHA256 resolves ALL ties.

    Rarest remaining positive label first, then larger label-count deficit,
    total co-label deficit, capacity deficit, hash parity. Equal fold sizes;
    no masks/scores used.
    """
    labels = np.asarray(labels, bool)
    keys = np.array([hashlib.sha256(('head-alpha-v1:'+s).encode()).hexdigest() for s in ids])
    folds = np.full(len(ids), -1, dtype=int)
    capacity = np.array([len(ids)//2, len(ids)-len(ids)//2])
    desired = np.outer(capacity/len(ids), labels.sum(0))
    while (folds < 0).any():
        remaining = np.flatnonzero(folds < 0)
        counts = labels[remaining].sum(0)
        positive = np.flatnonzero(counts)
        c = int(positive[np.argmin(counts[positive])]) if len(positive) else None
        candidates = remaining[labels[remaining,c]] if c is not None else remaining
        candidates = sorted(candidates, key=lambda i: keys[i])
        for i in candidates:
            available = np.flatnonzero(capacity > 0)
            best = max(available, key=lambda f: (desired[f,c] if c is not None else 0,
                                                 desired[f,labels[i]].sum(), capacity[f],
                                                 int(f == int(keys[i][-1],16)%2)))
            folds[i] = best
            capacity[best] -= 1
            desired[best] -= labels[i]
    return folds


def sufficient(confusion):
    """IoU needs only class TP, GT count and predicted count, not full21x21."""
    return np.stack([np.diagonal(confusion, axis1=-2, axis2=-1),
                     confusion.sum(-1), confusion.sum(-2)], -2).astype(np.float64)


def miou(stats):
    tp, gt, pred = np.moveaxis(stats, -2, 0)
    union = gt + pred - tp
    return np.nanmean(np.divide(tp, union, out=np.full_like(tp, np.nan), where=union>0), -1)


def paired_ci(fixed, reps=5000):
    """fixed:[configs,images,21,21], reference config0. Exact shared draws."""
    stats = sufficient(fixed)
    n = fixed.shape[1]
    flat = stats.transpose(1,0,2,3).reshape(n,-1)
    rng = np.random.default_rng(0)
    samples = []
    for start in range(0,reps,100):
        w = rng.multinomial(n, np.full(n,1/n), size=min(100,reps-start))
        sums = (w @ flat).reshape(-1, len(fixed),3,21)
        samples.append(miou(sums))
    samples = np.concatenate(samples)
    return np.quantile(samples,[.025,.975],axis=0), np.quantile(samples-samples[:,:1],[.025,.975],axis=0)


def crossfit(fixed, folds, reps=5000):
    """config0=native; remaining candidates. Reselect alpha inside EACH draw.

    Draw images separately in the prespecified halves and keep every candidate
    paired. Main number is size-weighted mean of held-out dataset-global mIoUs,
    not mIoU of pooled halves; also return pooled-confusion mIoU as diagnostic.
    """
    stats = sufficient(fixed)
    halves = [stats[:,folds==f].transpose(1,0,2,3) for f in (0,1)]
    weights = np.array([len(h) for h in halves],float) / len(folds)
    def evaluate(sums):
        # [draw,fold,config,3,21]
        values = miou(sums)
        selected = values[:,:,1:].argmax(-1)+1
        heldout = np.stack([values[:,f][np.arange(len(values)),selected[:,1-f]] for f in (0,1)],1)
        score = heldout @ weights
        reference = values[:,:,0] @ weights
        return score, reference, selected
    totals = np.stack([h.sum(0) for h in halves])[None]
    point, ref, selections = evaluate(totals)
    selected = selections[0]
    pooled = miou(sum(totals[0,f,selected[1-f]] for f in (0,1)))
    rng = np.random.default_rng(0)
    samples, selection_samples = [], []
    for start in range(0,reps,100):
        count = min(100,reps-start)
        sums = []
        for h in halves:
            w = rng.multinomial(len(h),np.full(len(h),1/len(h)),size=count)
            sums.append((w@h.reshape(len(h),-1)).reshape(count,len(fixed),3,21))
        score, reference, sel = evaluate(np.stack(sums,1))
        samples.append(np.stack([score,reference,score-reference],1))
        selection_samples.append(sel)
    ci = np.quantile(np.concatenate(samples),[.025,.975],axis=0)
    return dict(selected_candidate_indices=selected.tolist(), point=float(point[0]), reference=float(ref[0]),
                delta=float(point[0]-ref[0]), pooled_heldout_miou=float(pooled), ci95=ci.tolist(),
                bootstrap_reselects=True, fold_sizes=[len(h) for h in halves],
                selection_counts=[np.bincount(np.concatenate(selection_samples)[:,f],minlength=len(fixed)).tolist() for f in (0,1)])


def metric_rows(totals):
    rows=[]
    for total in totals:
        m=confusion_metrics(total)
        best=int(np.nanargmax(m['mean_iou']))
        fg=total[45,:,1:].sum()/total[45].sum()
        rows.append(dict(fixed_mean_iou_percent=float(100*m['mean_iou'][45]),
                         best_mean_iou_percent=float(100*m['mean_iou'][best]), best_threshold=best/100,
                         foreground_precision_percent=float(100*m['semantic_foreground_precision'][45]),
                         foreground_recall_percent=float(100*m['semantic_foreground_recall'][45]),
                         foreground_area_fraction_valid=float(fg)))
    return rows


def unlabeled_ranking(means, criterion):
    """Freeze before CAM results: small gamma=class-specific; small kappa=compact."""
    index = {'kappa':0,'gamma':2}[criterion]
    return np.argsort(means[:,index],kind='stable').tolist()
