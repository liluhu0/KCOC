# from torch import *
from matplotlib.transforms import Transform
import torch
import torch.nn.functional as F
import os
import numpy as np
from enum import IntEnum
import cv2
import collections
import threading
import errno
import sys

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(os.path.dirname(fpath))
    torch.save(state, fpath)

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

#***********************************t-SNE utils********************************************
from os.path import abspath, dirname, join

import numpy as np
import scipy.sparse as sp

FILE_DIR = dirname(abspath(__file__))
DATA_DIR = join(FILE_DIR, "data")

MACOSKO_COLORS = {
    "Amacrine cells": "#A5C93D",
    "Astrocytes": "#8B006B",
    "Bipolar cells": "#2000D7",
    "Cones": "#538CBA",
    "Fibroblasts": "#8B006B",
    "Horizontal cells": "#B33B19",
    "Microglia": "#8B006B",
    "Muller glia": "#8B006B",
    "Pericytes": "#8B006B",
    "Retinal ganglion cells": "#C38A1F",
    "Rods": "#538CBA",
    "Vascular endothelium": "#8B006B",
}
ZEISEL_COLORS = {
    "Astroependymal cells": "#d7abd4",
    "Cerebellum neurons": "#2d74bf",
    "Cholinergic, monoaminergic and peptidergic neurons": "#9e3d1b",
    "Di- and mesencephalon neurons": "#3b1b59",
    "Enteric neurons": "#1b5d2f",
    "Hindbrain neurons": "#51bc4c",
    "Immature neural": "#ffcb9a",
    "Immune cells": "#768281",
    "Neural crest-like glia": "#a0daaa",
    "Oligodendrocytes": "#8c7d2b",
    "Peripheral sensory neurons": "#98cc41",
    "Spinal cord neurons": "#c52d94",
    "Sympathetic neurons": "#11337d",
    "Telencephalon interneurons": "#ff9f2b",
    "Telencephalon projecting neurons": "#fea7c1",
    "Vascular cells": "#3d672d",
}
MOUSE_10X_COLORS = {
    0: "#FFFF00",
    1: "#1CE6FF",
    2: "#FF34FF",
    3: "#FF4A46",
    4: "#008941",
    5: "#006FA6",
    6: "#A30059",
    7: "#FFDBE5",
    8: "#7A4900",
    9: "#0000A6",
    10: "#63FFAC",
    11: "#B79762",
    12: "#004D43",
    13: "#8FB0FF",
    14: "#997D87",
    15: "#5A0007",
    16: "#809693",
    17: "#FEFFE6",
    18: "#1B4400",
    19: "#4FC601",
    20: "#3B5DFF",
    21: "#4A3B53",
    22: "#FF2F80",
    23: "#61615A",
    24: "#BA0900",
    25: "#6B7900",
    26: "#00C2A0",
    27: "#FFAA92",
    28: "#FF90C9",
    29: "#B903AA",
    30: "#D16100",
    31: "#DDEFFF",
    32: "#000035",
    33: "#7B4F4B",
    34: "#A1C299",
    35: "#300018",
    36: "#0AA6D8",
    37: "#013349",
    38: "#00846F",
}

MY_COLORS = {
    0: "#FF0000",
    1: "#00FF00",
    2: "#0000FF",
    3: "#00FFFF",
    4: "#FF00FF",
    5: "#006FA6",
    6: "#A30059",
    7: "#FFDBE5",
    8: "#7A4900",   
    9: "#0000A6",
    10: "#63FFAC",
    11: "#B79762",
    12: "#004D43",
    13: "#8FB0FF",
    14: "#997D87",
    15: "#5A0007",
    16: "#809693",
    17: "#FEFFE6",
    18: "#1B4400",
    19: "#4FC601",
    20: "#3B5DFF",
    21: "#4A3B53",
}


def calculate_cpm(x, axis=1):
    """Calculate counts-per-million on data where the rows are genes.
    Parameters
    ----------
    x : array_like
    axis : int
        Axis accross which to compute CPM. 0 for genes being in rows and 1 for
        genes in columns.
    """
    normalization = np.sum(x, axis=axis)
    # On sparse matrices, the sum will be 2d. We want a 1d array
    normalization = np.squeeze(np.asarray(normalization))
    # Straight up division is not an option since this will form a full dense
    # matrix if `x` is sparse. Divison can be expressed as the dot product with
    # a reciprocal diagonal matrix
    normalization = sp.diags(1 / normalization, offsets=0)
    if axis == 0:
        cpm_counts = np.dot(x, normalization)
    elif axis == 1:
        cpm_counts = np.dot(normalization, x)
    return cpm_counts * 1e6


def log_normalize(data):
    """Perform log transform log(x + 1).
    Parameters
    ----------
    data : array_like
    """
    if sp.issparse(data):
        data = data.copy()
        data.data = np.log2(data.data + 1)
        return data

    return np.log2(data.astype(np.float64) + 1)


def pca(x, n_components=50):
    if sp.issparse(x):
        x = x.toarray()
    U, S, V = np.linalg.svd(x, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    x_reduced = np.dot(U, np.diag(S))
    x_reduced = x_reduced[:, np.argsort(S)[::-1]][:, :n_components]
    return x_reduced


def select_genes(
    data,
    threshold=0,
    atleast=10,
    yoffset=0.02,
    xoffset=5,
    decay=1,
    n=None,
    plot=True,
    markers=None,
    genes=None,
    figsize=(6, 3.5),
    markeroffsets=None,
    labelsize=10,
    alpha=1,
):
    if sp.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
        A = data.multiply(data > threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (
            1 - zeroRate[detected]
        )
    else:
        zeroRate = 1 - np.mean(data > threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.nanmean(
            np.where(data[:, detected] > threshold, np.log2(data[:, detected]), np.nan),
            axis=0,
        )

    lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < atleast
    # lowDetection = (1 - zeroRate) * data.shape[0] < atleast - .00001
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan

    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = (
                zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
            )
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low) / 2
            else:
                low = xoffset
                xoffset = (xoffset + up) / 2
        print("Chosen offset: {:.2f}".format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = (
            zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
        )

    if plot:
        import matplotlib.pyplot as plt

        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold > 0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1] + 0.1, 0.1)
        y = np.exp(-decay * (x - xoffset)) + yoffset
        if decay == 1:
            plt.text(
                0.4,
                0.2,
                "{} genes selected\ny = exp(-x+{:.2f})+{:.2f}".format(
                    np.sum(selected), xoffset, yoffset
                ),
                color="k",
                fontsize=labelsize,
                transform=plt.gca().transAxes,
            )
        else:
            plt.text(
                0.4,
                0.2,
                "{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}".format(
                    np.sum(selected), decay, xoffset, yoffset
                ),
                color="k",
                fontsize=labelsize,
                transform=plt.gca().transAxes,
            )

        plt.plot(x, y, linewidth=2)
        xy = np.concatenate(
            (
                np.concatenate((x[:, None], y[:, None]), axis=1),
                np.array([[plt.xlim()[1], 1]]),
            )
        )
        t = plt.matplotlib.patches.Polygon(xy, color="r", alpha=0.2)
        plt.gca().add_patch(t)

        plt.scatter(meanExpr, zeroRate, s=3, alpha=alpha, rasterized=True)
        if threshold == 0:
            plt.xlabel("Mean log2 nonzero expression")
            plt.ylabel("Frequency of zero expression")
        else:
            plt.xlabel("Mean log2 nonzero expression")
            plt.ylabel("Frequency of near-zero expression")
        plt.tight_layout()

        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num, g in enumerate(markers):
                i = np.where(genes == g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color="k")
                dx, dy = markeroffsets[num]
                plt.text(
                    meanExpr[i] + dx + 0.1,
                    zeroRate[i] + dy,
                    g,
                    color="k",
                    fontsize=labelsize,
                )

    return selected


def plot(
    x,
    y,
    cluster_labels=None,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    normalize = False,
    **kwargs
):
    import matplotlib
    
    xx = x
    if normalize:
        x = F.normalize(torch.tensor(x), dim=1)

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(8, 8))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 10)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(xx[mask, :2], axis=0))
        if normalize:
            centers = F.normalize(torch.tensor(np.array(centers)), dim=1)
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        if cluster_labels is None:
            cluster_labels = classes
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=zi,
                markeredgecolor="k",
            )
            for yi,zi in zip(classes, cluster_labels)
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)


def evaluate_embedding(
    embedding, labels, projection_embedding=None, projection_labels=None, sample=None
):
    """Evaluate the embedding using Moran's I index.
    Parameters
    ----------
    embedding: np.ndarray
        The data embedding.
    labels: np.ndarray
        A 1d numpy array containing the labels of each point.
    projection_embedding: Optional[np.ndarray]
        If this is given, the score will relate to how well the projection fits
        the embedding.
    projection_labels: Optional[np.ndarray]
        A 1d numpy array containing the labels of each projection point.
    sample: Optional[int]
        If this is specified, the score will be computed on a sample of points.
    Returns
    -------
    float
        Moran's I index.
    """
    has_projection = projection_embedding is not None
    if projection_embedding is None:
        projection_embedding = embedding
        if projection_labels is not None:
            raise ValueError(
                "If `projection_embedding` is None then `projection_labels make no sense`"
            )
        projection_labels = labels

    if embedding.shape[0] != labels.shape[0]:
        raise ValueError("The shape of the embedding and labels don't match")

    if projection_embedding.shape[0] != projection_labels.shape[0]:
        raise ValueError("The shape of the reference embedding and labels don't match")

    if sample is not None:
        n_samples = embedding.shape[0]
        sample_indices = np.random.choice(
            n_samples, size=min(sample, n_samples), replace=False
        )
        embedding = embedding[sample_indices]
        labels = labels[sample_indices]

        n_samples = projection_embedding.shape[0]
        sample_indices = np.random.choice(
            n_samples, size=min(sample, n_samples), replace=False
        )
        projection_embedding = projection_embedding[sample_indices]
        projection_labels = projection_labels[sample_indices]

    weights = projection_labels[:, None] == labels
    if not has_projection:
        np.fill_diagonal(weights, 0)

    mu = np.asarray(embedding.mean(axis=0)).ravel()

    numerator = np.sum(weights * ((projection_embedding - mu) @ (embedding - mu).T))
    denominator = np.sum((projection_embedding - mu) ** 2)

    return projection_embedding.shape[0] / np.sum(weights) * numerator / denominator
#*******************************************************************************

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None, console = sys.stdout):
        self.console = console
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        count = val.size
        v = val.sum()

        self.count += count
        self.sum += v

        self.avg = self.sum / self.count



def split_by_idxs(seq, idxs):
    '''A generator that returns sequence pieces, seperated by indexes specified in idxs. '''
    last = 0
    for idx in idxs:
        if not (-len(seq) <= idx < len(seq)):
          raise KeyError(f'Idx {idx} is out-of-bounds')
        yield seq[last:idx]
        last = idx
    yield seq[last:]
def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())
def save_model(m, p): torch.save(m.state_dict(), p)
def load_model(m, p):
    sd = torch.load(p, map_location=lambda storage, loc: storage)
    names = set(m.state_dict().keys())
    for n in list(sd.keys()): # list "detatches" the iterator
        if n not in names and n+'_raw' in names and n+'_raw' not in sd:
            sd[n+'_raw'] = sd[n]
            del sd[n]
    m.load_state_dict(sd)

def load_pre(pre, f, fn):
    m = f()
    path = os.path.dirname(__file__)
    if pre: load_model(m, f'{path}/weights/{fn}.pth')
    return m



def to_gpu(x, *args, **kwargs):
    USE_GPU = torch.cuda.is_available()
    '''puts pytorch variable to gpu, if cuda is avaialble and USE_GPU is set to true. '''
    return x.cuda(*args, **kwargs) if USE_GPU else x



def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mean_accuracy(ground_truths, predictions):
    ground_truths = np.array(ground_truths)
    predictions = np.array(predictions)

    class_acc0 = np.sum(
        ground_truths[ground_truths == 0] == predictions[ground_truths == 0]) / np.sum(ground_truths == 0)
    class_acc1 = np.sum(
        ground_truths[ground_truths == 1] == predictions[ground_truths == 1]) / np.sum(ground_truths == 1)
    return class_acc0, class_acc1, (class_acc0+class_acc1) / 2


def multiClassMeanAcc(ground_truths, predictions, class_num):
    ground_truths = np.array(ground_truths)
    predictions = np.array(predictions)

    class_acc = np.zeros(class_num)
    for i in np.arange(class_num):
        class_acc[i] = np.sum(ground_truths[ground_truths == i] == predictions[ground_truths == i]) \
                       / np.sum(ground_truths == i)
    meanAcc = np.mean(class_acc)
    return class_acc, meanAcc

def multiClassPrecision(ground_truths, predictions, class_num):
    
    ground_truths = np.array(ground_truths)
    predictions = np.array(predictions)

    class_precision = np.zeros(class_num)
    for i in np.arange(class_num):
        class_precision[i] = np.sum(ground_truths[ground_truths == i] == predictions[ground_truths == i]) \
                       / np.sum(predictions == i)
    meanPrecision = np.mean(class_precision)
    return class_precision, meanPrecision



def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('denseblock{}'.format(i))
        ft_module_names.append('transition{}'.format(i))
    ft_module_names.append('norm5')
    ft_module_names.append('classifier')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
            else:
                parameters.append({'params': v, 'lr': 0.0})

    return parameters


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def make_weights_for_balanced_classes(DF_train, n_classes):
    nclasses = n_classes
    count = [0] * nclasses
    for i, tempKey in enumerate(range(n_classes)):
        count[i] = np.sum(DF_train['diagnosis'] == tempKey)

    N = float(sum(count))
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(DF_train)
    # classList = [0]*len(DF_train)
    for idx in range(len(DF_train)):
        tempLabel = DF_train.loc[idx, 'diagnosis']
        tempweight = weight_per_class[tempLabel]
        weight[idx] = np.mean(np.array(tempweight))

    return weight

def make_weights_for_balanced_classes_rfmid(DF_train, n_classes):
    nclasses = n_classes
    count = [0] * nclasses
    for i, tempKey in enumerate(range(n_classes)):
        count[i] = np.sum(DF_train['Disease_Risk'] == tempKey)

    N = float(sum(count))
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(DF_train)
    # classList = [0]*len(DF_train)
    for idx in range(len(DF_train)):
        tempLabel = DF_train.loc[idx, 'Disease_Risk']
        tempweight = weight_per_class[tempLabel]
        weight[idx] = np.mean(np.array(tempweight))

    return weight

def make_weights_for_balanced_classes_qilu(DF_train, n_classes):
    nclasses = n_classes
    count = [0] * nclasses
    for i, tempKey in enumerate(range(n_classes)):
        count[i] = np.sum(DF_train['lable'] == tempKey)

    N = float(sum(count))
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(DF_train)
    # classList = [0]*len(DF_train)
    for idx in range(len(DF_train)):
        tempLabel = DF_train.loc[idx, 'lable']
        tempweight = weight_per_class[tempLabel]
        weight[idx] = np.mean(np.array(tempweight))

    return weight

def make_weights_for_balanced_classes_ddr(DF_train, n_classes):
    nclasses = n_classes
    count = [0] * nclasses
    for i, tempKey in enumerate(range(n_classes)):
        count[i] = np.sum(DF_train[:, 1] == tempKey)

    N = float(sum(count))
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(DF_train)
    # classList = [0]*len(DF_train)
    for idx in range(len(DF_train)):
        tempLabel = DF_train[idx, 1]
        tempweight = weight_per_class[tempLabel]
        weight[idx] = np.mean(np.array(tempweight))

    return weight

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        print('update lr: ', param_group['lr'])

import math
def lr_warmup(optimizer, epoch, initialLR, Warmup_epoch):
    if epoch < Warmup_epoch:
        lr = 0.5 * (1. + math.cos(math.pi * (epoch - Warmup_epoch + 1) / Warmup_epoch)) * initialLR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('update lr: ', optimizer.param_groups[-1]['lr'])
        return lr
    else :
        return optimizer.param_groups[-1]['lr']
def lr_warmup_MIL_VT(optimizer, epoch, initialLR, Warmup_epoch):
    if epoch < Warmup_epoch:
        lr = 0.5 * (1. + math.cos(math.pi * (epoch - Warmup_epoch + 1) / Warmup_epoch)) * initialLR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.param_groups[-1]['lr'] = 5 * lr
        print('update lr: ', optimizer.param_groups[0]['lr'], 'MIL_lr', optimizer.param_groups[-1]['lr'])
        return lr
    else :
        return optimizer.param_groups[-1]['lr']

def cosin_learning_rate(optimizer, epoch, total_epochs, lr, cos=True, min_lr = 0, schedule = None):
    """Decay the learning rate based on schedule"""
    if cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / total_epochs))
    lr = max(lr, min_lr)
    # else:  # stepwise lr schedule
    #     for milestone in schedule:
    #         lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

        DF1= pd.read_csv(Name_val, sep=' ') #2502
        DF2= pd.read_csv(Name_test, sep=' ') #3758
        df = DatasetConcat([DF0,DF1,DF2], DFqilu)
        df.to_csv('data/MyDRdataset/part_of_qilu_path_label.csv',
        index=False,header=['image_path','label']) 
    '''
