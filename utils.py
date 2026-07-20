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

#*******************************自动选择空闲GPU***************************************
import pynvml
pynvml.nvmlInit()
def ChooseGPU(GpuNumb):
    total = pynvml.nvmlDeviceGetCount()
    gpu_free = []
    for i in range(total):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_free.append(meminfo.free/1024**2)
    gpu_id = sorted(range(len(gpu_free)), key=lambda k: gpu_free[k],reverse = True)[0:GpuNumb]
    return gpu_id

def ChooseGPU_Memory(Memory):
    total = pynvml.nvmlDeviceGetCount()
    gpu_free = []
    for i in range(total):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_free.append(meminfo.free/1024**2)
    if max(gpu_free)>Memory:
        return True
    else:
        return False

#*******************************MinCrop************************************************
from PIL import Image, ImageFilter

def Mincrop(img):
    ba = np.array(img.filter(ImageFilter.BLUR))
    h, w, _ = ba.shape

    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        foreground = (ba > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if bbox is None:
            # print('bbox none for {} (???)'.format('the imamg'))
            pass
        else:
            left, upper, right, lower = bbox
            # if we selected less than 80% of the original 
            # height, just crop the square
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                # print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None

    if bbox is None:
        bbox = square_bbox(img)

    cropped = img.crop(bbox)
    return cropped

def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)
def no_bbox(img):
    w, h = img.size
    return (0, 0, w, h)


class MinCrop(torch.nn.Module):
    r"""crop the fundus image to cut out the black area.

    Notes:
        - If the image is smaller than the crop size, return the original image.
        - If efficientnet_style is set to False, the pipeline would be a simple
          center crop using the crop_size.
        - If efficientnet_style is set to True, the pipeline will be to first
          to perform the center crop with the ``crop_size_`` as:

        .. math::
            \text{crop\_size\_} = \frac{\text{crop\_size}}{\text{crop\_size} +
            \text{crop\_padding}} \times \text{short\_edge}

        And then the pipeline resizes the img to the input crop size.
    """


    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        return Mincrop(img)

    def __repr__(self):
        return 'crop the fundus image to cut out the black area'

#***************************************************************************************
import random
def minBlur_patch(image, kernel=(3, 3), limit=(0, 255), patch_size = 32):
    image_c = image.copy()
    if len(image_c.shape) == 2:
        image_c = image_c[:, :, np.newaxis]
    h, w, c = image_c.shape
    r1 = random.randint(0,h-patch_size[0])
    r2 = random.randint(0,w-patch_size[1])
    image_c1 = image_c.copy()
    for i in range(r1, r1+patch_size[0]):
        for j in range(r2, r2+patch_size[1]):
            x1 = max(j-kernel[0]//2, 0)
            x2 = min(x1 + kernel[0], w)
            y1 = max(i-kernel[1]//2, 0)
            y2 = min(y1 + kernel[1], h)
            for k in range(c):
                if image_c[i, j, k] >= limit[0] and image_c[i, j, k] <= limit[1]:
                    sub_img = image_c1[y1:y2, x1:x2, k]
                    image_c[i, j, k] = np.min(sub_img)
    if len(image.shape) == 2:
        image_c = image_c.reshape(h, w)
    return image_c

class MinBlur_patch(torch.nn.Module):
    def __init__(self, kernel=(3, 3), patch_size = (64,64)):
        super().__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size)
        self.kernel = kernel
        self.patch_size = patch_size
    def forward(self, img):
        img = np.array(img)
        # 极小值滤波
        img = minBlur_patch(img, kernel=self.kernel, patch_size = self.patch_size)
        return Image.fromarray(img)

import PIL
class Rotation(torch.nn.Module):
    def forward(self, img):
        Rotation = random.randint(0,1) * 180
        return transforms.RandomRotation([Rotation, Rotation])(img)
class NewRotation(torch.nn.Module):
    def __init__(self, n, r = [0,90,180,270]):
        super().__init__()
        self.n = n
        self.r = r
    def forward(self, img):
        Rotation = self.r[random.randint(0, self.n-1)]
        return transforms.RandomRotation([Rotation, Rotation])(img)

#************************************设置随机种子******************************************
import random
def setup_seed(seed, benchmark = True):
    torch.manual_seed(seed)          # 为CPU设置随机种子
    torch.cuda.manual_seed_all(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed(seed)     # 为所有GPU设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    if benchmark:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True  # 固定卷积算法
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True #for accelerating the running，cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率


#***********************************DDR、qilu合并****************************************

def DatasetConcat(ddrdataset, qiludataset, data_path='data/Qilu'):
    data_df = pd.DataFrame({'image_path':[], 'label':[]})
    data_df['image_path'] = data_df['image_path'].astype(str)
    data_df['label'] = data_df['label'].astype(int)
    # for i in range(len(ddrdataset)):
    #     dataset = ddrdataset[i].values
    #     for index in range(len(ddrdataset[i])):
    #         imgName = os.path.join(data_path_ddr[i], str(dataset[index, 0]))
    #         label = dataset[index, 1]
    #         data_df = pd.DataFrame(np.insert(data_df.values, len(data_df.index), values=[imgName, label], axis=0))

    for index in range(len(qiludataset)):
        Date = np.str(qiludataset.loc[index, '上传日期'])
        peple_name = qiludataset.loc[index, '患者姓名']
        img_path = os.path.join(data_path, Date[0:4]+ '/' + Date[0:4] + Date[5:7] + Date[8:10])

        for root, dirs, files in os.walk(img_path):
            for file in files:
                a_test = file.split('_')[0]
                if file.split('_')[0]==peple_name:
                    imgName = img_path + '/' + file
                    imgName = imgName.replace('\\', '/')
                    label = qiludataset.loc[index, 'lable']
                    data_df = pd.DataFrame(np.insert(data_df.values, len(data_df.index), values=[imgName, label], axis=0))
    return data_df



#************************************输出MIL的热图******************************************
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms

def heatmap(Img, heatmap, label, epoch = 0, batch_idx = 0, save_path = None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.rcParams['font.sans-serif']=['SimHei']  # 用于显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用于显示中文
    df = pd.DataFrame(heatmap)
                    #   index=[chr(i) for i in range(65, 90)],#DataFrame的行标签设置为大写字母
                    #   columns=["a","b","c","d","e","f"])
    fig = plt.figure(figsize=(8,3), dpi=120)
    ax1 = fig.add_subplot(121)
    ax1.imshow(Img)
    plt.title('fundus image-label%d'%(label))
    ax2 = fig.add_subplot(122)
    sns.heatmap(data=df,  #矩阵数据集，数据的index和columns分别为heatmap的y轴方向和x轴方向标签
                # vmin = 5, vmax = 8,  # 图例（右侧颜色条color bar）中最小显示值, 图例（右侧颜色条color bar）中最大显示值
                # cmap=plt.get_cmap('Greens'),#matplotlib中的颜色盘'Greens'
                # center=7,#color bar的中心数据值大小，可以控制整个热图的颜盘深浅
                # annot=True,#默认为False，当为True时，在每个格子写入data中数据
                # fmt=".2f",#设置每个格子中数据的格式，此处保留两位小数
                # annot_kws={'size':8,'weight':'normal', 'color':'blue'},#设置格子中数据的大小、粗细、颜色
                # cbar=False,#右侧图例(color bar)开关，默认为True显示
                # cbar_kws={'label': 'ColorbarName',  # color bar的名称
                #           'orientation': 'horizontal',  # color bar的方向设置，默认为'vertical'，可水平显示'horizontal'
                #           "ticks": np.arange(4.5, 8, 0.5),  # color bar中刻度值范围和间隔
                #           "format": "%.3f",  # 格式化输出color bar中刻度值
                #           "pad": 0.15,  # color bar与热图之间距离，距离变大热图会被压缩
                #           },
                # mask=df<6.0,#热图中显示部分数据：显示数值小于6的数据
                # xticklabels=['三连啊', '关注公众号啊', 'pythonic生物人', '收藏啊', '点赞啊', '老铁三连三连'],
                # # x轴方向刻度标签开关、赋值，可选“auto”, bool, list-like（传入列表）, or int,
                # yticklabels=True,  # y轴方向刻度标签开关、同x轴
                )
    plt.title('heatmap-epoch%d-batch%d'%(epoch, batch_idx))
    if save_path is not None:
        fig.savefig(save_path + '/epoch%d_batch%d_lable%d'%(epoch, batch_idx, label))

def makedirs(save_model_path, path='outputs'):
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    if path is not None:
        save_log_path = os.path.join(path, os.path.basename(os.path.dirname(save_model_path)))
        if not os.path.exists(save_log_path):
            os.makedirs(save_log_path)


#*****************************************输出CNN的热图********************************
import os
import torch
import torchvision.models as models
from timm.models.resnet import resnet50
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# os.environ["KMP_DUPLICATE_LIB_OK"]="True"

#单幅图
def draw_cam(model, img_path, save_path,  visheadmap=True, epoch = 0, batch_idx = 0, label = 0, avgpool=False, trans=False, threshold=0):
    if hasattr(model, 'module'):
        model = model.module
    else:
        model = model

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig = plt.figure(figsize=(18,6))
    if trans is not False:
        T = trans
    else:
        T = transforms.Compose([
            transforms.Resize(512+100),
            MinCrop(),
            transforms.Resize((512, 512))])
    T2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    im = Image.open(img_path).convert('RGB')
    im = T(im)
    ax1 = fig.add_subplot(131);ax1.imshow(T(im))
    plt.title('fundus image-label%d'%(label))

    img = T2(im).cuda()
    img = img.unsqueeze(0)
    model.eval()
    x = model.forward_features(img)
    features = x                #1x2048x7x7
    # print(features.shape)
    if avgpool:
        output = model.avgpool(x)
    else:
        output = model.global_pool(x)   #1x2048x1x1
    # print(output.shape)
    output = output.view(output.size(0), -1)
    # print(output.shape)         #1x2048
    output = model.fc(output)   #1x1000
    # print(output.shape)
    def extract(g):
        global feature_grad
        feature_grad = g
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
    features.register_hook(extract)
    pred_class.backward()
    greds = feature_grad
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(greds, (1, 1))
    pooled_grads = pooled_grads[0]
    features = features[0]
    for i in range(2048):
        features[i, ...] *= pooled_grads[i, ...]
    headmap = features.cpu().detach().numpy()
    headmap = np.mean(headmap, axis=0)
    headmap /= np.max(headmap)
    # headmap[headmap>0.0] = 1
    headmap[headmap<=threshold] = 0
    ax2 = fig.add_subplot(132)
    if visheadmap:
        ax2.matshow(headmap)
    plt.title('heatmap-epoch%d-batch%d'%(epoch, batch_idx))

    img = np.array(im)
    headmap = cv2.resize(headmap, (img.shape[1], img.shape[0]))
    headmap = np.uint8(255*headmap)
    ax1.imshow(headmap)
    headmap = cv2.applyColorMap(headmap, cv2.COLORMAP_JET)
    cam = np.float32(headmap) + np.float32(img)
    superimposed_img = np.uint8(255 * cam/np.max(cam))
    # cv2.imwrite(save_path, superimposed_img)
    superimposed_img = Image.fromarray(superimposed_img)
    ax3 = fig.add_subplot(133)
    ax3.imshow(superimposed_img)
    plt.title('fundus image-label%d+heatmap-epoch%d-batch%d'%(label, epoch, batch_idx))
    plt.savefig(save_path + '/epoch%d_batch%d_lable%d'%(epoch, batch_idx, label))
    
    plt.cla()
    plt.close("all")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

@torch.enable_grad()
def cam_AttentionMap(model, img1, img_path, label, save_path, epoch=0, batch_idx=0, visheadmap=True):
        # img2是仅仅经过裁剪和resize的图像，img3是变成tensor和normalize后的图像
    if hasattr(model, 'module'):
        model = model.module
    else:
        model = model
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    T = transforms.Compose([
        transforms.Resize(512+100),
        MinCrop(),
        transforms.Resize((512, 512))])
    img2 = T(Image.open(img_path).convert('RGB'))

    model.eval()
    img1 = img1.unsqueeze(0)
    x = model.forward_features(img1)
    features = x                #Bx2048x7x7
    output = model.global_pool(x)   #Bx2048x1x1
    output = output.view(output.size(0), -1)#Bx2048
    output = model.fc(output)   #Bx1000
    def extract(g):
        global feature_grad
        feature_grad = g
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
    features.register_hook(extract)
    pred_class.backward()
    greds = feature_grad
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(greds, (1, 1))
    pooled_grads = pooled_grads[0]
    features = features[0]
    for i in range(2048):
        features[i, ...] *= pooled_grads[i, ...]
    headmap = features.cpu().detach().numpy()
    headmap = np.mean(headmap, axis=0)
    headmap /= np.max(headmap)
    # headmap[headmap>0.1] = 1
    # headmap[headmap<=0.1] = 0
    
    img = np.array(img2)
    hmap0 = (headmap-np.min(headmap))/(np.max(headmap)-np.min(headmap)+1e-8)
    hmap = cv2.resize(hmap0, (img.shape[1], img.shape[0]))

    if visheadmap:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig = plt.figure(figsize=(18,6))
        ax1 = fig.add_subplot(131);ax1.imshow(img2)
        plt.title('fundus image-label%d'%(label))
        ax2 = fig.add_subplot(132);ax2.matshow(headmap)
        plt.title('heatmap-epoch%d-batch%d-pred%d'%(epoch, batch_idx, pred_class))

        superimposed_img = np.zeros(img.shape)
        for i in range(3):
            superimposed_img[:,:,i] = img[:,:,i] * (hmap+0.5)/1.5
        # headmap = np.uint8(255*headmap)
        # headmap = cv2.applyColorMap(headmap, cv2.COLORMAP_JET)
        # cam = np.float32(headmap) + np.float32(img)
        # superimposed_img = np.uint8(255 * cam/np.max(cam))
        # cv2.imwrite(save_path, superimposed_img)
        superimposed_img = Image.fromarray(np.uint8(superimposed_img))
        ax3 = fig.add_subplot(133);ax3.imshow(superimposed_img)
        plt.title('fundus image-label%d+heatmap-epoch%d-batch%d'%(label, epoch, batch_idx))
        plt.savefig(save_path + '/epoch%d_batch%d_lable%d'%(epoch, batch_idx, label))

        plt.cla()
        plt.close("all")
    return hmap
'''
def cam_AttentionMap(model, imgs1, img_path, labels, save_path, epoch=0, batch_idx=0, visheadmap=True):
    # img2是仅仅经过裁剪和resize的图像，img3是变成tensor和normalize后的图像
    if hasattr(model, 'module'):
        model = model.module
    else:
        model = model

    fig = plt.figure(figsize=(18,6))
    T = transforms.Compose([
        transforms.Resize(512+100),
        MinCrop(),
        transforms.Resize((512, 512))])
    imgs2 = T(Image.open(img_path[0]).convert('RGB'))
    ax1 = fig.add_subplot(131);ax1.imshow(imgs2)
    plt.title('fundus image-label%d'%(labels[0]))

    model.eval()
    x = model.forward_features(imgs1)
    features = x                #Bx2048x7x7
    output = model.global_pool(x)   #Bx2048x1x1
    output = output.view(output.size(0), -1)#Bx2048
    output = model.fc(output)   #Bx1000
    def extract(g):
        global feature_grad
        feature_grad = g
    pred = torch.argmax(output, axis=1).item()
    pred_class = output[:, pred]
    features.register_hook(extract)
    pred_class.backward()
    greds = feature_grad
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(greds, (1, 1))
    pooled_grads = pooled_grads[0]
    # features = features[0]
    for i in range(2048):
        features[:,i, ...] *= pooled_grads[:,i, ...]
    headmaps = features.cpu().detach().numpy()
    headmaps = np.mean(headmaps, axis=1)
    headmaps_max = np.max(headmaps, axis=(1,2))
    for i in range(headmaps.shape[0]):
        headmaps[i] /= headmaps_max[i]
    # headmap[headmap>0.1] = 1
    # headmap[headmap<=0.1] = 0
    ax2 = fig.add_subplot(132)
    if visheadmap:
        ax2.matshow(headmaps[0])
    plt.title('heatmap-epoch%d-batch%d'%(epoch, batch_idx))

    img = np.array(imgs2)
    hmap = np.zeros(headmaps.shape)
    for i in range(headmaps.shape[0]):
        headmaps[i] = cv2.resize(headmaps[i], (img.shape[1], img.shape[2]))
        hmap[i] = (headmaps[i]-np.min(headmaps[i]))/(np.max(headmaps[i])-np.min(headmaps[i]))

    # cam = np.zeros(img.shape)
    # superimposed_img = cam
    # for j in range(img.shape[0]):
    #     for i in range(3):
    #         cam[j,:,:,i] = img[j,:,:,i] * hmap[j]
    #         superimposed_img[j] = Image.fromarray(np.uint8(255 * cam[j]/np.max(cam[j])))

    cam = np.zeros(img.shape[1:])
    for i in range(3):
        cam[:,:,i] = img[0,:,:,i] * hmap[0]
    superimposed_img = np.uint8(255 * cam/np.max(cam))
    ax3 = fig.add_subplot(133)
    ax3.imshow(superimposed_img[0])
    plt.title('fundus image-label%d+heatmap-epoch%d-batch%d'%(labels[0], epoch, batch_idx))
    plt.savefig(save_path + '/epoch%d_batch%d_lable%d'%(epoch, batch_idx, labels[0]))
    
    plt.cla()
    plt.close("all")
    return hmap
'''
def pnorm(weights, p):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    return ws

#************************************Computes cos simillarity of gradients*******************************
def get_grad(model, key):  
    try:
        grad = []
        for i,(k,v) in enumerate(model.state_dict(keep_vars=True).items()):
            if (k.startswith(key)) and (v.requires_grad):
                grad.append(v.grad.reshape((-1)))
        return torch.cat(grad, axis = 0)
    except:
        return None
        # grad = torch.cat(
        #     [v.grad.reshape((-1))
        #         for i,(k,v) in enumerate(model.state_dict(keep_vars=True).items())
        #         if (key.startswith(k)) and (v.requires_grad)], 
        #     axis = 0)
    

def get_grad_cos_sim(grad1, grad2):
    """Computes cos simillarity of gradients after flattening of tensors.
    
    It hasn't been stated in paper if batch normalization is considered as model trainable parameter,
    but from my perspective only convolutional layer's cosine similarities should be measured.
    """
    # perform min(max(-1, dist),1) operation for eventual rounding errors (there's about 1 every epoch)
    try:
        dist = float(torch.cosine_similarity(grad1,grad2,axis = 0))
    except:
        dist=None
    return dist


def get_model_grad_simillarity(model, loss1, loss2):
    # except_name = ['fc1','fc2'] 字符串数组
    dict1 = dict()
    dict2 = dict()
    dict3 = dict()
    model.zero_grad(set_to_none=True)
    loss1.backward(retain_graph=True)
    for name, layer in model._modules.items():
        dict1[name] = get_grad(model, name)
    model.zero_grad(set_to_none=True)
    loss2.backward(retain_graph=True)
    for name, layer in model._modules.items():
        dict2[name] = get_grad(model, name)
    model.zero_grad(set_to_none=True)
    for name, layer in model._modules.items():
        dict3[name] = get_grad_cos_sim(dict1[name], dict2[name])
    return dict3
#************************************************************************************************

# K近邻算法 
def kNN(net, queue, queue_labels, trainloader, testloader, K, sigma, recompute_memory=False, class_number=5, only_feature=False):
    net.eval()
    total = 0

    with torch.no_grad():
        if recompute_memory:
            transform_bak = trainloader.dataset.transform
            trainloader.dataset.transform = testloader.dataset.transform
            temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=4)
            for batch_idx, (inputs, _,_) in enumerate(temploader):
                
                batchSize = inputs.size(0)
                inputs = inputs.cuda()

                _,features = net(inputs)
                if batch_idx == 0:
                    trainFeatures = features.data.t()
                else:
                    trainFeatures = torch.cat((trainFeatures, features.data.t()), 1)
                    
            try:
                trainLabels = torch.LongTensor(temploader.dataset.clean_labels).cuda()
            except:
                trainLabels = torch.LongTensor(temploader.dataset.targets).cuda()
            trainloader.dataset.transform = transform_bak
        else:
            trainFeatures = queue
            trainLabels = queue_labels
        C = class_number
        top1 = 0.
        top5 = 0.

        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, _) in enumerate(testloader):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            inputs = inputs.cuda()
            x1, x2= net(inputs, inputs, targets, True)
            features = x2 if only_feature else x1

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            if torch.min(retrieval)<0:
                retrieval[retrieval<0] = 3

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = torch.exp(torch.div(yd.clone(), sigma))
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)

    return top1/total, top5/total
def kNN_ViT(net, queue, queue_labels, trainloader, testloader, K, sigma, recompute_memory=False, class_number=5, only_feature=False):
    net.eval()
    total = 0

    with torch.no_grad():
        if recompute_memory:
            transform_bak = trainloader.dataset.transform
            trainloader.dataset.transform = testloader.dataset.transform
            temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=4)
            for batch_idx, (inputs, _,_) in enumerate(temploader):
                
                batchSize = inputs.size(0)
                inputs = inputs.cuda()

                _,features = net(inputs)
                if batch_idx == 0:
                    trainFeatures = features.data.t()
                else:
                    trainFeatures = torch.cat((trainFeatures, features.data.t()), 1)
                    
            try:
                trainLabels = torch.LongTensor(temploader.dataset.clean_labels).cuda()
            except:
                trainLabels = torch.LongTensor(temploader.dataset.targets).cuda()
            trainloader.dataset.transform = transform_bak
        else:
            trainFeatures = queue
            trainLabels = queue_labels
        C = class_number
        top1 = 0.
        top5 = 0.

        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, _) in enumerate(testloader):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            inputs = inputs.cuda()
            x1, [x2, _] = net(inputs, inputs, targets, MIL=False, eval=True)
            features = x2 if only_feature else x1

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            if torch.min(retrieval)<0:
                retrieval[retrieval<0] = 3

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = torch.exp(torch.div(yd.clone(), sigma))
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)
        # ── 清理 ─────────────────────────────────────────────────
        del dist, yd, yi, features, candidates, retrieval, yd_transform, probs, predictions, correct, inputs, targets
        del retrieval_one_hot
        del trainFeatures, trainLabels
    return top1/total, top5/total
def kNN2(net, queue, queue_labels, trainloader, testloader, K, sigma, recompute_memory=False, class_number=5, only_feature=False):
    net.eval()
    total = 0

    with torch.no_grad():
        if recompute_memory:
            transform_bak = trainloader.dataset.transform
            trainloader.dataset.transform = testloader.dataset.transform
            temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=4)
            for batch_idx, (inputs, _,_) in enumerate(temploader):
                
                batchSize = inputs.size(0)
                inputs = inputs.cuda()

                _,features = net(inputs)
                if batch_idx == 0:
                    trainFeatures = features.data.t()
                else:
                    trainFeatures = torch.cat((trainFeatures, features.data.t()), 1)
                    
            try:
                trainLabels = torch.LongTensor(temploader.dataset.clean_labels).cuda()
            except:
                trainLabels = torch.LongTensor(temploader.dataset.targets).cuda()
            trainloader.dataset.transform = transform_bak
        else:
            trainFeatures = queue
            trainLabels = queue_labels
        C = class_number
        top1 = 0.
        top5 = 0.

        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, _) in enumerate(testloader):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            inputs = inputs.cuda()
            features= net(inputs)

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            if torch.min(retrieval)<0:
                retrieval[retrieval<0] = 3

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = torch.exp(torch.div(yd.clone(), sigma))
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)

    return top1/total, top5/total
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    # import copy
    # model = resnet50(pretrained=True)
    # model.reset_classifier(num_classes = 2)
    # transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # draw_cam(copy.deepcopy(model).cuda(), 'data/example.jpg', 'outputs/cam_1.png', visheadmap=True)
    
    # import time
    # for i in range(1000):
    #     if ChooseGPU_Memory(10000):
    #         break
    #     else:
    #         time.sleep(3)
    # print(ChooseGPU_Memory(10000))

    output = torch.rand(4,16)
    target = torch.tensor([1,0,2,3])
    accuracy(output, target, topk=(1,))
    '''
    if __name__ == "__main__":
        import pandas as pd
        data_root = 'data/DDR/DR_grading/'
        root = 'data/'
        """Basic Setting"""
        data_path_train = data_root + 'train/'
        data_path_val = data_root + 'valid/'
        data_path_test = data_root + 'test/'
        data_path_ddr = [data_path_train, data_path_val, data_path_test]
        Name_train = root + 'data_hu/data/DDR/012train.txt'
        Name_val = root + 'data_hu/data/DDR/012valid.txt'
        Name_test = root + 'data_hu/data/DDR/012test.txt'
        data_path = 'data/Qilu/'
        exl_path = 'data/qilu/labels.xlsx'
        DFqilu= pd.read_excel(exl_path, engine='openpyxl')
        DF0= pd.read_csv(Name_train, sep=' ') #6834
        DF1= pd.read_csv(Name_val, sep=' ') #2502
        DF2= pd.read_csv(Name_test, sep=' ') #3758
        df = DatasetConcat([DF0,DF1,DF2], DFqilu)
        df.to_csv('data/MyDRdataset/part_of_qilu_path_label.csv',
        index=False,header=['image_path','label']) 
    '''
