import numpy  as np
import matplotlib.pylab as plt

def binary_accuracy(y_pred,y_true, class_labels=[1,-1]):
    """Accuracy function for binary classification models."""
    threshold = min(class_labels) + (max(class_labels) - min(class_labels)) / 2.
    pred_labels = np.where(y_pred >= threshold, max(class_labels), min(class_labels))
    return np.mean(pred_labels == y_true)*100

def accuracy(y_pred, y_true, one_hot_encoded_labels=True):    
    if one_hot_encoded_labels:
        y_pred = np.argmax(y_pred,axis=-1)
        y_true = np.argmax(y_true,axis=-1)
    return np.mean(y_pred == y_true, axis=0) * 100

def mean_squared_error(y_pred,y_true):
    return 0.5*np.mean((y_pred - y_true)**2)

def mean_absolute_error(y_pred,y_true):
    return np.mean(np.abs(y_pred - y_true))

def cross_entropy(y_pred,y_true):
    return np.mean(np.sum(-y_true*np.log(y_pred), axis=-1))

def plot_confusion_matrix(cm, cmap="Blues", figsize=(6, 5), class_names=None, title="Confusion Matrix", ax=None, show_plt=True, rotation=45, ha='right'):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n_classes = cm.shape[0]

    if class_names is None:
        class_names = np.arange(n_classes)

    # Tick labels
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
            xticklabels=class_names,
            yticklabels=class_names,
           ylabel="True label",
           xlabel="Predicted label",
           title=title)

    # Rotate x-tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=rotation,ha=ha, rotation_mode="anchor")

    # Annotate each cell
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i,
                format(cm[i, j], 'd'),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )
    if show_plt:
        plt.tight_layout()
        plt.show()

def confusion_matrix(y_true, y_pred, num_classes=None, plot=True, **kwargs):

    if num_classes is None:
        num_classes = y_true.shape[-1]

    true_labels = np.argmax(y_true, axis=-1)
    pred_labels = np.argmax(y_pred, axis=-1)

    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1

    if plot:
        plot_confusion_matrix(cm, **kwargs)
    else:
        return cm

def binary_confusion_matrix(y_true, y_pred,plot=True,labels=[0,1], **kwargs):
    """Calculate confusion matrix for binary classification"""
    y_true_=np.where(y_true==labels[0], 0,1 )
    y_pred_=np.where(y_pred==labels[0], 0,1 )

    cm = np.zeros((2, 2), dtype=int)

    for t, p in zip(y_true_, y_pred_):
        cm[t, p] += 1

    if plot:
        plot_confusion_matrix(cm, **kwargs)
    else:
        return cm

def precision(y_pred, y_true, true_label=1):
    """
    calculate precision of binary classification
    
    Parameters:
        - y_pred: array;  predicted labels
        - y_true: array;  actual labels
        - true_label: the label considered as true (default=1)
    """
    tp=np.count_nonzero((y_pred==y_true) & (y_true==true_label)) # true positives
    return 0 if tp==0 else tp/np.count_nonzero(y_pred==true_label) # avoid dividing by zero

def recall(y_pred, y_true, true_label=1):
    """
    calculate recall for binary classification
    
    Parameters:
        - y_pred: array;  predicted labels
        - y_true: array;  actual labels
        - true_label: the label considered as true (default=1)
    """
    tp=np.count_nonzero((y_pred==y_true) & (y_true==true_label)) # true positives
    return 0 if tp==0 else tp/np.count_nonzero(y_true==true_label) # avoid dividing by zero

def f1_score(y_pred,y_true, true_label=1):
    """
    calculate f1-score for binary classification
    
    Parameters:
        - y_pred: array;  predicted labels
        - y_true: array;  actual labels
        - true_label: the label considered as true (default=1)
    """
    prec=precision(y_pred,y_true, true_label) # precision
    rec=recall(y_pred, y_true, true_label) # recall
    return 0 if (prec+rec)== 0 else 2* prec*rec/(prec+rec) # avoid dividing by zero
    










