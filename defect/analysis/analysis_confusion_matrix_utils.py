import mxnet as mx
from mxnet import gluon
from gluoncv.model_zoo import get_model
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

def get_loaded_model(models_fname, net_name, classes, ctx):
    net = get_model(net_name, ctx=ctx, classes=classes, pretrained=False)
#    net.load_parameters(models_fname, ctx=ctx, allow_missing=True, ignore_extra=True)
    net.load_parameters(models_fname, ctx=ctx)
    return net

def draw_confusion_matrix(cm, classes):
    plt.figure(figsize=(25,25))
    sns.heatmap(cm,cmap="Blues",annot=True,xticklabels=classes,yticklabels=classes) 
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def create_confusion_matrix(net, ctx, val_data, labels):
    all_labels = []
    all_outputs = []

    for i, batch in enumerate(val_data):
        if (ctx != mx.cpu) and (mx.context.num_gpus() > 0):
            data = gluon.utils.split_and_load(
                batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(
                batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        else:
            data = batch[0].as_in_context(ctx)
            label = batch[1].as_in_context(ctx)
        outputs = [net(X) for X in data]
        
        for l in label:
            all_labels.extend(l.asnumpy().tolist())

        for l in outputs:
            for o in l:
                all_outputs.append(np.argmax(o.asnumpy()))

    cm = confusion_matrix(all_labels, all_outputs, normalize='true')
    
    draw_confusion_matrix(cm=cm,classes=labels)
    print('Classification Report: \n',classification_report(all_labels,all_outputs, digits=3))
    
    print('y_test: ' + str(sorted(Counter(all_labels).items())))
    print('y_pred: ' + str(sorted(Counter(all_outputs).items())))