"""
This will contain more than just metrics plots
"""


from matplotlib import pyplot as plt
from os.path import join
from config import hyparams, loc
from sklearn.metrics import roc_curve, auc
import wandb 
import numpy as np

def loss_plot(history, plot_loc, nc, save_wandb):
    """
    history:  trained model with details for plots
    plot_loc: directory to save images for metrics 
    nc: naming convention
    """
    fig,ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(history.history['loss'], '-', 
                    color='black', label='train_loss')

    ax.plot(history.history['val_loss'], '-', 
                    color='red', label='val_loss')
    
    ax.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    img_path = join(    plot_loc, 
                        '{}_loss_{}_{}_{}_{}.jpg'.format(
                        *nc
                        ))
    fig.savefig(img_path)
    # might have a problem if I try saving lstm model loss
    if save_wandb:
        if hyparams['networks']['binary_classifier']['wandb']:    
            wandb.log({"losses": wandb.Image(img_path)})
    

    print('Saving Done for Loss')

def accuracy_plot(history, plot_loc, nc):
    """
    history:  trained model with details for plots
    plot_loc: directory to save images for metrics 
    nc: naming convention
    """
    fig,ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(history.history['accuracy'], '-', 
                    color='black', label='train_acc')

    ax.plot(history.history['val_accuracy'], '-', 
                    color='red', label='val_acc')
    
    ax.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    img_path = join( plot_loc, 
                '{}_acc_{}_{}_{}_{}.jpg'.format(
                *nc
                ))
    fig.savefig(img_path)
    if hyparams['networks']['binary_classifier']['wandb']:    
        wandb.log({"acc": wandb.Image(img_path)})

    print('Saving Done for Acc')


# def roc_plot(model,data, plot_loc, nc, wandb_name):
def roc_plot(y_true, y_pred, plot_loc, nc, wandb_name):
    """
    y_true: true y_values
    y_pred: predicted y_values
    plot_loc: directory to save images for metrics 
    nc: naming convention
    wandb_name: string that controls name of files saved
                in wandb

    """
    # to do need to fix for classifer calling was
    # passing in model and data dict with x and y values 
    # y_pred = model.predict(data['x'])
    fig,ax = plt.subplots(nrows=1, ncols=1)
    # fpr, tpr, thresholds = roc_curve(   data['y'],
    #                                     y_pred
    #                                     )

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    AUC = auc(fpr, tpr)
    print('AUC is {}'.format(AUC))
    
    ax.plot(fpr, tpr, linewidth=2, label ='AUC = {:.4f}'.format(AUC) )
    ax.plot([0, 1], [0, 1], 'k--')
    ax.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    img_path = join( plot_loc, 
                '{}_roc_{}_{}_{}_{}.jpg'.format(
                *nc
                ))
    fig.savefig(img_path)
    if hyparams['networks']['binary_classifier']['wandb']:    
        wandb.log({"rocs": wandb.Image(img_path)})
        wandb.log({wandb_name[0]: wandb.Image(img_path)})
    
   
        # # For Wandb
        # y_pred_0 = np.array([1-i for i in y_pred]).reshape(-1,1)
        # y_pred = np.append(y_pred_0, y_pred,axis=1)
        # # wromg one wandb.log({"roc_curve" : wandb.plot.roc_curve( data['y'], y_pred, labels=["normal", "abnormal"] ) } )
        # wandb.log({wandb_name[1] : wandb.plot.roc_curve( data['y'], y_pred, labels=["normal", "abnormal"] ) } )





def plot_iou(prob_iou,xlabel, ped_type, plot_loc, nc):
    """
    envisioned this to show how the abnormal pedestrains iou look
    prob_iou: this is the prob iou
    xlabel: xlabel for plot
    ped_type: 'normal_ped' , abnormal_ped
    """
    fig,ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.arange(0,len(prob_iou)), prob_iou, '.' )
    ax.plot(np.arange(0,len(prob_iou)), 0.5*np.ones([1, len(prob_iou)]), '-b', label='midpoint' )
    ax.plot(np.arange(0,len(prob_iou)), np.mean(prob_iou)*np.ones([1, len(prob_iou)]), '-r', label='mean + {:.4f}'.format(np.mean(prob_iou)) )
    ax.legend()
    plt.xlabel(xlabel)
    plt.ylabel('Probability (1-IOU)')


    img_path = join( plot_loc, 
            '{}_{}_{}_{}_{}_{}.jpg'.format(
            *nc, ped_type
            ))
    fig.savefig(img_path)