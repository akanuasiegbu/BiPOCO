from matplotlib import pyplot as plt
from os.path import join
from config import hyparams, loc

def loss_plot(history, plot_loc):
    """
    history:  trained model with details for plots
    plot_loc: directory to save images for metrics 
    """
    fig,ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(history.history['loss'], '-', 
                    color='black', label='loss')

    ax.plot(history.history['val_loss'], '-', 
                    color='red', label='Validation Loss')
    
    ax.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    fig.savefig(join(   plot_loc, 
                        'loss_{}_{}_{}_{}.jpg'.format(
                        loc['nc']['model_name'],
                        loc['nc']['data_coordinate_out'],
                        loc['nc']['dataset_name'],
                        hyparams['frames']
                        )))
    print('Saving Done for Loss')

def accuracy_plot(history, plot_loc):
    """
    history:  trained model with details for plots
    plot_loc: directory to save images for metrics 
    """
    fig,ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(history.history['accuracy'], '-', 
                    color='black', label='Acc')

    ax.plot(history.history['val_accuracy'], '-', 
                    color='red', label='Validation Acc')
    
    ax.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    fig.savefig(join(   plot_loc, 
                        'loss_{}_{}_{}_{}.jpg'.format(
                        loc['nc']['model_name'],
                        loc['nc']['data_coordinate_out'],
                        loc['nc']['dataset_name'],
                        hyparams['frames']
                        )))
    print('Saving Done for Acc')


def roc_plot(history, plot_loc):
    """
    history:  trained model with details for plots
    plot_loc: directory to save images for metrics 
    """
    pass

