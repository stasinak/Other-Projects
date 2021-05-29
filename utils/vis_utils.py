from matplotlib import pyplot as plt
from os import path

def plot_history(history, log_dir):

    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.savefig(path.join(log_dir,"history_plot.png"))
