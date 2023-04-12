import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import logging
import os

logging.basicConfig(level=logging.DEBUG)


def save_model(epochs, model, optimizer, criterion, name='model'):
    """
    Function to save the trained model to disk.
    """
    filename = f'outputs/{name}.pth'
    logging.info(f'{filename=}')
    
    if not os.path.exists('outputs'):
        logging.info('"outputs" folder does not exist. Creating.')
        os.makedirs('outputs')
    
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, filename)
    

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    
    if not os.path.exists('outputs'):
        logging.info('"outputs" folder does not exist. Creating.')
        os.makedirs('outputs')
    
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')
    
    
