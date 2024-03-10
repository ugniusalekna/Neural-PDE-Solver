import numpy as np
import torch
import matplotlib.pyplot as plt
import os 
import shutil
import imageio
from tqdm import tqdm

import matplotlib.pyplot as plt
import os

def plot_figure(x_train, y_train, y_pred, model, epoch, loss, figure_name):
    plt.figure(figsize=(10, 6))
    plt.plot(x_train.detach().numpy(), y_train.detach().numpy(), label='Actual')
    plt.plot(x_train.detach().numpy(), y_pred.detach().numpy(), 'r', label='Predicted')
    plt.title(f'Epoch {epoch} Loss: {loss:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([x_train.min(), x_train.max()])
    plt.ylim([y_train.min()-1, 1.5*y_train.max()])
    
    last_layer_weights = model.fc_out.weight.data.tolist()[0]
    last_layer_bias = model.fc_out.bias.data.item()
    
    if len(last_layer_weights) <= 6:
        weights_text = 'Weights of Last Layer: ' + ', '.join([f'{w:.4f}' for w in last_layer_weights])
        bias_text = 'Bias of Last Layer: ' + f'{last_layer_bias:.4f}'
        info_text = f'{weights_text}\n{bias_text}'
        
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    plt.legend(loc='upper right')
    plt.grid(True)
    
    figure_directory = figure_name.rsplit('/', 1)[0]
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)
    
    save_path = f'{figure_name}/plot_epoch_{epoch}.png'
    
    plt.savefig(save_path)
    plt.close()

    return save_path




def save_gif(model_name, images):
    
    if not os.path.exists('gif'):
        os.makedirs('gif')
    
    gif_path = f'gif/{model_name}.gif'
    
    with imageio.get_writer(gif_path, mode='I') as writer:
        for filepath in images:
            image = imageio.imread(filepath)
            writer.append_data(image)
    print(f"GIF saved at {gif_path}")

    
def train(model, x_train, y_train, num_epochs, optimizer, loss_fn, plot_interval, model_name, atol=1e-5):
    images = []
    last_save_path = False
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=100)

    for epoch in epoch_pbar:
        optimizer.zero_grad()
        
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        
        loss.backward()
        optimizer.step()
    
        epoch_pbar.set_postfix_str(f"Train Loss: {loss.item():.8f}", refresh=True)

        if 0 < loss.item() < atol:
            print(f'Stopping criterion met at epoch {epoch}: Loss is less than {atol}.')
            last_save_path = plot_figure(x_train, y_train, y_pred, model, epoch, loss, figure_name=model_name)
            break
        
    # - - GIF SAVING - - 
    
        if epoch % plot_interval == 0 or epoch == num_epochs - 1:            
            save_path = plot_figure(x_train, y_train, y_pred, model, epoch, loss, figure_name=model_name)
            images.append(save_path)
    
    if last_save_path:
        images.append(last_save_path)
    save_gif(model_name, images)
    
    if os.path.exists(model_name):
        shutil.rmtree(model_name)



def select_available(cuda='all'):
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_info = ', '.join([f'{i}: {torch.cuda.get_device_name(i)}' for i in range(device_count)])
        print(f'Available CUDA GPUs ({device_count}): {device_info}')
        
        selected_device = f'cuda:{cuda}' if cuda != 'all' else 'cuda'
        device = torch.device(selected_device)
        device_name = torch.cuda.get_device_name(0 if cuda == 'all' else int(cuda))
        print(f'Using CUDA Device: {selected_device} | {device_name}')
        
    elif torch.backends.mps.is_available():
        print('Using MPS')
        device = torch.device('mps')
        
    else:
        print('Using CPU')
        device = torch.device('cpu')
    
    return device