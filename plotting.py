import os
import numpy as np
import matplotlib.pyplot as plt
import imageio


def write_gif(save_path, images):
    
    if not os.path.exists(save_path.split("/")[0]):
        os.makedirs(save_path.split("/")[0] )
    
    gif_path = f'{save_path}.gif'
    
    with imageio.get_writer(gif_path, mode='I') as writer:
        for filepath in images:
            image = imageio.imread(filepath)
            writer.append_data(image)
    
    print(f"GIF saved at {gif_path}")


def plot_figure(domain, outputs, solution=None, model=None, epoch=None, loss=None):
    
    domain_np = domain.detach().cpu().numpy()
    if solution is not None:
        solution_np = solution.detach().cpu().numpy()
    outputs_np = outputs.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    if solution is not None:
        plt.plot(domain_np, solution_np, label='Actual')
    plt.plot(domain_np, outputs_np, 'r', label='Predicted')
    plt.title(f'Epoch {epoch} Loss: {loss:.8f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([domain_np.min(), domain_np.max()])
    if solution is not None:
        plt.ylim([solution_np.min()-1, 1.5*solution_np.max()])
    else:
        plt.ylim([outputs_np.min()-1, 1.5*outputs_np.max()])
    
    last_layer_weights = model.blocks[-1].layers[-1].weight.data.tolist()[0]
    last_layer_bias = model.blocks[-1].layers[-1].bias.data.item()
    
    if len(last_layer_weights) <= 6:
        weights_text = 'Weights of Last Layer: ' + ', '.join([f'{w:.4f}' for w in last_layer_weights])
        bias_text = 'Bias of Last Layer: ' + f'{last_layer_bias:.4f}'
        info_text = f'{weights_text}\n{bias_text}'
        
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    plt.legend(loc='upper right')
    plt.grid(True)
    
    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    save_path = f'temp/plot_epoch_{epoch}.png'
    
    plt.savefig(save_path)
    plt.close()
    
    
def plot_losses(losses_domain, losses_ic):
    plt.figure(figsize=(10, 6))
    plt.plot(losses_domain, label='Domain Loss', color='blue')
    plt.plot(losses_ic, label='Initial Condition Loss', color='red')
    plt.title('Losses During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()