import os
import shutil
import torch
from tqdm import tqdm

from plotting import plot_figure, write_gif


def train_func_approx(model, domain, solution, num_epochs, optimizer, loss_fn, device, atol=1e-5, plot_interval=10, gif_save_path=None):
    images = []
    last_save_path = False
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=100)

    if gif_save_path is not None:
        gif_save_path = f"{gif_save_path}/{model.activation}_{'_'.join(map(str, model.hidden_layers))}"

    for epoch in epoch_pbar:
        optimizer.zero_grad()
        
        model = model.to(device)
        domain, solution = domain.to(device), solution.to(device)
        
        outputs = model(domain)
        loss = loss_fn(outputs, solution)
        
        loss.backward()
        optimizer.step()
    
        epoch_pbar.set_postfix_str(f"Train Loss: {loss.item():.8f}", refresh=True)

        if 0 < loss.item() < atol:
            print(f'Stopping criterion met at epoch {epoch}: Loss is less than {atol}.')
            if gif_save_path is not None:
                last_save_path = plot_figure(domain, solution, outputs, model, epoch, loss)
            break
        
    # - - GIF SAVING - - 
    
        if gif_save_path is not None and (epoch % plot_interval == 0 or epoch == num_epochs - 1):            
            save_path = plot_figure(domain, solution, outputs, model, epoch, loss)
            images.append(save_path)
    
    if gif_save_path is not None:
        if last_save_path:
            images.append(last_save_path)
        write_gif(gif_save_path, images)
    
    if os.path.exists('temp'):
        shutil.rmtree('temp')


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