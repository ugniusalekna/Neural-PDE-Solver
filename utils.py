import torch


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