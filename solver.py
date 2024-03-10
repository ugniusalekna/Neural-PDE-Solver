import os
import shutil
import torch
import inspect
from tqdm import tqdm

from plotting import plot_figure, write_gif

        
class ODESolver:
    def __init__(self, model, rhs_function, initial_conditions, norm, device):
        self.model = model
        self.rhs_function = rhs_function
        self.initial_conditions = initial_conditions
        self.ode_order = len(inspect.signature(rhs_function).parameters) - 1
        self.norm = norm
        self.device = device
        
    def compute_gradients(self, domain, model_output):
            
        gradients = [model_output]
        for _ in range(self.ode_order):

            grad = torch.autograd.grad(
                outputs=gradients[-1], inputs=domain,
                grad_outputs=torch.ones_like(gradients[-1]).to(self.device),
                create_graph=True,
            )[0]
            gradients.append(grad)
        return gradients[1:]
    
    def compute_domain_loss(self, domain, outputs, norm='L2'):
        
        gradients = self.compute_gradients(domain, outputs)
        match norm:
            case 'L1':
                loss_fn = lambda x, y: torch.mean(torch.abs(x - y))
            case 'L2':
                loss_fn = lambda x, y: torch.mean((x - y)**2)
                
        if self.ode_order > 1:
            loss_domain = loss_fn(gradients[-1], self.rhs_function(domain, outputs, *gradients[:-1]))
        else:
            loss_domain = loss_fn(gradients[-1], self.rhs_function(domain, outputs))

        return loss_domain
    
    def compute_ic_loss(self, initial_conditions, norm='L2'):
        
        x_0 = torch.tensor([initial_conditions[0]], requires_grad=True, device=self.device, dtype=torch.float32)
        actual_values = torch.tensor(initial_conditions[1:], device=self.device, dtype=torch.float32)

        y_pred = self.model(x_0)

        predictions = self.compute_gradients(x_0, y_pred)
        predictions = torch.stack(predictions).squeeze()
        
        match norm:
            case 'L1':
                loss_fn = lambda x, y: torch.mean(torch.abs(x - y))
            case 'L2':
                loss_fn = lambda x, y: torch.mean((x - y)**2)
                
        loss_ic = loss_fn(predictions, actual_values)

        return loss_ic
    
    def train(self, domain, num_epochs, optimizer, solution=None, atol=1e-5, save_gif=False):
        
        losses_domain, losses_ic = [], []
        epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=100)

        for epoch in epoch_pbar:
            optimizer.zero_grad()
            
            model = self.model.to(self.device)
            domain = domain.to(self.device)
            domain.requires_grad_(True)
            solution = solution.to(self.device) if solution is not None else None
            
            outputs = model(domain)

            loss_domain = self.compute_domain_loss(domain, outputs, self.norm)
            loss_ic = self.compute_ic_loss(self.initial_conditions, self.norm)
            loss = loss_domain + loss_ic
            
            loss.backward()
            optimizer.step()
        
            epoch_pbar.set_postfix_str(f"Train Loss: {loss.item():.8f}", refresh=True)
            losses_domain.append(loss_domain.item())
            
            losses_ic.append(loss_ic.item())

            if 0 < loss.item() < atol:
                print(f'Stopping criterion met at epoch {epoch}: Loss is less than {atol}.')
                if save_gif:
                    plot_figure(domain, outputs, solution, model, epoch, loss.item())
                break
                    
            if save_gif:
                plot_interval = 10
                if epoch % plot_interval == 0 or epoch == num_epochs - 1:
                    plot_figure(domain, outputs, solution, model, epoch, loss.item())

        return losses_domain, losses_ic
            
    def create_gif(self, gif_save_path):
        
        gif_save_path = f"{gif_save_path}/{self.model.activation}_{'_'.join(map(str, self.model.hidden))}"
        
        def sort_key(filename):
            return int(filename.split('_')[2].split('.')[0])

        images = sorted(
            [os.path.join('temp', filename)
             for filename in os.listdir('temp')
             if filename.endswith('.png')],
            key=sort_key)
        
        write_gif(gif_save_path, images)

        if os.path.exists('temp'):
            shutil.rmtree('temp')