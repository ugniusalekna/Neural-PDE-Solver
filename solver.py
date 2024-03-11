import os
import shutil
import torch
import inspect
from tqdm import tqdm

from plotting import plot_figure, write_gif

        
class ODESolver:
    def __init__(self, model, rhs_function, initial_conditions, device):
        self.model = model
        self.rhs_function = rhs_function
        self.initial_conditions = initial_conditions
        self.ode_order = len(inspect.signature(rhs_function).parameters) - 1
        self.device = device
    
        self.lambda_domain = 1.0
        self.lambda_ic = 1.0
        
    def compute_domain_loss(self, domain, outputs, norm):
        
        gradients = self._compute_gradients(domain, outputs)
        loss_fn = self._get_norm_func(norm)
        
        if self.ode_order > 1:
            loss_domain = loss_fn(gradients[-1], self.rhs_function(domain, outputs, *gradients[:-1]))
        else:
            loss_domain = loss_fn(gradients[-1], self.rhs_function(domain, outputs))

        return loss_domain

    def compute_ic_loss(self, initial_conditions, norm):
        x_0 = torch.tensor([initial_conditions[0]], requires_grad=True, device=self.device, dtype=torch.float32)

        output = self.model(x_0)
        predicted_ics = [output] + self._compute_gradients(x_0, output)

        ic_loss = 0
        loss_fn = self._get_norm_func(norm)

        for i in range(1, self.ode_order + 1):
            ic_loss += loss_fn(predicted_ics[i - 1], initial_conditions[i])

        return ic_loss

    def _compute_gradients(self, domain, model_output):
            
        gradients = [model_output]
        for _ in range(self.ode_order):
            grad = torch.autograd.grad(
                outputs=gradients[-1], inputs=domain,
                grad_outputs=torch.ones_like(gradients[-1]).to(self.device),
                create_graph=True,
            )[0]
            gradients.append(grad)
            
        return gradients[1:]
    

    def _get_norm_func(self, norm):
        match norm:
            case 'L1':
                loss_fn = lambda x, y: torch.mean(torch.abs(x - y))
            case 'L2':
                loss_fn = lambda x, y: torch.mean((x - y)**2)
            case 'Linf':
                loss_fn = lambda x, y: torch.max(torch.abs(x - y))
                
        return loss_fn
    
    def train(self, domain, num_epochs, optimizer, solution=None, atol=1e-5, save_gif=False):
        
        losses_domain, losses_ic = [], []
        epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=100)

        for epoch in epoch_pbar:
            optimizer.zero_grad()
            
            domain = domain.to(self.device)
            domain.requires_grad_(True)
            solution = solution.to(self.device) if solution is not None else None
            
            outputs = self.model(domain)

            loss_domain = self.compute_domain_loss(domain, outputs, norm='L2')
            loss_ic = self.compute_ic_loss(self.initial_conditions, norm='L2')
            loss = self.lambda_domain * loss_domain + self.lambda_ic * loss_ic
            
            loss.backward()
            optimizer.step()
        
            epoch_pbar.set_postfix_str(f"Train Loss: {loss.item():.8f}", refresh=True)
            losses_domain.append(loss_domain.item())
            
            losses_ic.append(loss_ic.item())

            if 0 < loss.item() < atol:
                print(f'Stopping criterion met at epoch {epoch}: Loss is less than {atol}.')
                if save_gif:
                    plot_figure(domain, outputs, solution, self.model, epoch, loss.item())
                break
                    
            if save_gif:
                plot_interval = 10
                if epoch % plot_interval == 0 or epoch == num_epochs - 1:
                    plot_figure(domain, outputs, solution, self.model, epoch, loss.item())

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