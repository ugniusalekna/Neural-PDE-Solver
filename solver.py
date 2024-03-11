import os
import shutil
import inspect
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from plotting import plot_figure, write_gif

        
class ODESolver:
    def __init__(self, model, data, device):
        
        self.domain, self.rhs_function, self.initial_conditions = data
        self.model = model
        self.ode_order = len(inspect.signature(self.rhs_function).parameters) - 1
        self.device = device
        
    def _compute_domain_loss(self, domain, outputs, norm):
        
        gradients = self._compute_gradients(domain, outputs)
        loss_fn = self._get_norm_func(norm)
        
        if self.ode_order > 1:
            loss_domain = loss_fn(gradients[-1], self.rhs_function(domain, outputs, *gradients[:-1]))
        else:
            loss_domain = loss_fn(gradients[-1], self.rhs_function(domain, outputs))

        return loss_domain

    def _compute_ic_loss(self, initial_conditions, norm):
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
    
    def compile(self, optimizer, lr=1e-3, momentum=0.9, loss_weights=[1.0, 1.0]):

        self.optimizer = {
            'adam': torch.optim.Adam(self.model.parameters(), lr=lr),
            'adagrad': torch.optim.Adagrad(self.model.parameters(), lr=lr),
            'SGD': torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum),
            'LBFGS': torch.optim.LBFGS(self.model.parameters(), lr=lr)
        }.get(optimizer, None)
        
        self.lambda_domain, self.lambda_ic = loss_weights     
    
    def train(self, num_epochs, atol=1e-5, solution=None, save_gif=False):
        
        self.scheduler = StepLR(self.optimizer, step_size=num_epochs//100, gamma=0.95)

        losses_domain, losses_ic = [], []
        epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=120)

        for epoch in epoch_pbar:
            self.optimizer.zero_grad()
            
            domain = self.domain.to(self.device)
            domain.requires_grad_(True)
            solution = solution.to(self.device) if solution is not None else None
            
            outputs = self.model(domain)

            loss_domain = self._compute_domain_loss(domain, outputs, norm='L2')
            loss_ic = self._compute_ic_loss(self.initial_conditions, norm='L2')
            loss = self.lambda_domain * loss_domain + self.lambda_ic * loss_ic
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            epoch_pbar.set_postfix_str(f'Loss: {loss.item():.8f} | LR: {self.scheduler.get_last_lr()[0]:.8f}', refresh=True)
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
        
        gif_save_path = f"{gif_save_path}/{self.model.activation}_{'_'.join(map(str, self.model.hidden_layers))}"
        
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