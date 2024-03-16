import os
import shutil
import inspect
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from abc import ABC, abstractmethod
from collections import defaultdict

from plotting import plot_figure, write_gif

        
class BaseSolver:
    def __init__(self, model, data, device=torch.device('cpu')):
        self.model = model.to(device)
        self.data = data
        self.domain = data.domain
        self.device = device
        
    def compile(self, optimizer, lr=1e-3, **kwargs):

        optimizer_kwargs = kwargs.get('opt', {})
        scheduler_kwargs = kwargs.get('sch', {})

        self.optimizer = self._configure_optimizer(optimizer, lr, **optimizer_kwargs)
        self.scheduler = self._configure_scheduler(**scheduler_kwargs)
        self._remove_temp()
            
    def _remove_temp(self):
        if os.path.exists('temp'):
            shutil.rmtree('temp')
            
    def _configure_optimizer(self, optimizer, lr, **kwargs):
        optimizers = {
            'adam': optim.Adam(self.model.parameters(), lr=lr, **kwargs),
            'adamW': optim.AdamW(self.model.parameters(), lr=lr, **kwargs),
            'adagrad': optim.Adagrad(self.model.parameters(), lr=lr, **kwargs),
            'SGD': optim.SGD(self.model.parameters(), lr=lr, **kwargs),
            'LBFGS': optim.LBFGS(self.model.parameters(), lr=lr, **kwargs)
        }
        return optimizers.get(optimizer, None)
     
    def _configure_scheduler(self, **kwargs):
        if 'step_size' in kwargs and 'gamma' in kwargs:
            return StepLR(self.optimizer, **kwargs)
        else:
            return StepLR(self.optimizer, step_size=10, gamma=1.0)

    def _get_norm_func(self, norm):
        loss_fn = {
            'L1': torch.nn.L1Loss(),
            'L2': torch.nn.MSELoss(),
            'Linf': lambda x, y: torch.max(torch.abs(x - y))
        }.get(norm, None)       
        return loss_fn
    
    def train(self, num_epochs, atol=1e-5, save_gif=False):
        
        loss_records = defaultdict(list)
        epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=120)

        domain = self.domain.to(self.device)
        domain.requires_grad_(True)
        solution = self.solution.to(self.device) if self.solution is not None else None
        
        for epoch in epoch_pbar:
            self.optimizer.zero_grad()
            outputs = self.model(domain)
            loss = self.compute_loss(domain, outputs, loss_records)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            epoch_pbar.set_postfix_str(f'Loss: {loss.item():.8f} | LR: {self.scheduler.get_last_lr()[0]:.8f}', refresh=True)

            if 0 < loss.item() < atol:
                print(f'Stopping criterion met at epoch {epoch}: Loss is less than {atol}.')
                if save_gif:
                    plot_figure(domain, outputs, solution, self.model, epoch, loss.item())
                break
                    
            if save_gif:
                plot_interval = 10
                if epoch % plot_interval == 0 or epoch == num_epochs - 1:
                    plot_figure(domain, outputs, solution, self.model, epoch, loss.item())

        return loss_records
            
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
        self._remove_temp()
   

class FunctionApproximator(BaseSolver):
    def __init__(self, model, data, device=torch.device('cpu')):
        super().__init__(model, data, device)
        self.solution = data.get_solution
    
    def compute_loss(self, domain, outputs, loss_records):
        loss_fn = self._get_norm_func(norm='L2')
        loss = loss_fn(outputs, self.solution.detach().to(self.device))
        loss_records['loss'].append(loss.item())
        return loss
    
    def evaluate(self, value):
        self.model.eval()
        parameter_count = self.model.param_count
        value = torch.tensor([value], requires_grad=True).to(self.device)
        output = self.model(value)
        
        results = f"Model parameter count: {parameter_count}\n--- At t = {value.item():.4f} ---\n"
                
        exact_value = self.data.solution_fn(value)
        value_error = abs(output.item() - exact_value.item())
        results += f"Solution | Computed: {output.item():.4f}, Exact: {exact_value.item():.4f}, Abs Error: {value_error:.4f}\n"

        print(results)
        
        
class ODESolver(BaseSolver):
    def __init__(self, model, data, device=torch.device('cpu')):
        super().__init__(model, data, device)
        self.ode, self.ics = data.ode, data.ics
        self.solution = data.get_solution
        self.ode_order = len(inspect.signature(self.ode).parameters) - 2
                
    def compile(self, optimizer, lr=1e-3, loss_weights=[1.0, 1.0], **kwargs):
        super().compile(optimizer, lr, **kwargs)
        self.lambda_domain, self.lambda_ic = loss_weights     
    
    def compute_loss(self, domain, outputs, loss_records):
        
        loss_domain = self._compute_domain_loss(domain, outputs, norm='L2')
        loss_ic = self._compute_ic_loss(self.ics, norm='L2')
        loss = self.lambda_domain * loss_domain + self.lambda_ic * loss_ic
        
        loss_records['loss_ic'].append(loss_ic.item())
        loss_records['loss_domain'].append(loss_domain.item())
        
        return loss
    
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
        
    def _compute_domain_loss(self, domain, outputs, norm):
        
        gradients = self._compute_gradients(domain, outputs)
        loss_fn = self._get_norm_func(norm)
        ode = self.ode(domain, outputs, *gradients)
        zero = torch.zeros_like(ode, dtype=torch.float32, device=self.device, requires_grad=True)
        loss_domain = loss_fn(ode, zero)

        return loss_domain

    def _compute_ic_loss(self, ics, norm):
        x_0 = torch.tensor([ics[0]], requires_grad=True, device=self.device, dtype=torch.float32)

        output = self.model(x_0)
        predicted_ics = [output] + self._compute_gradients(x_0, output)
        true_ics = [torch.tensor([value], device=self.device, dtype=torch.float32) if not torch.is_tensor(value) else value.view(1, -1) for value in ics]

        ic_loss = 0
        loss_fn = self._get_norm_func(norm)

        for i in range(1, self.ode_order + 1):
            pred_ic = predicted_ics[i - 1].view(1, -1)
            true_ic = true_ics[i].view(1, -1)
            ic_loss += loss_fn(pred_ic, true_ic)
            
        return ic_loss

    def evaluate(self, value, exact_derivatives=None):
        self.model.eval()
        parameter_count = self.model.param_count
        value = torch.tensor([value], requires_grad=True).to(self.device)
        output = self.model(value)
        computed_derivatives = self._compute_gradients(value, output)
        
        results = f"Model parameter count: {parameter_count}\n--- At t = {value.item():.4f} ---\n"
        
        if self.data.solution_fn is not None:
            exact_value = self.data.solution_fn(value).item()
            value_error = abs(output.item() - exact_value)
            results += f"Solution | Computed: {output.item():.4f}, Exact: {exact_value:.4f}, Abs Error: {value_error:.4f}\n"
        else:
            results += f"Solution | Computed: {output.item():.4f}\n"

        if exact_derivatives is not None:
            for i, (computed_derivative, exact_derivative_fn) in enumerate(zip(computed_derivatives, exact_derivatives)):
                exact_derivative = exact_derivative_fn(value)
                derivative_error = abs(computed_derivative.item() - exact_derivative.item())
                results += f"Derivative order {i+1} | Computed: {computed_derivative.item():.4f}, Exact: {exact_derivative.item():.4f}, Abs Error: {derivative_error:.4f}\n"
        else:
            for i, computed_derivative in enumerate(computed_derivatives):
                results += f"Derivative order {i+1} | Computed: {computed_derivative.item():.4f}\n"

        print(results)