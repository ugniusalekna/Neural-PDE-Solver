import os
import shutil
import inspect
import torch
import torch.optim as optim
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
        
    def compile(self, optimizer, scheduler=None, lr=1e-3, **kwargs):
        optimizer_kwargs = kwargs.get('opt', {})
        scheduler_kwargs = kwargs.get('sch', {})
        self.optimizer = self._configure_optimizer(optimizer, lr, **optimizer_kwargs)
        self.scheduler = self._configure_scheduler(scheduler, **scheduler_kwargs)
        self._remove_temp()
            
    def _remove_temp(self):
        if os.path.exists('temp'):
            shutil.rmtree('temp')
            
    def _configure_optimizer(self, optimizer, lr=1e-3, **kwargs):
        return {
            'adam': optim.Adam(self.model.parameters(), lr=lr, **kwargs),
            'adamW': optim.AdamW(self.model.parameters(), lr=lr, **kwargs),
            'adagrad': optim.Adagrad(self.model.parameters(), lr=lr, **kwargs),
            'SGD': optim.SGD(self.model.parameters(), lr=lr, **kwargs),
            'LBFGS': optim.LBFGS(self.model.parameters(), lr=lr, **kwargs),
        }.get(optimizer, None)
     
    def _configure_scheduler(self, scheduler, **kwargs):
        match scheduler:
            case 'step':
                return optim.lr_scheduler.StepLR(self.optimizer, **kwargs)
            case 'onecycle':
                return optim.lr_scheduler.OneCycleLR(self.optimizer, **kwargs)
            case _:
                return optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)

    def _get_norm_func(self, norm):
        loss_fn = {
            'L1': torch.nn.L1Loss(reduction='sum'),
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
                    plot_figure(domain, outputs, solution, epoch, loss.item())
                break
                    
            if save_gif:
                plot_interval = 10
                if epoch % plot_interval == 0 or epoch == num_epochs - 1:
                    plot_figure(domain, outputs, solution, epoch, loss.item())

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
        self.solution = data.solution.evaluate(self.domain)
    
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
                
        exact_value = self.data.solution.evaluate(value)
        value_error = abs(output.item() - exact_value.item())
        results += f"Solution | Computed: {output.item():.4f}, Exact: {exact_value.item():.4f}, Abs Error: {value_error:.4f}\n"

        print(results)


class ODESolver(BaseSolver):
    def __init__(self, model, data, device=torch.device('cpu')):
        super().__init__(model, data, device)
        self.odes = data.equation
        self.ics = data.ics
        self.solution = data.solution.evaluate(self.domain)
        self.ode_order = len(inspect.signature(self.odes.f).parameters) - 2
        self.n_equations = model.output_features

    def compile(self, optimizer, scheduler, lr=1e-3, loss_weights=[1.0, 1.0], **kwargs):
        super().compile(optimizer, scheduler, lr, **kwargs)
        self.lambda_domain, self.lambda_ic = loss_weights     

    def compute_loss(self, domain, outputs, loss_records):
        loss_domain = self._compute_domain_loss(domain, outputs, norm='L2')
        loss_ic = self._compute_ic_loss(self.ics, norm='L2')
        loss = self.lambda_domain * loss_domain + self.lambda_ic * loss_ic
        loss_records['loss_ic'].append(loss_ic.item())
        loss_records['loss_domain'].append(loss_domain.item())
        return loss

    def _compute_domain_loss(self, domain, outputs, norm):
        gradients = self.compute_derivatives(domain, outputs)
        loss_fn = self._get_norm_func(norm)
        odes = self.odes.evaluate(domain, outputs, *gradients)
        zeros = torch.zeros_like(odes, requires_grad=True, dtype=torch.float32, device=self.device)
        loss_domain = loss_fn(odes, zeros)
        return loss_domain

    def _compute_ic_loss(self, ics, norm):
        x_0 = torch.tensor([[ics[0][0]]], requires_grad=True, dtype=torch.float32, device=self.device)
        y_0 = self.model(x_0)
        predicted_ics = self._get_predicted_ics(x_0, y_0)
        true_ics = self._get_true_ics(ics)
        loss_fn = self._get_norm_func(norm)
        ic_loss = loss_fn(predicted_ics, true_ics)
        return ic_loss

    def derivative(self, y, x):
        dydx = torch.autograd.grad(
                    outputs=y, inputs=x,
                    grad_outputs=torch.ones_like(y).to(self.device),
                    create_graph=True,
                )[0]
        return dydx

    def compute_derivatives(self, domain, model_output):
        derivatives = [model_output]
        for _ in range(self.ode_order):
            grad = []
            for j in range(self.n_equations):
                grad.append(self.derivative(derivatives[-1][:, j], domain))
            derivatives.append(torch.hstack(grad))
        return derivatives[1:]
    
    def _get_true_ics(self, ics):
        ic_matrix = torch.zeros((self.n_equations, self.ode_order), dtype=torch.float32, device=self.device)
        for eq_idx, values in enumerate(ics):
            ic_matrix[eq_idx, :] = torch.tensor(values[1:], dtype=torch.float32, device=self.device)
        return ic_matrix
            
    def _get_predicted_ics(self, x_0, y_0):
        derivatives = self.compute_derivatives(x_0, y_0)[:-1]
        ic_matrix = torch.zeros((self.n_equations, self.ode_order), dtype=torch.float32, device=self.device)
        ic_matrix[:, 0] = y_0
        for eq_idx, derivative in enumerate(derivatives):
            ic_matrix[:, eq_idx + 1] = derivative
        return ic_matrix

    def evaluate(self, value, exact_derivatives=None):
        
        self.model.eval()
        value = torch.tensor([[value]], requires_grad=True, dtype=torch.float32).to(self.device)
        outputs = self.model(value)
        outputs = outputs.unsqueeze(0) if outputs.dim() == 1 else outputs
        computed_derivatives = self.compute_derivatives(value, outputs)
        
        results = f"Model parameter count: {self.model.param_count}\n--- At t = {value.item():.4f} ---\n"
        for eq_index in range(outputs.shape[1]):
            output = outputs[:, eq_index]

            if self.solution is not None:
                exact_value = self.data.solution.evaluate(value)[:, eq_index]
                value_error = abs(output - exact_value)
                results += f"Equation {eq_index + 1} | Solution: Computed: {output.item():.4f}, Exact: {exact_value.item():.4f}, Abs Error: {value_error.item():.4f}\n"
            else:
                results += f"Equation {eq_index + 1} | Solution: Computed: {output.item():.4f}\n"
            
            # if exact_derivatives is not None:
            #     exact_d_value = exact_derivatives.evaluate(value)[:, eq_index]
            #     for order, (comp_ders, exact_ders) in enumerate(zip(computed_derivatives, exact_derivatives)):
            #         exact_der = exact_ders(value)[:, eq_index]
            #         comp_der = comp_ders[:, eq_index]
            #         derivative_error = abs(comp_der - exact_der)
            #         results += f"Equation {eq_index + 1}, Derivative order {order + 1} | Computed: {comp_der.item():.4f}, Exact: {exact_der.item():.4f}, Abs Error: {derivative_error.item():.4f}\n"
        
        print(results)