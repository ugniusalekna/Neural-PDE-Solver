import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio


class BasePlotter:
    def __init__(self, solver):
        self.solver = solver
        self.model = solver.model
        self.domain = self._get_domain()
        self.output = self._get_output()
        self.solution = self._get_solution()

    def _get_domain(self, torch=False):
        return self.solver.domain.detach().cpu().numpy() if not torch else self.solver.domain.detach().cpu()

    def _get_output(self):
        domain = self.solver.domain
        output = self.model(domain)
        return output.detach().cpu().numpy()

    def _get_solution(self):
        return self.solver.solution.detach().cpu().numpy() if self.solver.solution is not None else None
        
    def losses(self, loss_records):
        plt.figure(figsize=(10, 6))
        for name, loss in loss_records.items():
            plt.plot(loss, label=name)
        plt.title('Losses over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.show()

    def numerical_solution(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.domain, self.output, label='Approximated', color='red')
        if self.solution is not None:
            plt.plot(self.domain, self.solution, label='Exact', color='blue')
        plt.xlabel('$t$')
        plt.ylabel('$y(t)$')
        plt.grid(True)
        plt.legend()
        plt.title('Solution plot')
        plt.show()


class ODEPlotter(BasePlotter):
    def __init__(self, solver):
        super().__init__(solver)
        self.derivatives = self._get_derivatives()
        self.solution = self._get_solution()
        
    def _get_derivatives(self):
        domain = self.solver.domain
        output = self.model(domain)
        derivatives = self.solver._compute_gradients(domain, output)
        return [d.detach().cpu().numpy() for d in derivatives]

    def phase_space(self, exact_derivatives=None):
        ode_order = self.solver.ode_order
        if ode_order == 2:
            self._plot_2d_phase_space(exact_derivatives)
        elif ode_order == 3:
            self._plot_3d_phase_space(exact_derivatives)
        else:
            raise ValueError('Can only plot phase space for ODEs of order 2 and 3')
        
    def extended_phase_space(self, exact_derivatives=None):
        ode_order = self.solver.ode_order
        if ode_order == 2:
            self._plot_3d_extended_phase_space(exact_derivatives)
        else:
            raise ValueError('Can only plot phase space for ODEs of order 2')
        
    def _plot_2d_phase_space(self, exact_derivatives):
        plt.figure(figsize=(10, 6))
        plt.plot(self.output, self.derivatives[0], label='Approximated', color='red')
        if exact_derivatives is not None and self.solution is not None:
            exact_dy = exact_derivatives[0](self._get_domain(torch=True)).detach().cpu().numpy()
            plt.plot(self.solution, exact_dy, label='Exact', color='blue')
        plt.xlabel('$y(t)$')
        plt.ylabel('$y\'(t)$')
        plt.grid(True)
        plt.legend()
        plt.title('Phase space')
        plt.show()
        
    def _plot_3d_phase_space(self, exact_derivatives):
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.output, self.derivatives[0], self.derivatives[1], label='Approximated', color='red')
        if exact_derivatives is not None and self.solution is not None:
            exact_dy = exact_derivatives[0](self._get_domain(torch=True)).detach().cpu().numpy()
            exact_dy2 = exact_derivatives[1](self._get_domain(torch=True)).detach().cpu().numpy()
            ax.plot(self.solution, exact_dy, exact_dy2, label='Exact', color='blue')
        ax.set_xlabel('$y(t)$')
        ax.set_ylabel('$y\'(t)$')
        ax.set_zlabel('$y\'\'(t)$')
        plt.grid(True)
        plt.legend()
        plt.title('Phase space')
        plt.show()
        
    def _plot_3d_extended_phase_space(self, exact_derivatives):
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.domain, self.output, self.derivatives[0], label='Approximated', color='red')
        if exact_derivatives is not None and self.solution is not None:
            exact_dy = exact_derivatives[0](self._get_domain(torch=True)).detach().cpu().numpy()
            ax.plot(self.domain, self.solution, exact_dy, label='Exact', color='blue')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$y(t)$')
        ax.set_zlabel('$y\'(t)$')
        plt.grid(True)
        plt.legend()
        plt.title('Extended phase space')
        plt.show()
    
    
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
    
    last_layer_weights = model.blocks[-1].layers[-1].weight.data.tolist()[0] if model is not None else None
    last_layer_bias = model.blocks[-1].layers[-1].bias.data.item() if model is not None else None
    
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