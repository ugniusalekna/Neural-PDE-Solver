import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio


class BasePlotter:
    def __init__(self, solver):
        self.solver = solver
        self.model = solver.model
        self.n_equations = self.model.output_features
        self.domain = self._get_domain()
        self.output = self._get_output()
        self.solution = self._get_solution()
        self.system = True if self.n_equations > 1 else False

    def _get_domain(self, torch=False):
        return self.solver.domain.detach().cpu().numpy() if not torch else self.solver.domain.detach().cpu()

    def _get_output(self):
        domain = self.solver.domain
        output = self.model(domain)
        return [output[:, i].detach().cpu().numpy() for i in range(self.n_equations)]

    def _get_solution(self):
        if self.solver.solution is not None:
            return [self.solver.solution[:, i].detach().cpu().numpy() for i in range(self.n_equations)]
        else: return None
        
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
        fig, axs = plt.subplots(self.n_equations, figsize=(10, 6 * self.n_equations), squeeze=False)

        for i in range(self.n_equations):
            ax = axs[i, 0]
            ax.plot(self.domain, self.output[i], label='Approximated', color='red')            
            if self.solution is not None:
                ax.plot(self.domain, self.solution[i], label='Exact', color='blue')
            ax.set_xlabel('$t$')
            ax.set_ylabel(f'$y_{i+1}(t)$' if self.n_equations > 1 else '$y(t)$') 
            ax.legend()
            ax.grid(True)

        plt.tight_layout(pad=3.0)
        plt.suptitle('Solution plot', fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.show()


class ODEPlotter(BasePlotter):
    def __init__(self, solver):
        super().__init__(solver)
        self.derivatives = self._get_derivatives()
        
    def _get_derivatives(self):
        domain = self.solver.domain
        output = self.model(domain)
        derivatives = self.solver._compute_gradients(domain, output)
        return [d.detach().cpu().numpy() for d in derivatives]

    def phase_space(self, exact_derivatives=None):
        count = self.n_equations if self.system else self.solver.ode_order
        if count == 2:
            self._plot_2d_phase_space(exact_derivatives)
        elif count == 3:
            self._plot_3d_phase_space(exact_derivatives)
        else:
            raise ValueError('Can only plot phase space for ODE systems or ODEs of order 2 and 3')

    def extended_phase_space(self, exact_derivatives=None):
        ode_order = self.solver.ode_order
        if ode_order == 2 and not self.system:
            self._plot_3d_extended_phase_space(exact_derivatives)
        else:
            raise ValueError('Can only plot phase space for ODEs of order 2')
        
    def _plot_2d_phase_space(self, exact_derivatives):
        plt.figure(figsize=(10, 6))
        if self.system:
            plt.plot(self.output[0], self.output[1], label='Approximated', color='red')
        else:
            plt.plot(self.output[0], self.derivatives[0], label='Approximated', color='red')
        if self.solution is not None:
            if self.system:
                plt.plot(self.solution[0], self.solution[1], label='Exact', color='blue')
            elif exact_derivatives is not None:
                exact_dy = exact_derivatives[0](self._get_domain(torch=True)).detach().cpu().numpy()
                plt.plot(self.solution[0], exact_dy, label='Exact', color='blue')            
        plt.xlabel('$y_1(t)$' if self.system else '$y(t)$')
        plt.ylabel('$y_2(t)$' if self.system else '$y\'(t)$')
        plt.grid(True)
        plt.legend()
        plt.title('Phase space')
        plt.show()
        
    def _plot_3d_phase_space(self, exact_derivatives):
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')
        if self.system:
            ax.plot(self.output[0], self.output[1], self.output[2], label='Approximated', color='red')
        else:
            ax.plot(self.output[0], self.derivatives[0], self.derivatives[1], label='Approximated', color='red')
        if self.solution is not None:
            if self.system:
                ax.plot(self.solution[0], self.solution[1], self.solution[2], label='Exact', color='blue')
            elif exact_derivatives is not None:
                exact_dy = exact_derivatives[0](self._get_domain(torch=True)).detach().cpu().numpy()
                exact_dy2 = exact_derivatives[1](self._get_domain(torch=True)).detach().cpu().numpy()
                ax.plot(self.solution[0], exact_dy, exact_dy2, label='Exact', color='blue')
        ax.set_xlabel('$y_1(t)$' if self.system else '$y(t)$')
        ax.set_ylabel('$y_2(t)$' if self.system else '$y\'(t)$')
        ax.set_zlabel('$y_3(t)$' if self.system else '$y\'\'(t)$')
        plt.grid(True)
        plt.legend()
        plt.title('Phase space')
        plt.show()
        
    def _plot_3d_extended_phase_space(self, exact_derivatives):
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.domain, self.output[0], self.derivatives[0], label='Approximated', color='red')
        if exact_derivatives is not None and self.solution is not None:
            exact_dy = exact_derivatives[0](self._get_domain(torch=True)).detach().cpu().numpy()
            ax.plot(self.domain, self.solution[0], exact_dy, label='Exact', color='blue')
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


def plot_figure(domain, outputs, solution=None, epoch=None, loss=None):
    n_equations = outputs.shape[1] if outputs.dim() > 1 else 1
    domain_np = domain.detach().cpu().numpy()
    
    fig, axs = plt.subplots(n_equations, 1, figsize=(10, 6 * n_equations), squeeze=False)
    
    for i in range(n_equations):
        ax = axs[i, 0]
        
        if solution is not None:
            solution_np = solution[:, i].detach().cpu().numpy() if solution.dim() > 1 else solution.detach().cpu().numpy()
            ax.plot(domain_np, solution_np, label='Actual')
            
        outputs_np = outputs[:, i].detach().cpu().numpy() if outputs.dim() > 1 else outputs.detach().cpu().numpy()
        ax.plot(domain_np, outputs_np, 'r', label='Predicted')
        
        ax.set_title(f'Equation {i+1}' if n_equations > 1 else '')
        ax.set_xlabel('$t$')
        ax.set_ylabel(f'$y_{i+1}(t)$' if n_equations > 1 else '$y(t)$')
        ax.set_xlim([domain_np.min(), domain_np.max()])
        plt.suptitle(f'Epoch {epoch} Loss: {loss:.8f}')
        if solution is not None:
            ax.set_ylim([solution_np.min()-1, 1.5*solution_np.max()])
        else:
            ax.set_ylim([outputs_np.min()-1, 1.5*outputs_np.max()])
            
        ax.legend(loc='upper right')
        ax.grid(True)
        
        
    if not os.path.exists('temp'):
        os.makedirs('temp')
        
    save_path = f'temp/plot_epoch_{epoch}.png'
    
    plt.savefig(save_path)
    plt.close()