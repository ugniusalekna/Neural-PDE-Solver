import numpy as np
import torch


class Interval:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self._linspace = torch.linspace(start, end, steps).unsqueeze(1)
        
    @property # to call w/o ()
    def linspace(self):
        return self._linspace.requires_grad_(True)
    
    def __repr__(self):
        return f'Interval({self.start}, {self.end}, steps={self.steps})'
        
        
class Function:
    def __init__(self, function):
        self.function = function
        
    def evaluate(self, x, *args):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor([x], requires_grad=True, dtype=torch.float32)
        return self.function(x, *args) if self.function is not None else None
        
    
class System:
    def __init__(self, system):
        self.system = system
    
    def evaluate(self, x, *args):
        evaluated_args = []
        for arg in args:
            if isinstance(arg, System):
                evaluated_args.append(arg.evaluate(x))
            else:
                evaluated_args.append(arg)
                
        return torch.stack(self.system(x, *evaluated_args), dim=1).squeeze(-1) if self.system is not None else None
    

class Data:
    def __init__(self, domain, equations=None, ics=None, solutions=None):
        self.domain = domain.linspace
        self.equation_fn = None
        self.equation_sys = None
        self.solution_fn = None
        self.solution_sys = None
        self.ics = ics
        
        self._assign_equations(equations)
        self._assign_solutions(solutions)

    def _assign_equations(self, equations):
        if isinstance(equations, Function):
            self.equation_fn = equations
        elif isinstance(equations, System):
            self.equation_sys = equations

    def _assign_solutions(self, solutions):
        if isinstance(solutions, Function):
            self.solution_fn = solutions
        elif isinstance(solutions, System):
            self.solution_sys = solutions