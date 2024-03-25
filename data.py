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
        self.f = function
        
    def evaluate(self, x, *args):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor([[x]], requires_grad=True, dtype=torch.float32)
        return self.f(x, *args) if self.f is not None else None
        
    
class System:
    def __init__(self, system):
        self.f = system
    
    def evaluate(self, x, *args):
        evaluated_args = []
        for arg in args:
            if isinstance(arg, System):
                evaluated_args.append(arg.evaluate(x))
            else:
                evaluated_args.append(arg)
        return torch.stack(self.f(x, *evaluated_args), dim=1).squeeze(-1) if self.f is not None else None
    

class Data:
    def __init__(self, domain, equation=None, ics=None, solution=None):
        self.domain = domain.linspace
        self.equation = equation
        self.solution = solution
        self.ics = ics