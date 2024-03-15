import numpy as np
import torch

class Point:
    def __init__(self, value, device, grad=True):
        self.value = value
        self.device = device
        self.grad = grad
        self.tensor = torch.tensor([value], requires_grad=grad).to(device)
        
    def __repr__(self):
        return f'Value({self.value}; device={self.device}; grad={self.grad})'
    
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
        
    
class Data:
    def __init__(self, domain, ode=None, ics=None, solution=None):
        
        self.domain = domain.linspace
        
        self.ode = ode
        self.ics = ics
        
        self.solution_fn = solution
        
    @property
    def get_solution(self):
        return self.solution_fn(self.domain) if self.solution_fn is not None else None
        