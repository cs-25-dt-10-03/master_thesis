from optimization.SingleEvOptimizer import SingleEvOptimizer
from optimization.multiEvOptimizer import MultiEvOptimizer

def optimizer_factory(optimizer_type: str, **kwargs):
    if optimizer_type == 'single_ev':
        return SingleEvOptimizer(**kwargs)
    elif optimizer_type == 'multi_ev':
        return MultiEvOptimizer(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")