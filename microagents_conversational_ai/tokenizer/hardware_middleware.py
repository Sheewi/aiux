"""
Hardware-Aware Token Rewriting Middleware
- Downgrades tokens if hardware (e.g., GPU) is unavailable
- Adds fallback priority for graceful degradation
"""
class HardwareProfile:
    def __init__(self, gpu_available=True, cpu_cores=4, ram_gb=8):
        self.gpu_available = gpu_available
        self.cpu_cores = cpu_cores
        self.ram_gb = ram_gb

class Token:
    def __init__(self, type_, args=None, priority=1.0):
        self.type = type_
        self.args = args or {}
        self.priority = priority

    def __repr__(self):
        return f"<Token type={self.type} priority={self.priority}>"

def simplify_args(args):
    # Example: downgrade selectors, reduce complexity
    args = args.copy()
    if 'selector' in args:
        args['selector'] = 'simple'  # fallback to simple selector
    return args

def rewrite_for_hardware(token, hw_profile):
    if getattr(hw_profile, 'gpu_available', True):
        return token
    else:
        return Token(
            type_=token.type,
            args=simplify_args(token.args),
            priority=token.priority + 0.2
        )
