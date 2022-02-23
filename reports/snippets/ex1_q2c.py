# in class KernelSVC
def f(self, x: ArrayLike):
    """Separating function evaluated at `x`"""
    K = self.kernel.kernel(self.X_support, x)
    return self.beta_support @ K
