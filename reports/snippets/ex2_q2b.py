# in class KernelSVR
def f(self, x: ArrayLike):
    """Regression function :maths:`f` evaluated at `x`"""
    K = self.kernel.kernel(self.X_support, x)
    return self.beta_support @ K
