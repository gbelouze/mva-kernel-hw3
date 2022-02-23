# in class KernelSVC
def fit(self, X, y):
    K = self.kernel.kernel(X, X)
    Y = np.diag(y)
    YKY = Y @ K @ Y
    N = len(y)

    def loss(alpha: np.ndarray):
        """Lagrange dual problem"""
        assert alpha.ndim <= 1, "alpha must be 0 or 1 dimensional"
        return 1 / 2 * alpha.T @ YKY @ alpha - alpha.sum()

    def grad_alpha_loss(alpha: np.ndarray):
        """Partial derivate of Ld on alpha"""
        assert alpha.ndim <= 1, "alpha must be 0 or 1 dimensional"
        return YKY @ alpha - np.ones(alpha.shape)

    def equality_constraint(alpha):
        return alpha @ y

    def jacobian_eq_cons(alpha):
        return y

    def inequality_constraint(alpha):
        return np.concatenate((alpha, self.C - alpha), axis=0)

    def jacobian_ineq_cons(alpha):
        return np.concatenate((np.diag(np.ones(N)), -np.diag(np.ones(N))), axis=0)

    constraints = (
        {"type": "eq", "fun": equality_constraint, "jac": jacobian_eq_cons},
        {"type": "ineq", "fun": inequality_constraint, "jac": jacobian_ineq_cons},
    )

    optRes = optimize.minimize(
        fun=loss,
        x0=np.ones(N),
        method="SLSQP",
        jac=grad_alpha_loss,
        constraints=constraints,
    )
    self.alpha = optRes.x

    self.alpha[self.alpha < self.epsilon] = 0
    self.alpha[self.alpha > self.C - self.epsilon] = self.C

    self.beta_support = self.alpha[self.alpha > 0] * y[self.alpha > 0]
    self.X_support = self.support = X[self.alpha > 0]
    self.b = np.mean(
        (1 / y - K @ Y @ self.alpha)[(0 < self.alpha) & (self.alpha < self.C)]
    )
    self.norm_f = self.alpha.T @ YKY @ self.alpha
