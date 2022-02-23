# in class KernelSVR
def fit(self, X, y):
    K = self.kernel.kernel(X, X)
    N = len(y)

    def loss(alpha: np.ndarray):
        """Lagrange dual problem"""
        alpha_p, alpha_m = alpha[:N], alpha[N:]
        diff = alpha_p - alpha_m
        return 0.5 * diff.T @ K @ diff - y @ diff + self.eta * alpha.sum()

    # Partial derivate of Ld on alpha
    def grad_alpha_loss(alpha: np.ndarray):
        """Partial derivate of Ld on alpha"""
        alpha_p, alpha_m = alpha[:N], alpha[N:]
        diff = alpha_p - alpha_m

        grad_p = K @ diff - y + self.eta
        grad_m = -K @ diff + y + self.eta

        return np.concatenate((grad_p, grad_m), axis=0)

    def equality_constraint(alpha):
        return np.sum(alpha[:N] - alpha[N:])

    def jacobian_eq_cons(alpha):
        return np.concatenate((np.ones(N), -np.ones(N)), axis=0)

    def inequality_constraint(alpha):
        return np.concatenate((alpha, self.C - alpha), axis=0)

    def jacobian_ineq_cons(alpha):
        return np.concatenate(
            (np.diag(np.ones(2 * N)), -np.diag(np.ones(2 * N))), axis=0
        )

    constraints = (
        {"type": "eq", "fun": equality_constraint, "jac": jacobian_eq_cons},
        {"type": "ineq", "fun": inequality_constraint, "jac": jacobian_ineq_cons},
    )

    optRes = optimize.minimize(
        fun=loss,
        x0=self.C * np.ones(2 * N),
        method="SLSQP",
        jac=grad_alpha_loss,
        constraints=constraints,
        tol=1e-7,
    )
    self.alpha = optRes.x

    self.alpha[self.alpha < self.epsilon] = 0
    self.alpha[self.alpha > self.C - self.epsilon] = self.C

    alpha_p, alpha_m = self.alpha[:N], self.alpha[N:]
    beta = alpha_p - alpha_m
    support_indices = np.abs(beta) > self.epsilon

    self.X_support = X[support_indices]
    self.support = np.stack(
        (X[support_indices].squeeze(), y[support_indices]), axis=-1
    )
    self.beta_support = (alpha_p - alpha_m)[support_indices]

    margin_p = np.mean(
        (y - K @ beta - self.eta)[(0 < alpha_p) & (alpha_p < self.C)]
    )
    margin_m = np.mean(
        (y - K @ beta + self.eta)[(0 < alpha_m) & (alpha_m < self.C)]
    )
    self.b = 0.5 * (margin_p + margin_m)
