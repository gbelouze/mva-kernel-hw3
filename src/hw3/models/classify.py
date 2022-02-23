import abc

import numpy as np
from hw3.models.kernel import Kernel
from numpy.typing import ArrayLike
from overrides import overrides  # type: ignore
from scipy import optimize  # type: ignore


class Predictor(abc.ABC):
    """Very general blue print for an implementation of a classifier."""

    @abc.abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike):
        """Learns the model to fit input `X` to output `y`."""
        ...

    @abc.abstractmethod
    def f(self, X: ArrayLike):
        """Prediction function.

        This can be directly linked to prediction for regressors, or indirectly linked for classifiers.

        Notes
        -----
        This function cannot be called before the model was fitted with `self.fit` on some training data.
        """

    @abc.abstractmethod
    def predict(self, X: ArrayLike):
        """Predicts output from input `X`.

        Notes
        -----
        This function cannot be called before the model was fitted with `self.fit` on some training data.
        """
        ...


class KernelSVC(Predictor):
    """Kernel Support Vector Classifier"""

    def __init__(self, C: float, kernel: Kernel, epsilon: float = 1e-3):
        assert epsilon < 1

        self.type = "non-linear"
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None

    @overrides(check_signature=False)
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
        self.alpha[self.alpha > self.C * (1 - self.epsilon)] = self.C

        self.beta_support = self.alpha[self.alpha > 0] * y[self.alpha > 0]
        self.X_support = self.support = X[self.alpha > 0]
        self.b = np.mean(
            (1 / y - K @ Y @ self.alpha)[(0 < self.alpha) & (self.alpha < self.C)]
        )
        self.norm_f = self.alpha.T @ YKY @ self.alpha

    @overrides(check_signature=False)
    def f(self, x: ArrayLike):
        """Separating function :maths:`f` evaluated at `x`"""
        K = self.kernel.kernel(self.X_support, x)
        return self.beta_support @ K

    @overrides(check_signature=False)
    def predict(self, X: ArrayLike):
        """Predict y values in {-1, 1}"""
        return 2 * (self.f(X) + self.b > 0) - 1


class KernelSVR(Predictor):
    """Kernel Support Vector Regressor"""

    def __init__(
        self, C: float, kernel: Kernel, eta: float = 1e-2, epsilon: float = 1e-3
    ):
        assert epsilon < 1

        self.C = C
        self.kernel = kernel
        self.alpha = None  # Vector of size 2*N
        self.support = None
        self.eta = eta
        self.epsilon = epsilon
        self.eps = 0.0
        self.type = "svr"

    @overrides(check_signature=False)
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

    @overrides(check_signature=False)
    def f(self, x: ArrayLike):
        """Regression function :maths:`f` evaluated at `x`"""
        K = self.kernel.kernel(self.X_support, x)
        return self.beta_support @ K

    @overrides(check_signature=False)
    def predict(self, X):
        return self.f(X) + self.b
