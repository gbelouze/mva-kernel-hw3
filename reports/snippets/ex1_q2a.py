class RBF(Kernel):

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def kernel(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        if X.ndim == 1 and Y.ndim == 1:
            norm = np.sum((X - Y) ** 2)
        elif X.ndim == 1:
            norm = np.sum((X[None, :] - Y) ** 2, axis=1)
        elif Y.ndim == 1:
            norm = np.sum((X - Y[None, :]) ** 2, axis=1)
        else:
            norm = np.sum((X[..., :, None, :] -
                           Y[..., None, :, :]) ** 2, axis=-1)
        return np.exp(-norm / (2 * self.sigma**2))


class Linear(Kernel):

    def kernel(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        if X.ndim == 1 and Y.ndim == 1:
            return X * Y
        elif X.ndim == 1:
            return Y @ X
        elif Y.ndim == 1:
            return X @ Y
        else:
            return np.einsum("...id,...jd->...ij", X, Y)
