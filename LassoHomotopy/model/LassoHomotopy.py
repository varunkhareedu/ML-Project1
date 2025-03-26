import numpy as np

class LassoHomotopyModel():
    def __init__(self, lambda_val=None, tol=1e-6, max_iter=500):
        self.lambda_val = lambda_val
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n_samples, n_features = X.shape

        # Normalize features and center target
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        X = (X - self.mean_) / self.std_
        y_mean = y.mean()
        y = y - y_mean

        # Initialization
        beta = np.zeros((n_features, 1))
        residual = y.copy()
        active_set = []
        inactive_set = list(range(n_features))
        mu = 0
        step = 0

        while step < self.max_iter:
            # Compute correlations
            corr = X.T @ residual
            c = corr.flatten()

            # Choose variable with max correlation
            c_abs = np.abs(c)
            C = c_abs.max()
            if self.lambda_val is None:
                self.lambda_val = C

            if C < self.lambda_val + self.tol:
                break

            j = np.argmax(c_abs)
            if j not in active_set:
                active_set.append(j)
                inactive_set.remove(j)

            # Sign of correlation for active variables
            s = np.sign(c[active_set]).reshape(-1, 1)

            # Compute direction (equivalent to LARS step)
            Xa = X[:, active_set]
            G = Xa.T @ Xa
            G_inv = np.linalg.pinv(G)
            A = 1 / np.sqrt(s.T @ G_inv @ s)
            w = A * (G_inv @ s)  # direction in coefficient space
            w = w.flatten()
            u = Xa @ w.reshape(-1, 1)  # direction in prediction space

            # Determine how far we can go in that direction
            a = X.T @ u
            gamma_candidates = []

            for k in inactive_set:
                if (A - a[k]) != 0:
                    gamma1 = (C - c[k]) / (A - a[k])
                    if gamma1 > 0:
                        gamma_candidates.append(gamma1)
                if (A + a[k]) != 0:
                    gamma2 = (C + c[k]) / (A + a[k])
                    if gamma2 > 0:
                        gamma_candidates.append(gamma2)

            # Also check when any coefficient would become zero (drop from active)
            gamma3 = np.inf
            for idx, i in enumerate(active_set):
                if w[idx] != 0:
                    gamma_i = -beta[i] / w[idx]
                    if gamma_i > 0:
                        gamma3 = min(gamma3, gamma_i)

            if gamma_candidates:
                gamma = min(min(gamma_candidates), gamma3)
            else:
                gamma = gamma3

            if np.isinf(gamma):
                break

            # Update coefficients and residual
            w = w.flatten()
            for idx, i in enumerate(active_set):
                beta[i, 0] = beta[i, 0] + gamma * w[idx]

            residual = y - X @ beta
            step += 1

            # Remove if any coefficient is effectively zero
            for i in active_set[:]:
                if abs(beta[i]) < self.tol:
                    active_set.remove(i)
                    inactive_set.append(i)

        # Denormalize coefficients
        self.coef_ = beta.flatten() / self.std_
        self.intercept_ = y_mean - np.dot(self.coef_, self.mean_)
        return LassoHomotopyResults(self.coef_, self.intercept_)

class LassoHomotopyResults():
    def __init__(self, coef_, intercept_):
        self.coef_ = coef_
        self.intercept_ = intercept_

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
