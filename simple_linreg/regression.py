"""Simple (one-feature) linear regression, implemented from scratch."""
import numpy as np


class LinearRegression:
    """Ordinary least squares for y = intercept + slope * x."""

    def __init__(self):
        self.slope_ = None
        self.intercept_ = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have the same length")
        if x.size < 2:
            raise ValueError("need at least 2 points to fit a line")
        x_mean, y_mean = x.mean(), y.mean()
        sxx = np.sum((x - x_mean) ** 2)
        if sxx == 0:
            raise ValueError("x is constant: the slope is undefined")
        self.slope_ = np.sum((x - x_mean) * (y - y_mean)) / sxx
        self.intercept_ = y_mean - self.slope_ * x_mean
        return self                      # allows model.fit(x, y).predict(...)

    def _check_fitted(self):
        if self.slope_ is None:
            raise RuntimeError("call fit(x, y) before using the model")

    def predict(self, x):
        self._check_fitted()
        return self.intercept_ + self.slope_ * np.asarray(x, dtype=float)

    def score(self, x, y):
        """Coefficient of determination R^2."""
        self._check_fitted()
        y = np.asarray(y, dtype=float).ravel()
        ss_res = np.sum((y - self.predict(x)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / ss_tot)

    def summary(self, x, y):
        """Same dictionary as in Lab 2: intercept, slope, R2, std_err."""
        self._check_fitted()
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = x.size
        ss_res = np.sum((y - self.predict(x)) ** 2)
        sxx = np.sum((x - x.mean()) ** 2)
        std_err = float(np.sqrt(ss_res / (n - 2) / sxx)) if n > 2 else float("nan")
        return {"intercept": float(self.intercept_),
                "slope": float(self.slope_),
                "R2": self.score(x, y),
                "std_err": std_err}
