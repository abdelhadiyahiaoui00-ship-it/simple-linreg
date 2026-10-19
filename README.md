# simple-linreg

Simple linear regression implemented from scratch with NumPy (SEDS Lab 3).

## Installation

```bash
pip install simple-linreg
```

## Usage

```python
from simple_linreg import LinearRegression

x = [1, 2, 3, 4, 5]
y = [2.1, 3.9, 6.2, 7.8, 10.1]

model = LinearRegression().fit(x, y)
model.predict([6])        # array([12.0])
model.score(x, y)         # R^2
model.summary(x, y)       # {'intercept': ..., 'slope': ..., 'R2': ..., 'std_err': ...}
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
