```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))
```

<img src="https://github.com/user-attachments/assets/6f226d4a-a99b-41a5-89d8-6853e8283eea" width="450" alt="screen_de_la_database">
