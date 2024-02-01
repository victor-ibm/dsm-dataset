Load matrix in Julia:
```julia
using JSON3

data = JSON3.read("file_name.json")
D = data[id].doubly_stochastic_matrix
D = reshape(D,n,n)
```


Load matrix in Python:
```python
import json
import numpy as np 

f = open("file_name.json")
data = json.load(f)
D = data[id]['doubly_stochastic_matrix']
D = np.array(D)
D = np.reshape(D, (n,n))
```
