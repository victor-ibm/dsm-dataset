# Dataset of doubly stochastic matrices

### Dataset 
Files follows the structure ```dataset_n_s_sparse/dense.json``` where ```n``` is the size of the doubly stochstic matrix, ```s``` the number of samples in the dataset, and ```sparse/dense``` indicates whether the doubly stochastic matrices are sparse or dense. Sparse matrices are generated by sampling $n$ permutations matrices, and dense matrices with $n^2$ permutations. 

### How doubly stochastic matrices are generated
Doubly stochastic matrices are generated by sampling $k$ permutation matrices uniformly at random with the ```randperm``` function in the  [Random](https://docs.julialang.org/en/v1/stdlib/Random/) package in Julia. 

For the weights, we draw $k - 1$ integers uniformly at random in the interval $[0, 10^d]$ where $d = 2 + \text{number of digits} (k^2)$. Sort the $k-1$ integers in ascending order and add $0$ and $10^d$ to the list. Compute the differences between the integers. The weights are the difference between the integers divided by $10^d$. Note the weight will have at most $d$ decimal digits.

For example, if $k=4$, we have that $d = 4$ and draw $3$ integers in the interval $[1,1000]$. Suppose the integers drawn are $[1000, 5000, 7000]$. Adding $0$ and $10000$ to the list we have $[0, 1000, 5000, 7000, 10000]$. The differences between the integers in the list are $[1000, 4000, 2000, 3000]$. Dividing the elments in the list by $10^d$, we obtain $[0.1,0.4,0.2,0.3]$.

### How to read the files
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
