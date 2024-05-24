# DSM Dataset

Dataset with $n \times n$ (scaled) doubly stochastic matrices. Entries are non-negative and rows and columns sum to $s$. Each entry in the ```json``` file has the fields:
  - **n**: ```Int``` with size of the (scaled) doubly stochastic matrix
  - **s:** ```Int``` with the scale of the doubly stochastic matrix   
  - **scaled_doubly_stochastic_matrix:** ```Array{Int}``` with the scaled doubly stochastic matrix (columns stacked as a vector)
  - **weights:** ```Array{Int}```of length $k$. The array entries sum to $s$ 
  - **permutations:** ```Array{Int}``` with $k$ permutation vectors of length $n$ stacked as a sigle vector

**File structure:** ```qbench_n_sparse/dense.json``` where ```n``` is the size of the doubly stochstic matrix and ```sparse/dense``` indicates whether the doubly stochastic matrices are sparse or dense. Sparse and dense matrices are generated by sampling $k=n$ and $k=n^2$ permutations, respectively.


### How doubly stochastic matrices are generated
The (scaled) doubly stochastic matrices are generated by sampling $k$ permutation matrices uniformly at random with the ```randperm``` function in the  [Random](https://docs.julialang.org/en/v1/stdlib/Random/) package in Julia. 

For the weights, we draw $k - 1$ integers uniformly at random in the interval $[0, s]$ where $s = 10^d$ with $d = 2 + \text{number of digits} (k^2)$. Sort the $k-1$ integers in ascending order and add $0$ and $10^d$ to the list. Compute the differences between the integers. The weights are the difference between the integers. Example with $k=4$: $d$ is equal to $4$ and draw $3$ integers in the interval $[1,1000]$. Suppose the integers drawn are $[1000, 5000, 7000]$. Adding $0$ and $10000$ to the list we have $[0, 1000, 5000, 7000, 10000]$. The differences between the integers in the list are $[1000, 4000, 2000, 3000]$. Note the integers in the list sum to $s$. 

### How to read the files
In Julia:
```julia
using JSON3

data = JSON3.read("file_name.json")
D = data[id].doubly_stochastic_matrix
D = reshape(D,n,n)
```

In Python:
```python
import json
import numpy as np 

f = open("file_name.json")
data = json.load(f)
D = data[id]['scaled_doubly_stochastic_matrix']
D = np.array(D)
D = np.reshape(D, (n,n))
```
