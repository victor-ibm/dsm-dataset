import numpy as np

import sys
sys.path.append('../')


def birkhoff_constraints(n):
    "Function that returns a matrix with the Birkhoff constraints, i.e., sum columns and rows must be equal to 1. " 
    # Initialize M as a numpy array of zeros
    M = np.zeros((n*n, 2*n), dtype=int)

    # Fill the first part of matrix with ones
    for i in range(n):
        M[:,i][(i*n) : (i*n) + n] = np.ones((1,n))

    # Fill the second part of matrix with ones
    for k in range(0,n):
        for i in range(0,n):
            for j in range(0,n):
                if i == j:
                    M[k*n + i, n + j] = 1
                    
    return M


########## DOCPLEX ##########

from docplex.mp.model import Model

class BirkCplex():

    def __init__(self, x_star) -> None:
        self.x_star = x_star
        self.n = int(np.sqrt(self.x_star.size))
        self.time_limit = 60

    def set_time_limit(self, time_limit):
        self.time_limit = time_limit
    
    def solve(self, k=None, p=None):       
        if k is None:
            return self.solve_permutations_and_weights()
        else:
            return None, None, None
        # elif k is not None and p is None:
        #     return self.solve_permutations_and_weights_fixed_size(k)
        # elif p is None:
        #     return self.solve_permutations_and_weights(k)
        # else:
        #     return self.solve_weights(p)


    def solve_permutations_and_weights_fixed_size(self, k):
        """
        Function that computes a decomopsition of a doubly stochastic 
        matrix 'x_star' as the convex combinatino of 'k' permutation matrices. 
        """
        model = Model()
        p = model.binary_var_list(self.n*self.n*k, name='permutations')
        w = model.continuous_var_list(k, name='weights')
        w_aux = model.continuous_var_list(self.n*self.n*k, name='aux_weights')
        A = birkhoff_constraints(self.n)
        M = 10

        # Constraints
        # sum rows and columns must be equal to 1
        for index in range(0,k):
            for j in range(0,2*self.n):
                model.add_constraint(model.sum(A[i,j]*p[self.n*self.n*index + i] for i in range(0,self.n*self.n)) == 1)
        # weights must sum to 1
        model.add_constraint(model.sum(w[index] for index in range(0,k)) == 1)
        # weights need to be larger than 0
        for i in range(0,k):
            model.add_constraint(w[i] >= 0)
        # decomposition must be exact
        for i in range(0,self.n*self.n*k):
            model.add_constraint(w_aux[i] <= M*p[i])
        for i in range(0,k):
            for j in range(self.n*self.n):
                model.add_constraint(w_aux[self.n*self.n*i + j] <= w[i])
        for i in range(0,k):
            for j in range(self.n*self.n):
                model.add_constraint(w_aux[i*k + j] >= w[i] - M*(1-p[i*k + j]))
        for i in range(0,self.n*self.n):
            model.add_constraint(model.sum(w_aux[i + j*self.n*self.n] for j in range(0,k)) == self.x_star[i])

        # Solve
        # model.print_information()
        model.set_time_limit(self.time_limit)
        solution = model.solve()
        # print(model.solve_details.status_code)
        # print(model.get_solve_details().best_bound)
        if (model.solve_details.status_code not in [101]):
                status = 'failed'
                return status, None, None
        
        ## Retrieve solutions
        status = 'success'
        perm = solution.get_value_list(p)
        c = solution.get_value_list(w)


        return status, perm, c

    def solve_weights(self, k, p):
        model = Model()
        w = model.continuous_var_list(k)

        ########## Find weights that minimize the objective SUM(X_star - X)^2 ##########
        # add constraints
        model.add_constraint(model.sum(w[index] for index in range(0,k)) == 1)
        for i in range(0,k):
            model.add_constraint(w[i] >= 0)
        for i in range(0,k):
            model.add_constraint(w[i] <= 1)

        # add objective
        model.minimize(model.sum_squares((model.sum(w[j]*p[i + j*self.n*self.n] for j in range(0,k)) - self.x_star[i]) for i in range(0,self.n*self.n)))

        # solve
        solution = model.solve()
        fval = solution.get_objective_value()
        # num_weights = np.sum([np.array(solution.get_value_list(w)) >= 1e-15])
        # print(f'Status: {model.solve_details.status}')
        # print(f'Number of Weights: {num_weights}')

        ########## Find sparse weights s.t. decomposition is accurate as above ##########
        model = Model()
        w = model.continuous_var_list(k)
        y = model.binary_var_list(k)

        # add constraints
        model.add_constraint(model.sum(w[index] for index in range(0,k)) == 1)
        for i in range(0,k):
            model.add_constraint(w[i] >= 0)
        for i in range(0,k):
            model.add_constraint(w[i] <= 1)
        for i in range(0,k):
            model.add_constraint(w[i] <= y[i])
        
        # add objective
        model.add_constraint((model.sum_squares((model.sum(w[j]*p[i + j*self.n*self.n] for j in range(0,k)) - self.x_star[i]) for i in range(0,self.n*self.n))) <= fval)
        model.minimize(model.sum(y[i] for i in range(0,k)))

        solution = model.solve()
        # num_weights = np.sum([np.array(solution.get_value_list(w)) >= 1e-15])
        # print(f'Status: {model.solve_details.status}')
        # print(f'Number of Weights: {num_weights}')
        # print("\n")

        
        return fval, np.array(solution.get_value_list(w))
    
    def solve_permutations_and_weights(self):
        for k in range(1,(self.n-1)**2 + 2):
            #print(f'iteration: {k}')
            status, p, w = self.solve_permutations_and_weights_fixed_size(k)
            if status == 'success':
                return status, p, w
        return None, None, None

########### Aproximate Birkhoff ###########

def approx_birkhoff(x_star, n, k):

    model = Model()
    p = model.binary_var_list(n*n*k, name='permutations')
    w = model.continuous_var_list(k, name='weights')
    w_aux = model.continuous_var_list(n*n*k, name='aux_weights')
    A = birkhoff_constraints(n)
    M = 10

    # Constraints
    # sum rows and columns must be equal to 1
    for index in range(0,k):
        for j in range(0,2*n):
            model.add_constraint(model.sum(A[i,j]*p[n*n*index + i] for i in range(0,n*n)) == 1)
    # weights must sum to 1
    model.add_constraint(model.sum(w[index] for index in range(0,k)) == 1)
    # weights need to be larger than 0
    for i in range(0,k):
        model.add_constraint(w[i] >= 0)
    # decomposition must be strictly less than x_star
    for i in range(0,n*n*k):
        model.add_constraint(w_aux[i] <= M*p[i])
    for i in range(0,k):
        for j in range(n*n):
            model.add_constraint(w_aux[n*n*i + j] <= w[i])
    for i in range(0,k):
        for j in range(n*n):
            model.add_constraint(w_aux[i*k + j] >= w[i] - M*(1-p[i*k + j]))
    for i in range(0,n*n):
        model.add_constraint(model.sum(w_aux[i + j*n*n] for j in range(0,k)) <= x_star[i])

    # Solve
    # model.print_information()
    solution = model.solve()
    #print(model.solve_details.status_code)
    if (model.solve_details.status_code not in [101]):
            status = 'failed'
            return status, None, None
    
    ## Retrieve solutions
    status = 'success'
    perm = solution.get_value_list(p)
    c = solution.get_value_list(w)

    return status, perm, c