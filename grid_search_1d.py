from klee_minty import *
from solver import *
import pandas as pd
from tabulate import tabulate

dimension_grid = [2, 4, 6, 10, 14]#, 18, 20]#, 22, 24, 26] # note that extended grid search uses a reduced dimension grid
tol = 1e-2
valb = 20
variables = ["num_iterations", "elapsed_time", "max_value"] # %costfunctionerror, %error
steps = [0.5]#[0.01, .05, .1, .3, .5, .7, .8, .9, .95, .99]
headers = ["Dimensão", "Simplex", "PI", "Híbrido"]
tables = {}
for var in variables: # instancia lista para tabelas
    tables[var] = []

for dim in dimension_grid:

    A, b, c = klee_minty(dimensions=dim, val_b=valb)

    # Simplex

    sol_simplex = simplex(A, b, c)
    simplex_var = [dim] + [sol_simplex[var] for var in variables[:-1]] # assume zero de erro
    simplex_var.append(0)
    optimum = sol_simplex["max_value"]

    # IP

    sol_ip = interior_point(A, b, c, alpha0=step, tolerance=tol)
    solution_value_ip = sol_ip["max_value"]
    ip_value_error = (1 - (solution_value_ip/optimum))*100
    ip_var = [dim] + [sol_ip[var] for var in variables[:-1]]
    ip_var.append(ip_value_error)


    # Hybrid: requires more attention, returns mutliple values for "var"

    sol_hybrid = hybrid(A, b, c, alpha0=step, tolerance=tol)
    # solution_value_hybrid = sol_hybrid["max_value"] # 0% de erro também?
    # hybrid_value_error = (1 - solution_value_hybrid/optimum)
    hybrid_var = [dim] + [sol_hybrid[var]["total"] for var in variables[:-1]] + [0] # assume zero de erro

    for v, var in enumerate(variables):
        row = []
        row.append(dim)
        row.append(simplex_var[v+1])
        row.append(ip_var[v+1])
        row.append(hybrid_var[v+1])
        tables[var].append(row)

for table in tables.values():
    print(tabulate(table, headers, tablefmt="latex"))
