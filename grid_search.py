from klee_minty import *
from solver import *
import pandas as pd
from tabulate import tabulate

dimension_grid = [2, 4, 6, 10, 14]# 18, 20, 22, 24, 26]
tol = 1e-2
step = .5
valb = 100
variables = ["num_iterations", "elapsed_time"]
headers = ["Dimensão", "Simplex", "PI", "Híbrido"]
tables = {}
for var in variables: # instancia lista para tabelas
    tables[var] = []

for dim in dimension_grid:

    A, b, c = klee_minty(dimensions=dim, val_b=5)

    # Simplex 

    sol_simplex = simplex(A, b, c)
    simplex_var = [dim] + [sol_simplex[var] for var in variables]

    # IP

    sol_ip = interior_point(A, b, c, alpha0=step, tolerance=tol)
    ip_var = [dim] + [sol_ip[var] for var in variables]

    # Hybrid: requires more attention, returns mutliple values for "var"

    sol_hybrid = hybrid(A, b, c, alpha0=step, tolerance=tol)
    hybrid_var = [dim] + [sol_hybrid[var]["total"] for var in variables]

    for v, var in enumerate(variables):
        row = []
        row.append(dim)
        row.append(simplex_var[v+1])
        row.append(ip_var[v+1])
        row.append(hybrid_var[v+1])
        tables[var].append(row)

for table in tables.values():
    print(tabulate(table, headers, tablefmt="latex"))
