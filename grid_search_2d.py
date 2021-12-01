from klee_minty import *
from solver import *
import pandas as pd
import pickle

search = "tol" # or tol

dimension_grid = [2, 4, 6, 10]#, 14, 18, 20]
step_grid = [0.01, 0.1, 0.2, 0.5, 0.6, 0.8, 0.9, .95, .99] if search == "step" else [0.5]
duality_gap_grid = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10] if search == "tol" else [1e-2]
variables = ["num_iterations", "elapsed_time", "solution_error"]
valb = 5

# tolerance is responsible for the error. but should it be that high?

tables = {}
hybrid_tables = {}
for var in variables: # instancia lista para tabelas
    tables[var] = []
    if var != "solution_error": hybrid_tables[var] = []

for tol in duality_gap_grid:
    for step in step_grid:

        table_rows = {}
        hybrid_table_rows = {}
        for var in variables:
            table_rows[var] = [step] if search == "step" else [tol]
            if var != "solution_error":
                hybrid_table_rows[var] = [step] if search == "step" else [tol]

        for dim in dimension_grid:
            A, b, c = klee_minty(dimensions=dim, val_b=valb)

            sol_hybrid = hybrid(A, b, c, alpha0=step, tolerance=tol)
            optimum = sol_hybrid["max_value"]["simplex"]

            sol_ip = interior_point(A, b, c, alpha0=step, tolerance=tol, optimum=optimum)

            # for n it only
            for var in variables:
                table_rows[var].append(sol_ip[var])
                if var != "solution_error": hybrid_table_rows[var].append(sol_hybrid[var]["total"])

        for var in variables:
            tables[var].append(table_rows[var])
            if var != "solution_error": hybrid_tables[var].append(hybrid_table_rows[var])

with open(f"{search}_2dtables.pickle", "wb") as handle:
    pickle.dump(tables, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f"{search}_hybrid_2dtables.pickle", "wb") as handle:
    pickle.dump(hybrid_tables, handle, protocol=pickle.HIGHEST_PROTOCOL)
