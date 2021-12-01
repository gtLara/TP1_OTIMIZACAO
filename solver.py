from time import time
from functools import wraps
from murty import point_to_vertex
from scipy.optimize import linprog


# See documentation for the SciPy functions at:
# https://docs.scipy.org/doc/scipy/reference/optimize.linprog-interior-point.html
# https://docs.scipy.org/doc/scipy/reference/optimize.linprog-revised_simplex.html


def _timer(func):
    """Decorator that times each solution"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        solution = func(*args, **kwargs)
        end = time()
        solution['elapsed_time'] = end - start
        return solution

    return wrapper


@_timer
def _generic_solver(method, A, b, c, x0=None, options=None, optimum=None):
    """
    Solve LP program of the form:
    maximize c*x with: A*x <= b and x >= 0
    :param method: revised simplex OR interior-point
    :return: Dictionary with solution info
    """
    header = f'SIMPLEX - Dim: {len(c)}' if method == 'revised simplex' else \
        f'INTERIOR POINT - Dim: {len(c)} / Alpha0: {options["alpha0"]} / Tolerance: {options["tol"]}'
    try:
        scipy_solution = linprog(c=-c, A_ub=A, b_ub=b, method=method, x0=x0,
                                 options=options)
    except Exception as error:
        return {'header': header, 'error': error}
    if optimum is None:
        solution = {'header': header, 'message': scipy_solution.message, 'status': scipy_solution.status,
                    'max_value': -scipy_solution.fun, 'solution': scipy_solution.x, 'num_iterations': scipy_solution.nit, 'solution_error':0.0}
    else:
        solution = {'header': header, 'message': scipy_solution.message, 'status': scipy_solution.status,
                    'max_value': -scipy_solution.fun,
                    'solution': scipy_solution.x,
                    'num_iterations': scipy_solution.nit, 'solution_error':round(100*(1 - (-scipy_solution.fun/optimum)))}
    return solution


def simplex(A, b, c, x0=None, max_iterations=100000000):
    """
    Uses revised SIMPLEX algorithm to solves LP problem of the form:
    maximize c*x with: A*x <= b and x >= 0
    :param x0: Initial vertex (Default is origin)
    :param max_iterations: Maximum number of iterations to be run
    :return: Dictionary with solution info
    """
    return _generic_solver('revised simplex', A, b, c, x0, {'maxiter': max_iterations})


def interior_point(A, b, c, alpha0=0.99995, tolerance=1e-8, max_iterations=100000000, optimum=None):
    """
    Uses Mosek interior point algorithm to solves LP problem of the form:
    maximize c*x with: A*x <= b x >= 0
    :param alpha0: Size of step
    :param tolerance: Termination tolerance
    :param max_iterations: Maximum number of iterations to be run
    :return: Dictionary with solution info
    """
    return _generic_solver('interior-point', A, b, c,
                           options={'maxiter': max_iterations, 'alpha0': alpha0, 'tol': tolerance}, optimum=optimum)


def hybrid(A, b, c, alpha0=0.99995, tolerance=1e-8, max_iterations=100000000):
    solution_ip = interior_point(A, b, c, alpha0, tolerance, max_iterations)
    x0_simplex = None if 'error' in solution_ip else point_to_vertex(solution_ip['solution'], A, b)
    solution_simplex = simplex(A, b, c, x0_simplex, max_iterations)
    return _merge_solutions(solution_ip, solution_simplex)


def _merge_solutions(solution_int_point, solution_simplex):
    """
    Merges solutions into a single solution for the hybrid algorithm
    """
    solution = {'header': solution_int_point['header'].replace('INTERIOR POINT', 'HYBRID')}
    for key, value in solution_int_point.items():
        if key == 'header':
            continue
        solution[key] = {'interior_point': value}
    for key, value in solution_simplex.items():
        if key == 'header':
            continue
        if key in solution:
            solution[key]['simplex'] = value
            if key == 'num_iterations' or key == 'elapsed_time':
                solution[key]['total'] = solution[key]['interior_point'] + solution[key]['simplex']
    return solution

def solution_to_str(solution, level=1, hide=('message', 'status', 'solution')):
    """Useful script to have solution as a more user-friendly text"""
    ans = ''
    for key, value in solution.items():
        if hide and key in hide:
            continue
        if key == 'header':
            ans += f'{value}\n'
            continue
        ans += '\t' * level + key + ': '
        if isinstance(value, dict):
            ans += '\n' + solution_to_str(value, level=level + 1)
        else:
            ans += f'{value}\n'
    return ans
