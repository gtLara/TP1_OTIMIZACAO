from numpy import array, identity

def klee_minty(dimensions=2, val_b=5):
    """
    Generate klee minty restrictions of the form:
    Maximize c*x
    Subject to A*x <= b and x >= 0
    b[i] = val_b ^ i
    :param dimensions: Number of dimensions
    :param val_b: Constant for restrictions (5 <= val_b <= 100)
    :return Tuple with A, b, c
    :c: cost function
    :A: restriction matrix
    :b: restriction vector (????)
    """
    b = []
    c = []
    A = identity(dimensions)

    for i in range(dimensions):
        b.append(val_b ** i)
        c.append(2 ** (dimensions - i - 1))
        for j in range(i):
            A[i][j] = 2 ** (i - j + 1)

    b = array(b)
    c = array(c)
    return A, b, c
