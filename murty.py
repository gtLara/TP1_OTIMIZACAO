from math import sqrt
from numpy import stack, append, array
from numpy.linalg import matrix_rank, solve as solve_linear_equation


def point_to_vertex(point, A, b):
    """
    Finds vertex of the solid {Ax <= b and x >= 0} which is closest to the point given
    :param point: Point in space
    :return: Vertex of the solid which is closest to the point given
    """
    feasible_region = _FeasibleRegion(A, b)
    half_spaces = feasible_region.all_half_spaces
    # Sort the faces of the polytope by the distance to the point
    half_spaces.sort(key=lambda half_space: half_space.distance(point))

    # Choose the faces, one by one, in order, until a vertex is found
    rank = half_spaces[0].dimension
    faces = []
    for half_space in half_spaces:
        # Check if new face is LI with the other ones already chosen
        # This algorithm is very inefficient, because the rank of the matrix is calculated twice for each new row that
        # I'm trying to add. But since the other parts of the program (SIMPLEX / Interior points) are much slower, I
        # won't bother to make this more efficient.
        if is_linear_independent(half_space.normal, list(map(lambda face: face.normal, faces))):
            faces.append(half_space)
            if len(faces) == rank:
                break
    A = array(list(map(lambda face: face.normal, faces)))
    b = array(list(map(lambda face: face.free_const, faces)))
    # Now that we have sufficient faces, we just need to find their intersection:
    return solve_linear_equation(A, b)


class _HalfSpace:
    """
    Region of space 'under' a given hyperplane
    """

    def __init__(self, n, k):
        """
        Initialize new half-space of the form n*x <= k
        :param n: Normal vector to hyperplane that cuts the half-space
        :param k: Free scalar constant
        """
        self.normal = n
        self.free_const = k

    @property
    def dimension(self):
        """Number of dimensions in which hyperplane lives"""
        return len(self.normal)

    def distance(self, point):
        """Calculates distance between hyperplane and given point"""
        normal_length = sqrt(_dot_product(self.normal, self.normal))
        return abs(_dot_product(self.normal, point) - self.free_const) / normal_length

    def __str__(self):
        return f'{self.normal} | {self.free_const}'


class _FeasibleRegion:
    """Feasible region of the LP problem"""

    def __init__(self, A, b):
        """
        Initialize new feasible region of the form:
        Ax <= b
        x >= 0
        """
        # Convert the matrix inequality into a list of half-spaces:
        self.__half_spaces = []
        for i, row in enumerate(A):
            self.__half_spaces.append(_HalfSpace(row, b[i]))

    @property
    def dimension(self):
        """
        Number of dimensions in which the region lives
        """
        if len(self.__half_spaces) == 0:
            return 0
        return self.__half_spaces[0].dimension

    @property
    def all_half_spaces(self):
        """Returns all the half-spaces which make up the region, including the xi >= 0 ones"""
        all_half_spaces = self.__half_spaces.copy()
        for i in range(self.dimension):
            normal = [0] * self.dimension
            normal[i] = -1
            new_hyperplane = _HalfSpace(normal, 0)
            all_half_spaces.append(new_hyperplane)
        return all_half_spaces

    def __str__(self):
        return '\n'.join(map(str, self.__half_spaces))


def is_linear_independent(new_vector, vectors):
    """Checks if new_vector is LI from given vectors"""
    if len(vectors) == 0:
        return True
    A = stack(vectors)
    rank_before = matrix_rank(A)
    A_new = append(A, [new_vector], axis=0)
    rank_after = matrix_rank(A_new)
    return rank_after != rank_before


def _dot_product(v1, v2):
    """Dot (scalar) product between two vectors"""
    product = 0
    for i in range(len(v1)):
        product += v1[i] * v2[i]
    return product
