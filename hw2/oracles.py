import scipy
import numpy as np 


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 1 / 2 * x.T @ self.A @ x - self.b.T @ x

    def grad(self, x):
        return 1 / 2 * (self.A.T + self.A) @ x - self.b


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x))
                 + regcoef / 2 ||x||_2^2.
    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()
    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        exps = np.exp(-self.b * self.matvec_Ax(x))
        res = np.mean(np.log(1 + exps)) \
            + self.regcoef / 2 * np.linalg.norm(x, 2) ** 2
        return res

    def grad(self, x):
        exp = np.exp(-self.b * self.matvec_ATx(x))
        # res = np.mean(exp * self.b.T @ )


def create_log_reg_oracle(A, b, regcoef):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    print('a:', A.shape, 'b:', b.shape)
    print('at:', A.T.shape)

    def matvec_Ax(x):
        return A.dot(x)  # your code here

    def matvec_ATx(x):
        print('x:', x.shape)
        return A.T.dot(x)  # your code here

    def matmat_ATsA(s):
        # your code here
        return A.T @ s @ A

    return LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
