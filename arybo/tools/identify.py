import six
import operator
from six.moves import reduce

import arybo.lib.mba_exprs as EX
from arybo.lib import MBA
from pytanque import Vector, Matrix, imm

def _identify_nonaffine(app,var_in):
    raise NotImplemented()

def identify(app,in_name):
    # TODO: identify number of independant inputs
    NL = app.nl().vector()
    M = app.matrix()
    nbits_in = M.ncols()
    nbits_out = M.nlines()
    if nbits_in != nbits_out:
        raise ValueError("do not yet support application with a different\
                number of input and output bits!")
    mba = MBA(nbits_in)
    var_in = mba.var(in_name)

    if NL != Vector(len(NL)):
        return _identify_nonaffine(app,var_in)
    C = EX.ExprCst(mba.from_vec(app.cst()).to_cst(), nbits_out)
    if M == Matrix(nbits_out, nbits_in):
        # This is just a constant
        return C

    ret = EX.ExprBV(var_in)
    matrix_empty = 0
    # Find empty columns in the matrix.
    for j in range(nbits_in):
        is_zero = reduce(operator.and_, (M.at(i,j) == imm(0) for i in range(nbits_out)), True)
        if is_zero:
            matrix_empty |= 1<<j
    matrix_and_mask = (~matrix_empty)%(2**nbits_out)
    if matrix_empty != 0:
        ret = EX.ExprAnd(ret, EX.ExprCst(matrix_and_mask, nbits_out))
    if mba.from_vec(M*var_in.vec)^(var_in & matrix_empty) == var_in:
        # This is a XOR
        return EX.ExprXor(ret, C)
    fds
