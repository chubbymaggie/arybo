import functools
import operator

from arybo.lib import MBA
from pytanque import imm, expand_esf_inplace, simplify_inplace, Vector, esf

class Expr:
    @property
    def nbits(self):
        raise NotImplementedError()

    @property
    def args(self):
        return None

    def init_ctx(self):
        return None

    def eval(self, vec, i, ctxes, use_esf):
        raise NotImplementedError()

    def get_ctx(self, ctxes):
        return ctxes.get(id(self), None)

# Leaves
class ExprCst(Expr):
    def __init__(self, n, nbits):
        assert(n >= 0)
        self.n = n & ((1<<nbits)-1)
        self.__nbits = nbits

    @property
    def nbits(self):
        return self.__nbits

    def eval(self, vec, i, ctxes, use_esf):
        return imm((self.n>>i)&1)

class ExprVar(Expr):
    def __init__(self, v):
        self.v = v

    @property
    def nbits(self):
        return self.v.nbits

    def eval(self, vec, i, ctxes, use_esf):
        return self.v.vec[i]

# Unary ops
class ExprUnaryOp(Expr):
    def __init__(self, arg):
        self.arg = arg

    @property
    def args(self):
        return [self.arg]

    @property
    def nbits(self):
        return self.arg.nbits

class ExprNot(ExprUnaryOp):
    def eval(self, vec, i, ctxes, use_esf):
        return self.arg.eval(vec, i, ctxes, use_esf) + imm(True)

# Nary ops
class ExprNaryOp(Expr):
    def __init__(self, *args):
        self._args = args

    @property
    def args(self):
        return self._args

    @property
    def nbits(self):
        # TODO assert every args has the same size
        return self.args[0].nbits

    @staticmethod
    def compute(vec, i, args, ctx, use_esf):
        raise NotImplementedError()

    def eval(self, vec, i, ctxes, use_esf):
        args = (a.eval(vec, i, ctxes, use_esf) for a in self.args)
        return self.compute(vec, i, args, self.get_ctx(ctxes), use_esf)

# Binary ops
# We can't implement this as a UnaryOp, because we need one context per binary
# operation (and in this case, they would share the same context, leading to
# incorrect results).
class ExprBinaryOp(Expr):
    def __init__(self, X, Y):
        if (X.nbits != Y.nbits):
            raise ValueError("X and Y must have the same number of bits!")
        self.X = X
        self.Y = Y

    @property
    def args(self):
        return [self.X,self.Y]

    @property
    def nbits(self):
        # TODO assert every args has the same size
        return self.X.nbits

    def eval(self, vec, i, ctxes, use_esf):
        X = self.X.eval(vec, i, ctxes, use_esf)
        Y = self.Y.eval(vec, i, ctxes, use_esf)
        return self.compute_binop(vec, i, X, Y, self.get_ctx(ctxes), use_esf)

    @staticmethod
    def compute_binop(vec, i, X, Y, ctx, use_esf):
        raise NotImplementedError()

# Nary ops
class ExprXor(ExprNaryOp):
    @staticmethod
    def compute(vec, i, args, ctx, use_esf):
        return sum(args, imm(0))

class ExprAnd(ExprNaryOp):
    @staticmethod
    def compute(vec, i, args, ctx, use_esf):
        return functools.reduce(lambda x,y: x*y, args)

class ExprOr(ExprNaryOp):
    @staticmethod
    def compute(vec, i, args, ctx, use_esf):
        args = list(args)
        ret = esf(1, args)
        for i in range(2, len(args)+1):
            ret += esf(i, args)
        if not use_esf:
            expand_esf_inplace(ret)
            simplify_inplace(ret)
        return ret

# Binary shifts
class ExprShl(ExprUnaryOp):
    def __init__(self, arg, n):
        super(ExprShl, self).__init__(arg)
        self.n = n

    def eval(self, vec, i, ctxes, use_esf):
        if i < self.n:
            return imm(False)
        return self.arg.eval(vec, i-self.n, ctxes, use_esf)

class ExprLShr(ExprUnaryOp):
    def __init__(self, arg, n):
        super(ExprLShr, self).__init__(arg)
        self.n = n

    def eval(self, vec, i, ctxes, use_esf):
        if i >= self.nbits-self.n:
            return imm(False)
        return self.arg.eval(vec, i+self.n, ctxes, use_esf)

class ExprRol(ExprUnaryOp):
    def __init__(self, arg, n):
        super(ExprRol, self).__init__(arg)
        self.n = n

    def eval(self, vec, i, ctxes, use_esf):
        return self.arg.eval(vec, (i-self.n)%self.nbits, ctxes, use_esf)

class ExprRor(ExprUnaryOp):
    def __init__(self, arg, n):
        super(ExprRor, self).__init__(arg)
        self.n = n

    def eval(self, vec, i, ctxes, use_esf):
        return self.arg.eval(vec, (i+self.n)%self.nbits, ctxes, use_esf)

# Concat/slice/{z,s}ext/broadcast

class ExprExtend(ExprUnaryOp):
    def __init__(self, arg, n):
        super(ExprExtend, self).__init__(arg)
        self.n = n
        self.arg_nbits = self.arg.nbits
        assert(n >= self.nbits)

    @property
    def nbits(self):
        return self.n

class ExprSX(ExprExtend):
    def init_ctx(self):
        return CtxUninitialized

    def eval(self, vec, i, ctxes, *args, **kwargs):
        if (i >= (self.arg_nbits-1)):
            ctx = self.get_ctx(ctxes) 
            last_bit = ctx.get()
            if last_bit is CtxUninitialized:
                last_bit = self.arg.eval(vec, self.arg_nbits-1, ctxes, *args, **kwargs)
                ctx.set(last_bit)
            return last_bit
        return self.arg.eval(vec, i, ctxes, *args, **kwargs)

class ExprZX(ExprExtend):
    def eval(self, vec, i, *args, **kwargs):
        if (i >= self.arg_nbits):
            return imm(0)
        return self.arg.eval(vec, i, *args, **kwargs)

class ExprSlice(ExprUnaryOp):
    def __init__(self, arg, slice_):
        super(ExprSlice, self).__init__(arg)
        if not isinstance(slice_, slice):
            raise ValueError("slice_ must a slice object")
        if (not slice_.step is None) and (slice_.step != 1):
            print(slice_.step)
            raise ValueError("only slices with a step of 1 are supported!")
        self.idxes = list(range(*slice_.indices(self.arg.nbits)))

    @property
    def nbits(self):
        return len(self.idxes)

    def eval(self, vec, i, *args, **kwargs):
        return self.arg.eval(vec, self.idxes[i], *args, **kwargs)

class ExprConcat(ExprNaryOp):
    @property
    def nbits(self):
        return sum((a.nbits for a in self.args))

    def eval(self, vec, i, *args, **kwargs):
        it = iter(self.args)
        cur_arg = next(it)
        cur_len = cur_arg.nbits
        while i >= cur_len:
            i -= cur_arg.nbits
            cur_arg = next(it)
        return cur_arg.eval(vec, i, *args, **kwargs)

class ExprBroadcast(ExprUnaryOp):
    def __init__(self, arg, idx, nbits):
        super(ExprBroadcast, self).__init__(arg)
        self.idx = idx
        self._nbits = nbits

    def init_ctx(self):
        return CtxUninitialized

    @property
    def nbits(self):
        return self._nbits

    def eval(self, vec, i, ctxes, *args, **kwargs):
        ctx = self.get_ctx(ctxes)
        ret = ctx.get()
        if ret is CtxUninitialized:
            ret = self.arg.eval(vec, self.idx, ctxes, *args, **kwargs)
            ctx.set(ret)
        return ret

# Arithmetic ops
class ExprAdd(ExprBinaryOp):
    def init_ctx(self):
        return imm(0)

    @staticmethod
    def compute_binop(vec, i, X, Y, ctx, use_esf):
        carry = ctx.get()
        
        sum_args = simplify_inplace(X+Y)
        ret = simplify_inplace(sum_args + carry)
        # TODO: optimize this like in mba_if
        carry = esf(2, [X, Y, carry])
        if not use_esf:
            expand_esf_inplace(carry)
            simplify_inplace(carry)
        ctx.set(carry)
        return ret

class ExprSub(ExprBinaryOp):
    def init_ctx(self):
        return imm(0)

    @staticmethod
    def compute_binop(vec, i, X, Y, ctx, use_esf):
        carry = ctx.get()
        
        sum_args = simplify_inplace(X+Y)
        ret = simplify_inplace(sum_args + carry)
        carry = esf(2, [X+imm(1), Y, carry])
        if not use_esf:
            expand_esf_inplace(carry)
            carry = simplify_inplace(carry)
        ctx.set(carry)
        return ret

# x*y = x*(y0+y1<<1+y2<<2+...)
class ExprMul(ExprBinaryOp):
    def __init__(self, X, Y):
        super(ExprMul, self).__init__(X,Y)
        nbits = X.nbits
        self._expr = ExprAnd(X, ExprBroadcast(Y, 0, nbits))
        for i in range(1, nbits):
            self._expr = ExprAdd(
                self._expr,
                ExprAnd(ExprShl(X, i), ExprBroadcast(Y, i, nbits)))

    def init_ctx(self):
        return init_ctxes(self._expr)

    def eval(self, vec, i, ctxes, use_esf):
        ctxes_inner = self.get_ctx(ctxes).get()
        return self._expr.eval(vec, i, ctxes_inner, use_esf)

# Generic visitors
def visit_dfs(e, cb):
    if e.args != None:
        for a in e.args:
            visit_dfs(a, cb)
    cb(e)

# Evaluator
class CtxWrapper:
    def __init__(self, v):
        self.__v = v
    def get(self): return self.__v
    def set(self, v): self.__v = v

class _CtxUninitialized:
    pass
CtxUninitialized = _CtxUninitialized()

def init_ctxes(e):
    all_ctxs = dict()
    def set_ctx(e_):
        ctx = e_.init_ctx()
        if not ctx is None:
            all_ctxs[id(e_)] = CtxWrapper(ctx)
    visit_dfs(e, set_ctx)
    return all_ctxs

def eval_expr(e,use_esf=False):
    all_ctxs = init_ctxes(e)

    ret = Vector(e.nbits)
    for i in range(e.nbits):
        ret[i] = e.eval(ret, i, all_ctxs, use_esf)
    mba = MBA(len(ret))
    return mba.from_vec(ret)
