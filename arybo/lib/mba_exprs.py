import functools
import operator
import six

from six.moves import range, reduce

from arybo.lib import MBA
from pytanque import imm, expand_esf_inplace, simplify_inplace, Vector, esf

class Expr(object):
    @property
    def nbits(self):
        raise NotImplementedError()

    @property
    def args(self):
        return None

    def init_ctx(self):
        return None

    def eval(self, vec, i, ctx, args, use_esf):
        raise NotImplementedError()

    def __parse_arg(self,v):
        if isinstance(v, six.integer_types):
            return ExprCst(v, self.nbits)
        if not isinstance(v, Expr):
            raise ValueError("argument must be an integer or an Expr")
        return v

    def __check_arg_int(self,v):
        if not isinstance(v, six.integer_types):
            raise ValueError("argument must be an integer")

    def __add__(self, o):
        o = self.__parse_arg(o)
        return ExprAdd(self, o)

    def __radd__(self, o):
        o = self.__parse_arg(o)
        return ExprAdd(o, self)

    def __sub__(self, o):
        o = self.__parse_arg(o)
        return ExprSub(self, o)

    def __rsub__(self, o):
        o = self.__parse_arg(o)
        return ExprSub(o,self)

    def __mul__(self, o):
        o = self.__parse_arg(o)
        return ExprMul(self, o)

    def __rmul__(self, o):
        o = self.__parse_arg(o)
        return ExprMul(o, self)

    def __xor__(self, o):
        o = self.__parse_arg(o)
        return ExprXor(self, o)

    def __rxor__(self, o):
        o = self.__parse_arg(o)
        return ExprXor(o, self)

    def __and__(self, o):
        o = self.__parse_arg(o)
        return ExprAnd(self, o)

    def __rand__(self, o):
        o = self.__parse_arg(o)
        return ExprAnd(o, self)

    def __or__(self, o):
        o = self.__parse_arg(o)
        return ExprOr(self, o)

    def __ror__(self, o):
        return ExprOr(o, self)

    def __lshift__(self, o):
        self.__check_arg_int(o)
        return ExprShl(self, o)

    def __rshift__(self, o):
        self.__check_arg_int(o)
        return ExprLShr(self, o)

    def __neg__(self):
        return ExprSub(ExprCst(0, self.nbits), self)

    def __invert__(self):
        return ExprNot(self)

    def zext(self,n):
        self.__check_arg_int(n)
        return ExprZX(self,n)

    def sext(self,n):
        self.__check_arg_int(n)
        return ExprSX(self,n)

    def rol(self,n):
        self.__check_arg_int(n)
        return ExprRol(self,n)

    def ror(self,n):
        self.__check_arg_int(n)
        return ExprRor(self,n)

    def __getitem__(self,s):
        if not isinstance(s, slice):
            raise ValueError("can only get slices")
        return ExprSlice(self, s)

# Leaves
class ExprCst(Expr):
    def __init__(self, n, nbits):
        assert(n >= 0)
        self.n = n & ((1<<nbits)-1)
        self.__nbits = nbits

    @property
    def nbits(self):
        return self.__nbits

    def eval(self, vec, i, ctx, args, use_esf):
        return imm((self.n>>i)&1)

    def to_cst(self):
        return self.n

class ExprBV(Expr):
    def __init__(self, v):
        self.v = v

    @property
    def nbits(self):
        return self.v.nbits

    def eval(self, vec, i, ctx, args, use_esf):
        return self.v.vec[i]

    def to_cst(self):
        return self.v.to_cst()

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
    def eval(self, vec, i, ctx, args, use_esf):
        return args[0].eval(vec, i, use_esf) + imm(True)

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

    def eval(self, vec, i, ctx, args, use_esf):
        args = (a.eval(vec, i, use_esf) for a in args)
        return self.compute(vec, i, args, ctx, use_esf)

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
        self._nbits = self.X.nbits

    @property
    def args(self):
        return [self.X,self.Y]

    @property
    def nbits(self):
        return self._nbits

    def eval(self, vec, i, ctx, args, use_esf):
        X,Y = args
        X = X.eval(vec, i, use_esf)
        Y = Y.eval(vec, i, use_esf)
        return self.compute_binop(vec, i, X, Y, ctx, use_esf)

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
        return reduce(lambda x,y: x*y, args)

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

    def eval(self, vec, i, ctx, args, use_esf):
        if i < self.n:
            return imm(False)
        return args[0].eval(vec, i-self.n, use_esf)

class ExprLShr(ExprUnaryOp):
    def __init__(self, arg, n):
        super(ExprLShr, self).__init__(arg)
        self.n = n

    def eval(self, vec, i, ctx, args, use_esf):
        if i >= self.nbits-self.n:
            return imm(False)
        return args[0].eval(vec, i+self.n, use_esf)

class ExprRol(ExprUnaryOp):
    def __init__(self, arg, n):
        super(ExprRol, self).__init__(arg)
        self.n = n

    def eval(self, vec, i, ctx, args, use_esf):
        return args[0].eval(vec, (i-self.n)%self.nbits, use_esf)

class ExprRor(ExprUnaryOp):
    def __init__(self, arg, n):
        super(ExprRor, self).__init__(arg)
        self.n = n

    def eval(self, vec, i, ctx, args, use_esf):
        return args[0].eval(vec, (i+self.n)%self.nbits, use_esf)

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

    def eval(self, vec, i, ctx, args, use_esf):
        arg = args[0]
        if (i >= (self.arg_nbits-1)):
            last_bit = ctx.get()
            if last_bit is CtxUninitialized:
                last_bit = arg.eval(vec, self.arg_nbits-1, use_esf)
                ctx.set(last_bit)
            return last_bit
        return arg.eval(vec, i, use_esf)

class ExprZX(ExprExtend):
    def eval(self, vec, i, ctx, args, use_esf):
        if (i >= self.arg_nbits):
            return imm(0)
        return args[0].eval(vec, i, use_esf)

class ExprSlice(ExprUnaryOp):
    def __init__(self, arg, slice_):
        super(ExprSlice, self).__init__(arg)
        if not isinstance(slice_, slice):
            raise ValueError("slice_ must a slice object")
        if (not slice_.step is None) and (slice_.step != 1):
            raise ValueError("only slices with a step of 1 are supported!")
        self.idxes = list(range(*slice_.indices(self.arg.nbits)))

    @property
    def nbits(self):
        return len(self.idxes)

    def eval(self, vec, i, ctx, args, use_esf):
        return args[0].eval(vec, self.idxes[i], use_esf)

class ExprConcat(ExprNaryOp):
    @property
    def nbits(self):
        return sum((a.nbits for a in self.args))

    def eval(self, vec, i, ctx, args, use_esf):
        it = iter(args)
        cur_arg = next(it)
        cur_len = cur_arg.nbits
        org_i = i
        while i >= cur_len:
            i -= cur_len
            cur_arg = next(it)
            cur_len = cur_arg.nbits
        if i < 0:
            print(org_i, [a.nbits for a in args])
            assert(i>=0)
        return cur_arg.eval(vec, i, use_esf)

class ExprBroadcast(ExprUnaryOp):
    def __init__(self, arg, idx, nbits):
        super(ExprBroadcast, self).__init__(arg)
        assert(idx >= 0)
        self.idx = idx
        self._nbits = nbits

    def init_ctx(self):
        return CtxUninitialized

    @property
    def nbits(self):
        return self._nbits

    def eval(self, vec, i, ctx, args, use_esf):
        ret = ctx.get()
        if ret is CtxUninitialized:
            ret = args[0].eval(vec, self.idx, use_esf)
            ctx.set(ret)
        return ret

# Arithmetic ops
class ExprBinopCarry(ExprBinaryOp):
    class CtxCache:
        def __init__(self, nbits):
            self.cache = [CtxUninitialized]*nbits
            self.last_bit = -1
            self.carry = imm(0)

    def init_ctx(self):
        return ExprBinopCarry.CtxCache(self.nbits)

    def eval(self, vec, i, ctx, args, use_esf):
        CC = ctx.get()
        ret = CC.cache[i]
        if not ret is CtxUninitialized:
            return ret
        if i < CC.last_bit:
            raise ValueError("asking for a bit before the last computed bit. This should not happen!")
        X,Y = args
        for j in range(CC.last_bit+1, i+1):
            a = X.eval(vec, j, use_esf)
            b = Y.eval(vec, j, use_esf)
            CC.cache[j] = self.compute_binop_(vec, j, a, b, CC, use_esf)
        CC.last_bit = i
        return CC.cache[i]

    @staticmethod
    def compute_binop_(vec, i, X, Y, ctx, use_esf):
        raise NotImplementedError()

class ExprAdd(ExprBinopCarry):
    @staticmethod
    def compute_binop_(vec, i, X, Y, CC, use_esf):
        carry = CC.carry
        
        sum_args = simplify_inplace(X+Y)
        ret = simplify_inplace(sum_args + carry)
        # TODO: optimize this like in mba_if
        carry = esf(2, [X, Y, carry])
        if not use_esf:
            expand_esf_inplace(carry)
            simplify_inplace(carry)

        CC.carry = carry
        return ret

class ExprSub(ExprBinopCarry):
    @staticmethod
    def compute_binop_(vec, i, X, Y, CC, use_esf):
        carry = CC.carry
        
        sum_args = simplify_inplace(X+Y)
        ret = simplify_inplace(sum_args + carry)
        carry = esf(2, [X+imm(1), Y, carry])
        if not use_esf:
            expand_esf_inplace(carry)
            carry = simplify_inplace(carry)

        CC.carry = carry
        return ret

class ExprInner(object):
    def __init__(self, e):
        self.inner_expr = e

    def eval(self, vec, i, ctx, args, use_esf):
        return ctx.eval(vec, i, use_esf)

# x*y = x*(y0+y1<<1+y2<<2+...)
class ExprMul(ExprInner, ExprBinaryOp):
    def __init__(self, X, Y):
        ExprBinaryOp.__init__(self,X,Y)
        nbits = X.nbits
        e = ExprAnd(X, ExprBroadcast(Y, 0, nbits))
        for i in range(1, nbits):
            e = ExprAdd(
                e,
                ExprAnd(ExprShl(X, i), ExprBroadcast(Y, i, nbits)))
        ExprInner.__init__(self,e)

# Generic visitors
def visit_dfs(e, cb):
    if e.args != None:
        for a in e.args:
            visit_dfs(a, cb)
    cb(e)

# Evaluator
class ExprWithCtx(object):
    def __init__(self, e, ctx):
        self.e = e
        self.ctx = ctx
        self.args = None

    @property
    def nbits(self):
        return self.e.nbits

    def eval(self, vec, i, use_esf):
        return self.e.eval(vec, i, self.ctx, self.args, use_esf)

class CtxWrapper:
    def __init__(self, v):
        self.__v = v
    def get(self): return self.__v
    def set(self, v): self.__v = v

class _CtxUninitialized:
    pass
CtxUninitialized = _CtxUninitialized()

def init_ctxes(E):
    all_ctxs = dict()
    def init_ctx(e_):
        ectx = all_ctxs.get(id(e_), None)
        if not ectx is None:
            return ectx
        if isinstance(e_, ExprInner):
            einn = e_.inner_expr
            ectx = init_ctx(einn)
        else:
            ectx = ExprWithCtx(e_, CtxWrapper(e_.init_ctx()))
            args = e_.args
            if not args is None:
                ectx.args = [init_ctx(a) for a in args]
        all_ctxs[id(e_)] = ectx
        return ectx
    return init_ctx(E)

def eval_expr(e,use_esf=False):
    ectx = init_ctxes(e)

    ret = Vector(e.nbits)
    for i in range(e.nbits):
        ret[i] = ectx.eval(ret, i, use_esf)
    mba = MBA(len(ret))
    return mba.from_vec(ret)
