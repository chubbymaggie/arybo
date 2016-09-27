import unittest
import operator
import functools

import arybo.lib.mba_exprs as EX
from arybo.lib import MBA

from pytanque import expand_esf_inplace, simplify_inplace

class MBAExprsTest:
    def setUp(self):
        self.mba4 = MBA(4)
        self.mba4.use_esf = False
        self.mba8 = MBA(8)
        self.mba8.use_esf = False

        self.x4 = self.mba4.var('x')
        self.y4 = self.mba4.var('y')
        self.z4 = self.mba4.var('z')
        self.x4_expr = EX.ExprVar(self.x4)
        self.y4_expr = EX.ExprVar(self.y4)
        self.z4_expr = EX.ExprVar(self.z4)

        self.x8 = self.mba8.var('x')
        self.y8 = self.mba8.var('y')
        self.z8 = self.mba8.var('z')

        self.x8_expr = EX.ExprVar(self.x8)
        self.y8_expr = EX.ExprVar(self.y8)
        self.z8_expr = EX.ExprVar(self.z8)

    def exprEqual(self, expr, e):
        expr = EX.eval_expr(expr, self.use_esf)
        if self.use_esf:
            expand_esf_inplace(expr.vec)
            simplify_inplace(expr.vec)
        simplify_inplace(e.vec)
        simplify_inplace(expr.vec)
        if expr != e:
            print(expr)
            print(e)
        self.assertEqual(expr, e)

    def test_leaves(self):
        self.exprEqual(self.x4_expr, self.x4)
        self.exprEqual(EX.ExprCst(0xff, 4), self.mba4.from_cst(0xff))

    def test_unary(self):
        self.exprEqual(
            EX.ExprNot(self.x4_expr),
            ~self.x4)

        self.exprEqual(
            EX.ExprBroadcast(EX.ExprCst(1, 4), 0, 4),
            self.mba4.from_cst(0xf))

        self.exprEqual(
            EX.ExprBroadcast(EX.ExprCst(1, 4), 1, 4),
            self.mba4.from_cst(0))

    def test_unaryops(self):
        ops = (
            (EX.ExprXor, operator.xor),
            (EX.ExprAnd, operator.and_),
            (EX.ExprOr,  operator.or_),
            (EX.ExprAdd, operator.add),
            (EX.ExprSub, operator.sub),
            (EX.ExprMul, operator.mul),
        )

        args_expr = (self.x4_expr, self.y4_expr, self.z4_expr)
        args = (self.x4, self.y4, self.z4)
        args_expr = args_expr[2:]
        args = args[2:]

        t0 = EX.ExprMul(self.x4_expr, self.y4_expr)
        t1 = EX.ExprMul(t0, self.z4_expr)
        #t1 = EX.ExprMul(EX.ExprVar(EX.eval_expr(t0)), self.z4_expr)
        print(t1.nbits)
        print(EX.eval_expr(t1))
        print(self.x4 * self.y4 * self.z4)

        for op in ops:
            self.exprEqual(
                functools.reduce(op[0], args_expr),
                functools.reduce(op[1], args))

    def test_logical(self):
        ops = (
            (EX.ExprLShr, operator.rshift),
            (EX.ExprShl, operator.lshift),
            (EX.ExprRol, lambda x,n: x.rol(n)),
            (EX.ExprRor, lambda x,n: x.ror(n))
        )

        for op in ops:
            for s in range(5):
                self.exprEqual(
                    op[0](self.x4_expr, s),
                    op[1](self.x4, s))

    def test_zx_sx(self):
        self.exprEqual(
            EX.ExprZX(EX.ExprCst(0xf, 4), 8),
            self.mba8.from_cst(0x0f))

        self.exprEqual(
            EX.ExprSX(EX.ExprCst(0x8, 4), 8),
            self.mba8.from_cst(0xf8))

    def test_extract_contract(self):
        self.exprEqual(
            EX.ExprSlice(self.x8_expr, slice(0, 4)),
            self.x4)

        self.exprEqual(
            EX.ExprConcat(EX.ExprCst(0xf, 4), EX.ExprCst(0, 4)),
            self.mba8.from_cst(0x0f))

class MBAExprsTestNoEsf(MBAExprsTest, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.use_esf = False

#class MBAExprsTestEsf(MBAExprsTest, unittest.TestCase):
#    def __init__(self, *args, **kwargs):
#        unittest.TestCase.__init__(self, *args, **kwargs)
#        self.use_esf = True
