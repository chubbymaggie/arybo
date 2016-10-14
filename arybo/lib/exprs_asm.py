# Assemble Arybo IR into ASM thanks to LLVM
# Map symbol names to register
# Warning: this tries to do its best to save modified temporary registers.
# There might be errors while doing this. The idea is not to regenerate clean
# binaries, but to help the reverser!

#try:
import llvmlite.ir as ll
import llvmlite.binding as llvm
import ctypes
llvmlite_available = True
__llvm_initialized = False
#except ImportError:
#    llvmlite_available = False

import arybo.lib.mba_exprs as EX
import six

class ToLLVMIr(object):
    def __init__(self, sym_to_value, IRB):
        self.IRB = IRB
        self.sym_to_value = sym_to_value

    def rol(self, arg, n):
        pass

    def ror(self, arg, n):
        pass

    def visit_Cst(self, e):
        return ll.Constant(ll.IntType(e.nbits), e.n)

    def visit_BV(self, e):
        name = e.v.name
        value = self.sym_to_value.get(name, None)
        if value is None:
            raise ValueError("unable to map BV name '%s' to an LLVM value!" % name)
        ret,nbits = value
        if e.nbits != nbits:
            raise ValueError("bit-vector is %d bits, expected %d bits (size of a register)" % (e.nbits, self.reg_size))
        return ret

    def visit_Not(self, e):
        return self.IRB.not_(EX.visit(e, self))

    def visit_UnaryOp(self, e):
        ops = {
            EX.ExprShl: self.IRB.shl,
            EX.ExprLShr: self.IRB.lshr,
            EX.ExprRol: self.rol,
            EX.ExprRor: self.ror,
        }
        op = ops[type(e)]
        return op(EX.visit(e.arg, self), ll.Constant(ll.IntType(32), e.n))

    def visit_ZX(self, e):
        return self.IRB.zext(EX.visit(e.arg, self), ll.IntType(e.n))

    def visit_SX(self, e):
        return self.IRB.sext(EX.visit(e.arg, self), ll.IntType(e.n))

    def visit_Concat(self, e):
        # Generate a suite of OR + shifts
        arg0 = e.args[0]
        ret = EX.visit(arg0, self)
        ret = self.zext(ret, e.nbits)
        cur_bits = arg0.nbits
        for a in e.args[1:]:
            cur_arg = EX.visit(a, self)
            ret = self.IRB.or_(ret,
                self.IRB.shl(cur_arg, ll.Constant(ll.IntType(32), cur_bits)))
            cur_bits += a.nbits
        return ret

    def visit_Slice(self, e):
        ret = EX.visit(e.arg, self)
        idxes = e.idxes
        # Support only sorted indxes for now
        if idxes != list(range(idxes[0], idxes[-1]+1)):
            raise ValueError("slice indexes must be continuous and sorted")
        if idxes[0] != 0:
            ret = self.IRB.lshr(ret, ll.Constant(ll.IntType(32), idxes[0]))
        return self.IRB.trunc(ret, ll.IntType(len(idxes)))

    def visit_Broadcast(self, e):
        # left-shift to get the idx as the MSB, and them use an arithmetic
        # right shift of nbits-1
        nbits = e.arg.nbits
        ret = EX.visit(e.arg, self)
        ret = self.IRB.shl(ret, nbits-e.idx-1)
        return self.IRB.ashr(ret, nbits-1)

    def visit_nary_args(self, e, op):
        return op(*(EX.visit(a, self) for a in e.args))

    def visit_BinaryOp(self, e):
        ops = {
            EX.ExprAdd: self.IRB.add,
            EX.ExprSub: self.IRB.sub,
            EX.ExprMul: self.IRB.mul,
            EX.ExprDiv: self.IRB.div
        }
        op = ops[type(e)]
        return self.visit_nary_args(e, op)

    def visit_NaryOp(self, e):
        ops = {
            EX.ExprXor: self.IRB.xor,
            EX.ExprAnd: self.IRB.and_,
            EX.ExprOr: self.IRB.or_,
        }
        op = ops[type(e)]
        return self.visit_nary_args(e, op)
    
def _get_target(triple_or_target):
    global __llvm_initialized
    if not __llvm_initialized:
        # Lazy initialisation
        llvm.initialize()
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
        __llvm_initialized = True

    if isinstance(triple_or_target, llvm.Target):
        return triple_or_target
    if triple_or_target is None:
        return llvm.Target.from_default_triple()
    return llvm.Target.from_triple(triple_or_target)

def _create_execution_engine(M, target):
    target_machine = target.create_target_machine()
    engine = llvm.create_mcjit_compiler(M, target_machine)
    return engine

def to_llvm_ir(expr, sym_to_value, IRB):
    if not llvmlite_available:
        raise RuntimeError("llvmlite module unavailable! can't assemble to LLVM IR...")

    visitor = ToLLVMIr(sym_to_value, IRB)
    return EX.visit(expr, visitor)

def asm_module(expr, dst_reg, sym_to_reg, triple_or_target=None):
    if not llvmlite_available:
        raise RuntimeError("llvmlite module unavailable! can't assemble...")

    target = _get_target(triple_or_target)

    M = ll.Module()
    fntype = ll.FunctionType(ll.VoidType(), [])
    func = ll.Function(M, fntype, name='__arybo')
    func.attributes.add("naked")
    func.attributes.add("nounwind")
    BB = func.append_basic_block()

    IRB = ll.IRBuilder()
    IRB.position_at_end(BB)

    sym_to_value = {sym: (IRB.load_reg(reg[1], reg[0], reg[0]),reg[1]) for sym,reg in six.iteritems(sym_to_reg)}

    ret = to_llvm_ir(expr, sym_to_value, IRB)
    IRB.store_reg(ret, dst_reg[1], dst_reg[0])
    # See https://llvm.org/bugs/show_bug.cgi?id=15806
    IRB.unreachable()

    return M

def asm_binary(expr, dst_reg, sym_to_reg, triple_or_target=None):
    if not llvmlite_available:
        raise RuntimeError("llvmlite module unavailable! can't assemble...")

    target = _get_target(triple_or_target)
    M = asm_module(expr, dst_reg, sym_to_reg, target)

    # Use LLVM to compile the '__arybo' function. As the function is naked and
    # is the only, we just got to dump the .text section to get the binary
    # assembly.
    # No need for keystone or whatever hype stuff. llvmlite does the job.

    M = llvm.parse_assembly(str(M))
    M.verify()
    target_machine = target.create_target_machine()
    obj_bin = target_machine.emit_object(M)
    open("/tmp/a.o","wb").write(obj_bin)
    obj = llvm.ObjectFileRef.from_data(obj_bin)
    for s in obj.sections():
        if s.is_text():
            return s.data()
    raise RuntimeError("unable to get the assembled binary!")
