from __future__ import annotations
import math, struct
from dataclasses import dataclass, replace
from typing import Callable, Sequence
from tinygrad.helpers import to_mv

MASK16, MASK32, MASK64 = 0xffff, 0xffffffff, 0xffffffffffffffff
TYPE_F16, TYPE_F32, TYPE_U16, TYPE_U32, TYPE_S16, TYPE_S32, TYPE_U8, TYPE_U8_32 = range(8)
TYPE_NAMES = ("f16", "f32", "u16", "u32", "s16", "s32", "u8", "u8_32")
HALF_TYPES = {TYPE_F16, TYPE_U16, TYPE_S16, TYPE_U8}
FLOAT_LUT = (0.0, 0.5, 1.0, 2.0, math.e, math.pi, 1 / math.pi, 1 / math.log2(math.e), math.log2(math.e),
             1 / math.log2(10), math.log2(10), 4.0)

def _bits(x: int, lo: int, hi: int) -> int: return (x >> lo) & ((1 << (hi-lo+1)) - 1)
def _sext(x: int, bits: int) -> int: return x - (1 << bits) if x & (1 << (bits-1)) else x
def _u32(x: int) -> int: return x & MASK32
def _s32(x: int) -> int: return _sext(x & MASK32, 32)
def _f32(x: int) -> float: return struct.unpack("<f", struct.pack("<I", x & MASK32))[0]
def _f32bits(x: float) -> int: return struct.unpack("<I", struct.pack("<f", x))[0]
def _f16(x: int) -> float: return struct.unpack("<e", struct.pack("<H", x & MASK16))[0]
def _f16bits(x: float) -> int: return struct.unpack("<H", struct.pack("<e", x))[0]

def _sfu(op: str, x: float) -> float:
    if op == "rcp": return math.copysign(math.inf,x) if x == 0 else 1/x
    if op in ("rsq","hrsq"):
        if x == 0: return math.copysign(math.inf,x)
        return math.nan if x < 0 else 1/math.sqrt(x)
    if op in ("log2","hlog2"):
        if x == 0: return -math.inf
        return math.nan if x < 0 else math.log2(x)
    if op in ("exp2","hexp2"):
        try: return 2**x
        except OverflowError: return math.inf
    if op == "sqrt": return math.nan if x < 0 else math.sqrt(x)
    if not math.isfinite(x): return math.nan
    return math.sin(x) if op == "sin" else math.cos(x)

def _unary_float(op: str, x: float) -> float:
    if op == "sign.f": return math.copysign(1.0,x) if x else 0.0
    if op == "absneg.f" or not math.isfinite(x): return x
    if op == "floor.f": return float(math.floor(x))
    if op == "ceil.f": return float(math.ceil(x))
    if op == "rndne.f": return float(round(x))
    return float(math.trunc(x))

@dataclass(frozen=True)
class IR3Operand:
    kind: str
    num: int = 0
    imm: int = 0
    half: bool = False
    modifier: int = 0
    repeat: bool = False

    def __str__(self):
        if self.kind == "imm": return str(self.imm)
        if self.kind in ("relative_gpr", "relative_const"):
            pfx = "c" if self.kind == "relative_const" else "r"
            return f"{pfx}<a0.x{self.num:+d}>"
        if self.kind == "addr": return f"a0.{('x','y','z','w')[self.num]}"
        if self.kind == "pred": return f"p0.{('x','y','z','w')[self.num]}"
        pfx = ("hc" if self.half else "c") if self.kind == "const" else "hr" if self.half else "r"
        return f"{pfx}{self.num//4}.{('x','y','z','w')[self.num&3]}"

@dataclass(frozen=True)
class IR3Instruction:
    pc: int
    raw: int
    opcode: str
    dst: IR3Operand | None = None
    srcs: tuple[IR3Operand, ...] = ()
    condition: int = 0
    src_type: int = -1
    dst_type: int = -1
    repeat: int = 0
    nop_count: int = 0
    width: int = 1
    saturate: bool = False
    branch_offset: int = 0
    invert: tuple[bool, ...] = ()
    rounding: int = 0

def _reg(num: int, half: bool=False, modifier: int=0, repeat: bool=False) -> IR3Operand:
    if not half and 0xf4 <= num <= 0xf7: return IR3Operand("addr", num=num-0xf4, modifier=modifier, repeat=repeat)
    if not half and 0xf8 <= num <= 0xfb: return IR3Operand("pred", num=num-0xf8, modifier=modifier, repeat=repeat)
    return IR3Operand("gpr", num=num, half=half, modifier=modifier, repeat=repeat)

def _cat2_dst(num: int, half: bool=False) -> IR3Operand:
    if 0xf4 <= num <= 0xfb: return IR3Operand("pred",num=num&3)
    return IR3Operand("gpr",num=num,half=half)

def _const(num: int, half: bool=False, modifier: int=0, repeat: bool=False) -> IR3Operand:
    return IR3Operand("const", num=num, half=half, modifier=modifier, repeat=repeat)

def _imm(val: int, modifier: int=0) -> IR3Operand: return IR3Operand("imm", imm=val, modifier=modifier)
def _pred(comp: int) -> IR3Operand: return IR3Operand("pred", num=comp)

def _decode_multisrc(x: int, half: bool, repeat: bool=False) -> IR3Operand:
    modifier, mode = _bits(x, 14, 15), _bits(x, 11, 13)
    if mode == 0: return _reg(x & 0xff, half, modifier, repeat)
    if mode == 1:
        kind = "relative_const" if _bits(x, 10, 10) else "relative_gpr"
        return IR3Operand(kind, num=_sext(x & 0x3ff, 10), half=half, modifier=modifier, repeat=repeat)
    if mode in (2, 6): return _const(x & 0x7ff, half, modifier, repeat)
    if mode == 4: return _imm(_sext(x & 0x7ff, 11), modifier)
    if mode == 5:
        idx = x & 0x3ff
        if idx >= len(FLOAT_LUT): raise NotImplementedError(f"IR3 float lookup immediate {idx}")
        return _imm(_f16bits(FLOAT_LUT[idx]) if _bits(x,10,10) else _f32bits(FLOAT_LUT[idx]), modifier)
    raise NotImplementedError(f"IR3 multisrc encoding {mode} is not supported")

def _decode_cat3_src(x: int, half: bool, immed: bool, repeat: bool=False, modifier: int=0) -> IR3Operand:
    if _bits(x,11,12) == 0: return _reg(x & 0xff, half, modifier, repeat)
    if _bits(x,11,12) == 1:
        kind = "relative_const" if _bits(x,10,10) else "relative_gpr"
        return IR3Operand(kind, num=_sext(x & 0x3ff,10), half=half, modifier=modifier, repeat=repeat)
    if _bits(x,12,12): return _imm(x & 0xfff,modifier) if immed else _const(x & 0x7ff,half,modifier,repeat)
    raise NotImplementedError(f"IR3 cat3 source encoding {x:#x} is not supported")

def decode_instruction(raw: int, pc: int=0) -> IR3Instruction:
    cat = raw >> 61
    if cat == 0:
        opc, hi, repeat = _bits(raw,55,58), _bits(raw,49,49), _bits(raw,40,42)
        common = {0:"nop",4:"ret",6:"end"}
        if opc in common and not hi: return IR3Instruction(pc,raw,common[opc],repeat=repeat)
        if opc == 5 and not hi:
            return IR3Instruction(pc,raw,"kill",srcs=(_pred(_bits(raw,53,54)),),invert=(bool(_bits(raw,52,52)),))
        if opc in (13,14,15) and hi: return IR3Instruction(pc,raw,{13:"predt",14:"predf",15:"prede"}[opc])
        if opc in (2,3) and not hi:
            return IR3Instruction(pc,raw,"jump" if opc == 2 else "call",branch_offset=_sext(raw&MASK32,32))
        if opc == 1 and not hi:
            brtype = _bits(raw,37,39)
            names = {0:"br",1:"brao",2:"braa",3:"brac",4:"bany",5:"ball",6:"brax"}
            if brtype not in names: raise NotImplementedError(f"IR3 cat0 branch type {brtype} at pc {pc}")
            branch_srcs: tuple[IR3Operand, ...] = () if brtype in (3,6) else (_pred(_bits(raw,53,54)),)
            branch_inv: tuple[bool, ...] = () if brtype in (3,6) else (bool(_bits(raw,52,52)),)
            if brtype in (1,2):
                branch_srcs += (_pred(_bits(raw,46,47)),)
                branch_inv += (bool(_bits(raw,45,45)),)
            return IR3Instruction(pc,raw,names[brtype],srcs=branch_srcs,branch_offset=_sext(raw&MASK32,32),invert=branch_inv)
        raise NotImplementedError(f"IR3 cat0 opcode {(hi<<4)|opc:#x} at pc {pc}")
    if cat == 1:
        src_type, dst_type, mode = _bits(raw,50,52), _bits(raw,46,48), _bits(raw,53,54)
        half, src_repeat = src_type in HALF_TYPES, bool(_bits(raw,43,43))
        if mode == 0:
            if _bits(raw,11,11):
                kind = "relative_const" if _bits(raw,10,10) else "relative_gpr"
                src = IR3Operand(kind,num=_sext(raw&0x3ff,10),half=half,repeat=src_repeat)
            else:
                if _bits(raw,11,31): raise NotImplementedError("extended cat1 gpr source")
                src = _reg(raw&0xff,half,repeat=src_repeat)
        elif mode == 1:
            if _bits(raw,11,31): raise NotImplementedError("extended cat1 const source")
            src = _const(raw&0x7ff,half,repeat=src_repeat)
        elif mode == 2: src = _imm(raw&MASK32)
        else: raise NotImplementedError(f"IR3 cat1 source mode {mode}")
        dst_num = _bits(raw,32,39)
        dst = IR3Operand("relative_gpr",num=dst_num,half=dst_type in HALF_TYPES) if _bits(raw,49,49) else _reg(dst_num,dst_type in HALF_TYPES)
        op = "mov" if src_type == dst_type else "cov"
        return IR3Instruction(pc,raw,f"{op}.{TYPE_NAMES[src_type]}{TYPE_NAMES[dst_type]}",dst,(src,),src_type=src_type,dst_type=dst_type,
                              repeat=_bits(raw,40,41),saturate=bool(_bits(raw,42,42)),rounding=_bits(raw,55,56))
    if cat == 2:
        opcs = {0:"add.f",1:"min.f",2:"max.f",3:"mul.f",4:"sign.f",5:"cmps.f",6:"absneg.f",7:"cmpv.f",
                9:"floor.f",10:"ceil.f",11:"rndne.f",12:"rndaz.f",13:"trunc.f",16:"add.u",17:"add.s",18:"sub.u",
                19:"sub.s",20:"cmps.u",21:"cmps.s",22:"min.u",23:"min.s",24:"max.u",25:"max.s",26:"absneg.s",
                28:"and.b",29:"or.b",30:"not.b",31:"xor.b",33:"cmpv.u",34:"cmpv.s",
                35:"mul.f.mul2",36:"add.f.mul2",37:"mul.f.div2",38:"add.f.div2",48:"mul.u24",49:"mul.s24",
                50:"mull.u",51:"bfrev.b",52:"clz.s",53:"clz.b",54:"shl.b",55:"shr.b",56:"ashr.b",58:"mgen.b",
                59:"getbit.b",61:"cbits.b"}
        unary = {4,6,9,10,11,12,13,26,30,51,52,53,61}
        opc = _bits(raw,53,58)
        if opc not in opcs: raise NotImplementedError(f"IR3 cat2 opcode {opc:#x} at pc {pc}")
        full, dst_conv = bool(_bits(raw,52,52)), bool(_bits(raw,46,46))
        dst_num, repeat = _bits(raw,32,39), _bits(raw,40,41)
        dst_half = full == dst_conv and dst_num <= 0xf7
        src1_r, src2_r = bool(_bits(raw,43,43)), bool(_bits(raw,51,51))
        nop_count = (int(src1_r)|(int(src2_r)<<1)) if repeat == 0 else 0
        srcs: tuple[IR3Operand, ...] = (_decode_multisrc(_bits(raw,0,15),not full,src1_r if repeat else False),)
        if opc not in unary: srcs += (_decode_multisrc(_bits(raw,16,31),not full,src2_r if repeat else False),)
        return IR3Instruction(pc,raw,opcs[opc],_cat2_dst(dst_num,dst_half),srcs,condition=_bits(raw,48,50),repeat=repeat,
                              nop_count=nop_count,saturate=bool(_bits(raw,42,42)))
    if cat == 3:
        regular = {0:"mad.u16",1:"madsh.u16",2:"mad.s16",3:"madsh.m16",4:"mad.u24",5:"mad.s24",6:"mad.f16",
                   7:"mad.f32",8:"sel.b16",9:"sel.b32",10:"sel.s16",11:"sel.s32",12:"sel.f16",13:"sel.f32",
                   14:"sad.s16",15:"sad.s32"}
        alternate = {8:"shrm",9:"shlm",10:"shrg",11:"shlg",12:"andg"}
        opc, al_op = _bits(raw,55,58), bool(_bits(raw,13,13))
        scaled = {4:"mad.f16.mul2",5:"mad.f32.mul2",6:"mad.f16.div2",7:"mad.f32.div2"}
        alt = al_op and opc >= 8
        table = alternate if alt else scaled if al_op else regular
        if opc not in table: raise NotImplementedError(f"IR3 cat3 opcode {opc:#x} at pc {pc}")
        full = bool(_bits(raw,42,42)) if alt else opc in {5,7} if al_op else opc in {1,3,4,5,7,9,11,13,15}
        dst_num, dst_conv, repeat = _bits(raw,32,39), bool(_bits(raw,46,46)), _bits(raw,40,41)
        dst_half = full == dst_conv and dst_num <= 0xf7
        repeats = (bool(_bits(raw,43,43)),bool(_bits(raw,15,15)),bool(_bits(raw,29,29)))
        src1 = _decode_cat3_src(_bits(raw,0,12),not full,alt,repeats[0] if repeat else False,_bits(raw,14,14))
        src2 = _reg(_bits(raw,47,54),not full,_bits(raw,30,30),repeats[1] if repeat else False)
        src3 = _decode_cat3_src(_bits(raw,16,28),not full,alt,repeats[2] if repeat else False,_bits(raw,31,31))
        nop_count = (int(repeats[0])|(int(repeats[1])<<1)) if repeat == 0 else 0
        return IR3Instruction(pc,raw,table[opc],_reg(dst_num,dst_half),(src1,src2,src3),repeat=repeat,nop_count=nop_count,
                              saturate=bool(_bits(raw,42,42)) if not alt else False)
    if cat == 4:
        opcs = {0:"rcp",1:"rsq",2:"log2",3:"exp2",4:"sin",5:"cos",6:"sqrt",9:"hrsq",10:"hlog2",11:"hexp2"}
        opc = _bits(raw,53,58)
        if opc not in opcs: raise NotImplementedError(f"IR3 cat4 opcode {opc:#x} at pc {pc}")
        full, dst_conv, repeat = bool(_bits(raw,52,52)), bool(_bits(raw,46,46)), _bits(raw,40,41)
        dst_num = _bits(raw,32,39)
        src = _decode_multisrc(_bits(raw,0,15),not full,bool(_bits(raw,43,43)) if repeat else False)
        return IR3Instruction(pc,raw,opcs[opc],_reg(dst_num,full == dst_conv and dst_num <= 0xf7),(src,),repeat=repeat,
                              saturate=bool(_bits(raw,42,42)))
    if cat == 6:
        opc, typ = _bits(raw,54,58), _bits(raw,49,51)
        if opc == 0:
            size = _bits(raw,24,26)
            if size < 1: raise NotImplementedError(f"IR3 cat6 load size {size}")
            src, dst = _reg(_bits(raw,14,21)), _reg(_bits(raw,32,39),typ in HALF_TYPES)
            if _bits(raw,22,22):
                return IR3Instruction(pc,raw,f"ldg.a.{TYPE_NAMES[typ]}",dst,
                                      (src,_reg(_bits(raw,1,8)),_imm(_bits(raw,9,10)),_imm(_bits(raw,12,13))),
                                      src_type=typ,dst_type=typ,width=size)
            return IR3Instruction(pc,raw,f"ldg.{TYPE_NAMES[typ]}",dst,(src,_imm(_sext(_bits(raw,1,13),13))),
                                  src_type=typ,dst_type=typ,width=size)
        if opc in (1,2,10):
            size = _bits(raw,24,31)
            if size < 1: raise NotImplementedError(f"IR3 cat6 load size {size}")
            name = {1:"ldl",2:"ldp",10:"ldlw"}[opc]
            return IR3Instruction(pc,raw,f"{name}.{TYPE_NAMES[typ]}",_reg(_bits(raw,32,39),typ in HALF_TYPES),
                                  (_reg(_bits(raw,14,21)),_imm(_sext(_bits(raw,1,13),13))),src_type=typ,dst_type=typ,width=size)
        if opc == 3:
            size = _bits(raw,24,26)
            if size < 1: raise NotImplementedError(f"IR3 cat6 store size {size}")
            if _bits(raw,52,52):
                return IR3Instruction(pc,raw,f"stg.a.{TYPE_NAMES[typ]}",None,
                                      (_reg(_bits(raw,41,48)),_reg(_bits(raw,32,39)),_imm(_bits(raw,9,10)),
                                       _imm(_bits(raw,12,13)),_reg(_bits(raw,1,8),typ in HALF_TYPES)),
                                      src_type=typ,dst_type=typ,width=size)
            off = _sext((_bits(raw,9,13)<<8)|_bits(raw,32,39),13)
            addr, val = _reg(_bits(raw,41,48)), _reg(_bits(raw,1,8),typ in HALF_TYPES)
            return IR3Instruction(pc,raw,f"stg.{TYPE_NAMES[typ]}",None,(addr,_imm(off),val),src_type=typ,dst_type=typ,width=size)
        if opc in (4,5,11):
            size = _bits(raw,24,31)
            if size < 1: raise NotImplementedError(f"IR3 cat6 store size {size}")
            name = {4:"stl",5:"stp",11:"stlw"}[opc]
            off = _sext((_bits(raw,9,13)<<8)|_bits(raw,32,39),13)
            return IR3Instruction(pc,raw,f"{name}.{TYPE_NAMES[typ]}",None,
                                  (_reg(_bits(raw,41,48)),_imm(off),_reg(_bits(raw,1,8),typ in HALF_TYPES)),
                                  src_type=typ,dst_type=typ,width=size)
        if 16 <= opc <= 26:
            atomic_names = ("add","sub","xchg","inc","dec","cmpxchg","min","max","and","or","xor")
            if _bits(raw,9,10) != 0 or _bits(raw,12,13) != 0:
                raise NotImplementedError(f"IR3 cat6 vector atomic at pc {pc}")
            src1_num, src2_num = _bits(raw,14,21), _bits(raw,24,31)
            src1 = _imm(src1_num) if _bits(raw,22,22) else _reg(src1_num)
            src2 = _imm(src2_num) if _bits(raw,23,23) else _reg(src2_num)
            space = "global" if _bits(raw,52,52) else "local"
            return IR3Instruction(pc,raw,f"atomic.{space}.{atomic_names[opc-16]}",_reg(_bits(raw,32,39)),(src1,src2),
                                  src_type=typ,dst_type=typ)
        raise NotImplementedError(f"IR3 cat6 opcode {opc:#x} at pc {pc}")
    if cat == 7:
        opc = _bits(raw,55,58)
        opcs = {0:"bar",1:"fence",2:"sleep",3:"icinv",4:"dccln",5:"dcinv",6:"dcflu"}
        if opc not in opcs: raise NotImplementedError(f"IR3 cat7 opcode {opc:#x} at pc {pc}")
        return IR3Instruction(pc,raw,opcs[opc])
    raise NotImplementedError(f"IR3 category {cat} at pc {pc}")

def decode_program(code: bytes|bytearray|memoryview) -> tuple[IR3Instruction, ...]:
    data = memoryview(code).cast("B")
    if data.nbytes & 7: raise ValueError("IR3 code size must be a multiple of 8")
    ret = []
    for i in range(0,data.nbytes,8):
        ret.append(decode_instruction(int.from_bytes(data[i:i+8],"little"),i//8))
        if ret[-1].opcode == "end": break
    return tuple(ret)

def _typed_value(x: int, typ: int):
    if typ == TYPE_F16: return _f16(x)
    if typ == TYPE_F32: return _f32(x)
    if typ == TYPE_U16: return x&MASK16
    if typ == TYPE_U32: return x&MASK32
    if typ == TYPE_S16: return _sext(x&MASK16,16)
    if typ == TYPE_S32: return _s32(x)
    if typ == TYPE_U8: return x&0xff
    if typ == TYPE_U8_32: return x&0xff
    raise ValueError(typ)

def _typed_bits(x, typ: int) -> int:
    if typ == TYPE_F16: return _f16bits(float(x))
    if typ == TYPE_F32: return _f32bits(float(x))
    bits = 16 if typ in (TYPE_U16,TYPE_S16) else 8 if typ == TYPE_U8 else 32
    return int(x)&((1<<bits)-1)

def _type_size(typ: int) -> int: return 2 if typ in (TYPE_F16,TYPE_U16,TYPE_S16) else 1 if typ in (TYPE_U8,TYPE_U8_32) else 4

class IR3Machine:
    def __init__(self, program: Sequence[IR3Instruction], constants: Sequence[int]|bytes|bytearray|memoryview=(),
                 translate_addr: Callable[[int],int]|None=None, shared_memory: bytearray|memoryview|None=None,
                 private_size: int=0):
        if private_size < 0: raise ValueError("IR3 private memory size must be nonnegative")
        self.program, self.regs, self.hregs = tuple(program), [0]*256, [0]*256
        self.addr, self.pred = [0]*4, [0]*4
        if isinstance(constants,(bytes,bytearray,memoryview)):
            mv = memoryview(constants).cast("B")
            if mv.nbytes&3: raise ValueError("IR3 constant data must be 4-byte aligned")
            self.constants = list(mv.cast("I"))
        else: self.constants = [_u32(x) for x in constants]
        self.translate_addr = translate_addr or (lambda x:x)
        self.shared = memoryview(shared_memory if shared_memory is not None else bytearray()).cast("B")
        self.private = memoryview(bytearray(private_size)).cast("B")
        self.pc, self.done, self.waiting, self.predication, self.predication_mask = 0, False, False, 0, False
        self.call_stack: list[int] = []

    def _offset(self, op: IR3Operand|None, amount: int) -> IR3Operand|None:
        if op is None or amount == 0: return op
        if op.kind in ("gpr","const"): return replace(op,num=op.num+amount)
        if op.kind in ("addr","pred"):
            num = op.num+amount
            if num >= 4: raise RuntimeError(f"special register repeat out of bounds: {op}")
            return replace(op,num=num)
        return op

    def _read(self, op: IR3Operand) -> int:
        if op.kind == "imm": return _u32(op.imm)
        if op.kind == "const":
            if op.num >= len(self.constants): raise RuntimeError(f"constant {op} out of bounds")
            return self.constants[op.num]&(MASK16 if op.half else MASK32)
        if op.kind == "gpr": return (self.hregs if op.half else self.regs)[op.num]&(MASK16 if op.half else MASK32)
        if op.kind == "addr": return self.addr[op.num]&MASK32
        if op.kind == "pred": return self.pred[op.num]&MASK32
        if op.kind in ("relative_gpr","relative_const"):
            num = self.addr[0]+op.num
            return self._read(IR3Operand("const" if op.kind == "relative_const" else "gpr",num=num,half=op.half))
        raise ValueError(op.kind)

    def _write(self, op: IR3Operand|None, val: int):
        if op is None: raise RuntimeError("cannot write missing operand")
        mask = MASK16 if op.half else MASK32
        if op.kind == "gpr": (self.hregs if op.half else self.regs).__setitem__(op.num,val&mask)
        elif op.kind == "addr": self.addr[op.num] = val&MASK32
        elif op.kind == "pred": self.pred[op.num] = val&MASK32
        elif op.kind == "relative_gpr": self._write(IR3Operand("gpr",num=self.addr[0]+op.num,half=op.half),val)
        else: raise RuntimeError(f"cannot write {op.kind}")

    def _source(self, inst: IR3Instruction, idx: int, iteration: int=0) -> int:
        op = inst.srcs[idx]
        offset_op = self._offset(op,iteration if op.repeat else 0)
        assert offset_op is not None
        val = self._read(offset_op)
        if not op.modifier: return val
        if ".f" in inst.opcode or inst.opcode in {"rcp","rsq","log2","exp2","sin","cos","sqrt","hrsq","hlog2","hexp2"}:
            fval = _f16(val) if op.half else _f32(val)
            if op.modifier&2: fval = abs(fval)
            if op.modifier&1: fval = -fval
            return _f16bits(fval) if op.half else _f32bits(fval)
        if ".s" in inst.opcode:
            sval = _sext(val&(MASK16 if op.half else MASK32),16 if op.half else 32)
            if op.modifier&2: sval = abs(sval)
            if op.modifier&1: sval = -sval
            return sval&(MASK16 if op.half else MASK32)
        return ~val&(MASK16 if op.half else MASK32) if op.modifier&1 else val

    def _addr64(self, op: IR3Operand, off: int=0) -> int:
        if op.kind != "gpr" or op.half or op.num == 255: raise RuntimeError(f"invalid 64-bit address reg {op}")
        return ((((self.regs[op.num+1]&MASK32)<<32)|(self.regs[op.num]&MASK32))+off)&MASK64

    def _load_global(self, addr: int, size: int) -> int:
        return int.from_bytes(to_mv(self.translate_addr(addr&MASK64),size).cast("B"),"little")

    def _store_global(self, addr: int, val: int, size: int):
        to_mv(self.translate_addr(addr&MASK64),size).cast("B")[:] = (val&((1<<(size*8))-1)).to_bytes(size,"little")

    @staticmethod
    def _load_local(memory: memoryview, addr: int, size: int, space: str) -> int:
        if addr < 0 or addr+size > memory.nbytes: raise RuntimeError(f"IR3 {space} load out of bounds: {addr:#x}+{size:#x}")
        return int.from_bytes(memory[addr:addr+size],"little")

    @staticmethod
    def _store_local(memory: memoryview, addr: int, val: int, size: int, space: str):
        if addr < 0 or addr+size > memory.nbytes: raise RuntimeError(f"IR3 {space} store out of bounds: {addr:#x}+{size:#x}")
        memory[addr:addr+size] = (val&((1<<(size*8))-1)).to_bytes(size,"little")

    def _read_scalar(self, op: IR3Operand, size: int) -> int:
        if size <= 4: return self._read(op)&((1<<(size*8))-1)
        if size != 8 or op.kind != "gpr" or op.half or op.num == 255: raise RuntimeError(f"invalid {size}-byte source {op}")
        return ((self.regs[op.num+1]&MASK32)<<32)|(self.regs[op.num]&MASK32)

    def _write_scalar(self, op: IR3Operand|None, value: int, size: int):
        if size <= 4: self._write(op,value)
        elif size == 8 and op is not None and op.kind == "gpr" and not op.half and op.num != 255:
            self.regs[op.num], self.regs[op.num+1] = value&MASK32, (value>>32)&MASK32
        else: raise RuntimeError(f"invalid {size}-byte destination {op}")

    def _atomic(self, inst: IR3Instruction):
        typ, op = inst.src_type, inst.opcode.rsplit(".",1)[1]
        global_space = ".global." in inst.opcode
        size = 8 if global_space and typ == 6 else _type_size(typ)
        if global_space:
            addr = self._addr64(inst.srcs[0])
            old = self._load_global(addr,size)
        else:
            addr = self._read(inst.srcs[0])
            old = self._load_local(self.shared,addr,size,"shared")
        value = self._read_scalar(inst.srcs[1],size)
        if typ in (TYPE_F16,TYPE_F32):
            old_value, value_value = _typed_value(old,typ), _typed_value(value,typ)
        elif typ in (TYPE_S16,TYPE_S32):
            old_value, value_value = _typed_value(old,typ), _typed_value(value,typ)
        else: old_value, value_value = old, value
        if op == "add": result = old_value+value_value
        elif op == "sub": result = old_value-value_value
        elif op == "xchg": result = value
        elif op == "inc": result = old+1
        elif op == "dec": result = old-1
        elif op == "min": result = min(old_value,value_value)
        elif op == "max": result = max(old_value,value_value)
        elif op == "and": result = old&value
        elif op == "or": result = old|value
        elif op == "xor": result = old^value
        elif op == "cmpxchg":
            if inst.srcs[1].kind != "gpr": raise RuntimeError("IR3 cmpxchg requires a register collect")
            replacement = self._read_scalar(replace(inst.srcs[1],num=inst.srcs[1].num+max(1,size//4)),size)
            result = replacement if old == value else old
        else: raise NotImplementedError(inst.opcode)
        result_bits = _typed_bits(result,typ) if typ in (TYPE_F16,TYPE_F32) and op in ("add","sub","min","max") else int(result)
        if global_space: self._store_global(addr,result_bits,size)
        else: self._store_local(self.shared,addr,result_bits,size,"shared")
        self._write_scalar(inst.dst,old,size)

    @staticmethod
    def _cond(condition: int, a, b) -> bool:
        conds = (a<b,a<=b,a>b,a>=b,a==b,a!=b)
        if condition >= len(conds): raise NotImplementedError(f"IR3 compare condition {condition}")
        return conds[condition]

    def _float_bits(self, value: float, half: bool, saturate: bool=False) -> int:
        if saturate: value = min(1.0,max(0.0,value))
        return _f16bits(value) if half else _f32bits(value)

    def _execute_alu(self, inst: IR3Instruction, iteration: int):
        src = [self._source(inst,i,iteration) for i in range(len(inst.srcs))]
        half = inst.srcs[0].half
        dst = self._offset(inst.dst,iteration)
        floats = [_f16(x) if half else _f32(x) for x in src]
        signed = [_sext(x&(MASK16 if half else MASK32),16 if half else 32) for x in src]
        op = inst.opcode
        if op.startswith(("mov.","cov.")):
            val = _typed_value(src[0],inst.src_type)
            if isinstance(val,float) and inst.dst_type not in (TYPE_F16,TYPE_F32):
                val = (round(val) if inst.rounding == 1 else math.ceil(val) if inst.rounding == 2 else
                       math.floor(val) if inst.rounding == 3 else int(val))
            if inst.saturate and inst.dst_type in (TYPE_F16,TYPE_F32): val = min(1.0,max(0.0,val))
            self._write(dst,_typed_bits(val,inst.dst_type))
        elif op in ("add.f","mul.f","min.f","max.f","mul.f.mul2","add.f.mul2","mul.f.div2","add.f.div2"):
            val = {"add.f":floats[0]+floats[1],"mul.f":floats[0]*floats[1],"min.f":min(floats),"max.f":max(floats),
                   "mul.f.mul2":floats[0]*floats[1]*2,"add.f.mul2":(floats[0]+floats[1])*2,
                   "mul.f.div2":floats[0]*floats[1]*0.5,"add.f.div2":(floats[0]+floats[1])*0.5}[op]
            self._write(dst,self._float_bits(val,dst.half if dst else half,inst.saturate))
        elif op in ("sign.f","absneg.f","floor.f","ceil.f","rndne.f","rndaz.f","trunc.f"):
            self._write(dst,self._float_bits(_unary_float(op,floats[0]),dst.half if dst else half,inst.saturate))
        elif op.startswith(("cmps.","cmpv.")):
            vals = floats if op.endswith(".f") else signed if op.endswith(".s") else src
            self._write(dst,int(self._cond(inst.condition,*vals[:2])))
        elif op in ("add.u","add.s","sub.u","sub.s"): self._write(dst,src[0]+src[1] if op.startswith("add") else src[0]-src[1])
        elif op in ("min.u","max.u","min.s","max.s"):
            vals = signed if op.endswith(".s") else src
            self._write(dst,min(vals) if op.startswith("min") else max(vals))
        elif op == "absneg.s": self._write(dst,signed[0])
        elif op in ("and.b","or.b","xor.b"): self._write(dst,{"and.b":src[0]&src[1],"or.b":src[0]|src[1],"xor.b":src[0]^src[1]}[op])
        elif op == "not.b": self._write(dst,~src[0])
        elif op in ("mul.u24","mul.s24","mull.u"):
            if op == "mul.s24": a,b = _sext(src[0]&0xffffff,24),_sext(src[1]&0xffffff,24)
            elif op == "mul.u24": a,b = src[0]&0xffffff,src[1]&0xffffff
            else: a,b = src[0],src[1]
            self._write(dst,a*b)
        elif op == "bfrev.b": self._write(dst,int(f"{src[0]&MASK32:032b}"[::-1],2))
        elif op in ("clz.s","clz.b"): self._write(dst,32-(src[0]&MASK32).bit_length())
        elif op == "cbits.b": self._write(dst,(src[0]&MASK32).bit_count())
        elif op == "shl.b": self._write(dst,src[0]<<(src[1]&31))
        elif op == "shr.b": self._write(dst,src[0]>>(src[1]&31))
        elif op == "ashr.b": self._write(dst,signed[0]>>(src[1]&31))
        elif op == "getbit.b": self._write(dst,(src[0]>>(src[1]&31))&1)
        elif op == "mgen.b": self._write(dst,((1<<min(src[0],32))-1)<<(src[1]&31))
        elif op.startswith("mad."):
            val = floats[0]*floats[1]+floats[2] if ".f" in op else src[0]*src[1]+src[2]
            if op.endswith(".mul2"): val *= 2
            if op.endswith(".div2"): val *= 0.5
            self._write(dst,self._float_bits(val,dst.half,inst.saturate) if ".f" in op and dst is not None else val)
        elif op.startswith("madsh."): self._write(dst,((src[0]*src[1])>>16)+src[2])
        elif op.startswith("sel."): self._write(dst,src[0] if src[1] else src[2])
        elif op.startswith("sad."): self._write(dst,abs(signed[0]-signed[1])+src[2])
        elif op in ("shrm","shlm","shrg","shlg","andg"):
            grouped = {"shrm":(src[1]>>(src[0]&31))&src[2],"shlm":(src[1]<<(src[0]&31))&src[2],"shrg":(src[1]>>(src[0]&31))|src[2],
                       "shlg":(src[1]<<(src[0]&31))|src[2],"andg":(src[1]&src[0])|src[2]}
            self._write(dst,grouped[op])
        elif op in {"rcp","rsq","log2","exp2","sin","cos","sqrt","hrsq","hlog2","hexp2"}:
            self._write(dst,self._float_bits(_sfu(op,floats[0]),dst.half if dst else half,inst.saturate))
        else: raise NotImplementedError(f"IR3 instruction {op} at pc {inst.pc}")

    def step(self):
        if self.done or self.waiting: return
        if not 0 <= self.pc < len(self.program): raise RuntimeError(f"IR3 pc out of range: {self.pc}")
        inst, next_pc = self.program[self.pc], self.pc+1
        if self.predication and inst.opcode not in ("predt","predf","prede"):
            enabled = self.predication_mask
            if self.predication == 2: enabled = not enabled
            if not enabled:
                self.pc = next_pc
                return
        if inst.opcode == "nop": pass
        elif inst.opcode == "end": self.done = True
        elif inst.opcode == "kill":
            if bool(self._read(inst.srcs[0]))^inst.invert[0]: self.done = True
        elif inst.opcode == "predt": self.predication, self.predication_mask = 1, bool(self.pred[0])
        elif inst.opcode == "predf": self.predication, self.predication_mask = 2, bool(self.pred[0])
        elif inst.opcode == "prede": self.predication = 0
        elif inst.opcode in ("jump","call"):
            if inst.opcode == "call": self.call_stack.append(next_pc)
            next_pc = inst.pc+inst.branch_offset
        elif inst.opcode == "ret":
            if not self.call_stack: raise RuntimeError("IR3 return with empty call stack")
            next_pc = self.call_stack.pop()
        elif inst.opcode in ("br","bany","ball","brao","braa","brac","brax"):
            values = [bool(self._read(x))^inv for x,inv in zip(inst.srcs,inst.invert)]
            take = (values[0] if inst.opcode in ("br","bany","ball") else any(values) if inst.opcode == "brao" else
                    all(values) if inst.opcode == "braa" else True)
            if take: next_pc = inst.pc+inst.branch_offset
        elif inst.opcode.startswith("ldg.a."):
            size = _type_size(inst.dst_type)
            shift = self._read(inst.srcs[3])
            type_shift = 0 if inst.dst_type >= TYPE_U8 else 1 if inst.dst_type in HALF_TYPES else 2
            addr = self._addr64(inst.srcs[0],((self._read(inst.srcs[1])<<shift)+self._read(inst.srcs[2]))<<type_shift)
            for i in range(inst.width): self._write(self._offset(inst.dst,i),self._load_global(addr+i*size,size))
        elif inst.opcode.startswith("ldg."):
            size, addr = _type_size(inst.dst_type), self._addr64(inst.srcs[0],_s32(self._read(inst.srcs[1])))
            for i in range(inst.width): self._write(self._offset(inst.dst,i),self._load_global(addr+i*size,size))
        elif inst.opcode.startswith(("ldl.","ldlw.","ldp.")):
            size, addr = _type_size(inst.dst_type), _s32(self._read(inst.srcs[0]))+_s32(self._read(inst.srcs[1]))
            memory, space = (self.private,"private") if inst.opcode.startswith("ldp.") else (self.shared,"shared")
            for i in range(inst.width): self._write(self._offset(inst.dst,i),self._load_local(memory,addr+i*size,size,space))
        elif inst.opcode.startswith("stg.a."):
            size = _type_size(inst.src_type)
            shift = self._read(inst.srcs[3])
            type_shift = 0 if inst.src_type >= TYPE_U8 else 1 if inst.src_type in HALF_TYPES else 2
            addr = self._addr64(inst.srcs[0],((self._read(inst.srcs[1])<<shift)+self._read(inst.srcs[2]))<<type_shift)
            for i in range(inst.width):
                value_op = self._offset(inst.srcs[4],i)
                assert value_op is not None
                self._store_global(addr+i*size,self._read(value_op),size)
        elif inst.opcode.startswith("stg."):
            size, addr = _type_size(inst.src_type), self._addr64(inst.srcs[0],_s32(self._read(inst.srcs[1])))
            for i in range(inst.width):
                value_op = self._offset(inst.srcs[2],i)
                assert value_op is not None
                self._store_global(addr+i*size,self._read(value_op),size)
        elif inst.opcode.startswith(("stl.","stlw.","stp.")):
            size, addr = _type_size(inst.src_type), _s32(self._read(inst.srcs[0]))+_s32(self._read(inst.srcs[1]))
            memory, space = (self.private,"private") if inst.opcode.startswith("stp.") else (self.shared,"shared")
            for i in range(inst.width):
                value_op = self._offset(inst.srcs[2],i)
                assert value_op is not None
                self._store_local(memory,addr+i*size,self._read(value_op),size,space)
        elif inst.opcode.startswith("atomic."): self._atomic(inst)
        elif inst.opcode == "bar": self.waiting = True
        elif inst.opcode in {"fence","sleep","icinv","dccln","dcinv","dcflu"}: pass
        else:
            for iteration in range(inst.repeat+1): self._execute_alu(inst,iteration)
        self.pc = next_pc

    def run(self):
        while not self.done:
            self.step()
            if self.waiting: self.waiting = False

def run_ir3(code: bytes|bytearray|memoryview, constants: Sequence[int]|bytes|bytearray|memoryview=(),
            global_size: tuple[int,int,int]=(1,1,1), local_size: tuple[int,int,int]=(1,1,1), local_id_reg: int|None=None,
            workgroup_id_const: int|None=None, translate_addr: Callable[[int],int]|None=None,
            init: Callable[[IR3Machine,tuple[int,int,int],tuple[int,int,int],tuple[int,int,int]],None]|None=None,
            shared_size: int=0, private_size: int=0):
    program = decode_program(code)
    if any(g <= 0 for g in global_size) or any(l <= 0 for l in local_size): raise ValueError("IR3 launch sizes must be positive")
    if any(g%l for g,l in zip(global_size,local_size)): raise ValueError("global_size must be divisible by local_size")
    if shared_size < 0 or private_size < 0: raise ValueError("IR3 memory sizes must be nonnegative")
    group_count = tuple(g//l for g,l in zip(global_size,local_size))
    for wz in range(group_count[2]):
        for wy in range(group_count[1]):
            for wx in range(group_count[0]):
                wgid, shared, machines = (wx,wy,wz), bytearray(shared_size), []
                for lz in range(local_size[2]):
                    for ly in range(local_size[1]):
                        for lx in range(local_size[0]):
                            lid = (lx,ly,lz)
                            gid = (wx*local_size[0]+lx,wy*local_size[1]+ly,wz*local_size[2]+lz)
                            machine = IR3Machine(program,constants,translate_addr,shared,private_size)
                            if local_id_reg is not None:
                                for i,val in enumerate(lid): machine._write(_reg(local_id_reg+i),val)
                            if workgroup_id_const is not None:
                                for i,val in enumerate(wgid): machine._write(_reg(workgroup_id_const+i),val)
                            if init is not None: init(machine,gid,lid,wgid)
                            machines.append(machine)
                while any(not machine.done for machine in machines):
                    runnable = [machine for machine in machines if not machine.done and not machine.waiting]
                    if runnable:
                        for machine in runnable: machine.step()
                    else:
                        for machine in machines: machine.waiting = False
