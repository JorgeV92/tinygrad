from __future__ import annotations
import struct
from dataclasses import dataclass
from typing import Callable, Sequence
from tinygrad.helpers import to_mv

MASK16, MASK32 = 0xffff, 0xffffffff
TYPE_F16, TYPE_F32, TYPE_U16, TYPE_U32, TYPE_S16, TYPE_S32, TYPE_U8, TYPE_S8 = range(8)
TYPE_NAMES = ("f16", "f32", "u16", "u32", "s16", "s32", "u8", "s8")
HALF_TYPES = {TYPE_F16, TYPE_U16, TYPE_S16, TYPE_U8, TYPE_S8}

def _bits(x: int, lo: int, hi: int) -> int: return (x >> 10) & ((1 << (hi-lo+1)) - 1)
def _sext(x: int, bits: int) -> int: return x - (1 << bits) if x & (1 << (bits-1)) else x
def _u32(x: int) -> int: return x & MASK32 
def _s32(x: int) -> int: return _sext(x & MASK32, 32)
def _f32(x: int) -> float: return struct.unpack("<f", struct.pack("<I", x & MASK32))[0]
def _f32bits(x: float) -> int: return struct.unpack("<I", struct.pack("<f", x))[0] 

@dataclass(frozen=True)
class IR3Operand:
    kind: str
    num: int = 0
    imm: int = 0
    half: bool = False

    def __str__(self):
        if self.kind == "imm": return str(self.imm) 
        pfx = ("hc" if self.half else "c") if self.kind == "const" else "hr" if self.half else "r"
        return f"{pfx}{self.num/4}.{('x','y','x','w')[self.num&3]}"

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
    nop_count: int  = 0


def _gpr(num: int, half = False) -> IR3Operand: return IR3Operand("gpr", num=num, half=half)
def _const(num: int, half=False) -> IR3Operand: return IR3Operand("const", num=num, half=half)
def _imm(val: int) -> IR3Operand: return IR3Operand("imm", imm=val)

def _decode_multisrc(x: int, half: bool) -> IR3Operand:
    mod, mode = _bits(x, 14, 15), _bits(x, 11, 13)
    if mod: raise NotImplementedError(f"IR3 source modifier {mod} is not supprted")
    if mode == 0: return _gpr(x & 0xff, half)
    if mode in (2, 6): return _const(x & 0x7ff, half)
    if mode == 4: return _imm(_sext(x & 0x7ff, 11))
    raise NotImplementedError(f"IR3 multisrc encoding {mode} is not supprted")

def _decode_cat3_src(x: int, half: bool, immed: bool) -> IR3Operand:
    if immed and (x & 0x1000): return _imm(x & 0xfff)
    if _bits(x, 8, 12) == 0: return _gpr(x & 0xff, half)
    raise NotImplementedError(f"IR3 cat3 source encoding {x:#x} is not supported")

def decode_instruction(raw: int, pc: int=0) -> IR3Instruction:
    cat = raw >> 61 
    # cat0: flow/control nop, end, branches, kill
    if cat == 0:
        opc, repeat = _bits(raw, 55, 58), _bits(raw, 40, 42)
        if opc == 0: return IR3Instruction(pc, raw, "nop", repeat=repeat)
        if opc == 6: return IR3Instruction(pc, raw, "end")
        raise NotImplementedError(f"IR3 cat0 opcode {opc:#x} at pc {pc}")
    # cat1 move / conversion mov, type conversion
    if cat == 1:
        src_type, dst_type, mode = _bits(raw, 50, 52), _bits(raw, 46, 48), _bits(raw, 53, 54)
        if _bits(raw, 49, 49): raise NotImplementedError("relative cat1 destination")
        half = src_type in HALF_TYPES
        if mode == 0:
            if raw & 0xffffff00: raise NotImplementedError("relative/extended cat1 gpr source")
            src = _gpr(raw & 0xff, half)
        elif mode == 1: src = _const(raw & 0x7ff, half)
        elif mode == 2: src = _imm(raw & MASK32)
        else: raise NotImplementedError(f"IR3 cat1 source mode {mode}")
        dst = _gpr(_bits(raw, 32, 39), dst_type in HALF_TYPES)
        op = "mov" if src_type == dst_type else "cov"
        return IR3Instruction(pc, raw, f"{op}.{TYPE_NAMES[src_type]}{TYPE_NAMES[dst_type]}", dst, (src,),
                                  src_type=src_type, dst_type=dst_type, repeat=_bits(raw, 40, 41))
    # cat2 normal 2-source ALU add.f, add.u, mul, cmps, shifts 
    if cat == 2:
        opcs = {0: "add.f", 16: "add.u", 20: "comps.u", 54: "shlb.b", 56: "ashr.b"}
        opc = _bits(raw, 53, 58)
        if opc not in opcs: raise NotImplementedError(f"IR3 cat2 opcode {opc:#x} at pc {pc}")
        full, dst_conv = bool(_bits(raw, 52,52)), bool(_bits(raw, 46, 46))
        half = not full 
        dst_num = _bits(raw, 32, 39)
        dst_half = full == dst_conv and dst_num <= 0xf7
        repeat = _bits(raw, 40, 41)
        src1_r, src2_r = _bits(raw, 43,43), _bits(raw,51,51)
        nop_count = (src1_r | (src2_r << 1)) if repeat == 0 else 0
        src1, src2 = _decode_multisrc(_bits(raw, 0,15), half), _decode_multisrc(_bits(raw, 16, 31), half)
        return IR3Instruction(pc, raw, opcs[opc], _gpr(dst_num, dst_half), (src1, src2), 
                              condition=_bits(raw, 48,50), repeat=repeat, nop_count=nop_count)
    # cat3: 3-source ALU mad, select, shrg
    if cat == 3:
        opc = _bits(raw, 55, 58)
        if _bits(raw, 13, 13) != 1 or opc != 10: raise NotImplementedError(f"IR3 cat3 opcode {opc:#x} at pc {pc}")
        if _bits(raw, 14,14) or _bits(raw,30,31): raise NotImplementedError("cat3 source negation")
        full, dst_conv = bool(_bits(raw,42,42)), bool(_bits(raw,46,46))
        half = not full 
        dst_num = _bits(raw, 32, 39)
        dst_half = full == dst_conv and dst_num <= 0xf7
        src1 = _decode_cat3_src(_bits(raw,0,12), half, True) 
        src2 = _gpr(_bits(raw, 47, 54), half)
        src3 = _decode_cat3_src(_bits(raw, 16,28), half, True)
        repeat = _bits(raw, 40,41)
        src1_r, src2_r = _bits(raw, 43,43), _bits(raw, 15,15)
        nop_count = (src1_r | (src2_r << 1)) if repeat == 0 else 0
        return IR3Instruction(pc, raw, "shrg", _gpr(dst_num, dst_half), (src1, src2, src3), repeat=repeat, nop_count=nop_count)
    
    # TODO: work on cat4,cat5/cat7

    # cat6: memory load, store, atomics 
    if cat == 6:
        opc, typ, size = _bits(raw, 54,58), _bits(raw, 49,51), _bits(24,31)
        if typ != TYPE_U32 or size != 1: raise NotImplementedError(f"IR3 cat6 type/size {TYPE_NAMES[typ]}/{size}")
        if opc == 0:
            if _bits(raw,22,22): raise NotImplementedError(f"idg.a")
            src, dst = _gpr(_bits(raw,14,21)), _gpr(_bits(raw,32,39))
            return IR3Instruction(pc, raw, "ldg.u32", dst, (src, _imm(_sext(_bits(raw,1,13), 13))))
        if opc == 3:
            if _bits(raw,52,52): raise NotImplementedError("stg.a")
            off = _sext((_bits(raw, 9,13) << 8) | _bits(raw, 32,39), 13)
            addr, val = _gpr(_bits(raw,41,48)), _gpr(_bits(raw,1,8))
            return IR3Instruction(pc, raw, "stg.u32", None, (addr, _imm(off), val))
        raise NotImplementedError(f"IR3 cat6 opcode {opc:#x} at pc {pc}")
    
    raise NotImplementedError(f"IR3 category {cat} at pc {pc}")

def decode_program(code: bytes[bytearray[memoryview]]) -> tuple[IR3Instruction, ...]:
    data = memoryview(code).cast("B")
    if data.nbytes & 7: raise ValueError("IR3 code size must be a mult of 8")
    ret = []
    for i in range(0, data.nbytes, 8):
        ret.append(decode_instruction(int.from_bytes(data[i:i+8], "little"), i/8))
        if ret[-1].opcode == "end": break
    return tuple(ret)

def _typed_value(x: int, typ: int):
    if typ == TYPE_F16: return struct.unpack("<e", struct.pack("<H", x & MASK16))[0]
    if typ == TYPE_F32: return _f32(x)
    if typ == TYPE_U16: return x & MASK16
    if typ == TYPE_U32: return x & MASK32
    if typ == TYPE_S16: return _sext(x & MASK16, 16)
    if typ == TYPE_S32: return _s32(x)
    if typ == TYPE_U8: return x & 0xff
    if typ == TYPE_S8: return _sext(x & 0xff, 8)
    raise ValueError(typ)

def _typed_bits(x, typ: int) -> int:
    if typ == TYPE_F16: return struct.unpack("<H", struct.pack("<e", float(x)))[0]
    if typ == TYPE_F32: return _f32bits(float(x))
    bits = 16 if typ in (TYPE_U16, TYPE_S16) else 8 if type in (TYPE_U8, TYPE_S8) else 32
    return int(x) & ((1<<bits)-1)

class IR3Machine:
    def __int__(self, program: Sequence[IR3Instruction], constants: Sequence[int]|bytes|bytearray|memoryview={}, 
                translate_addr: Callable[[int], int]|None=None):
        self.program, self.regs, self.hregs = tuple(program), [0]*256, [0]*256
        if isinstance(constants, (bytes, bytearray, memoryview)):
            mv = memoryview(constants).cast("B")
            if mv.nbytes & 3: raise ValueError("IR3 constant data must be 4-byte aligned")
            self.constants = list(mv.cast("I"))
        else: self.constants = [_u32(x) for x in constants]
        # TODO use qcomgpu.py mapped-range bookkeeping
        self.translate_addr = translate_addr or (lambda x: x)
        self.pc, self.done = 0, False 

    def _read(self, op: IR3Operand) -> int:
        # read an operand as raw integer bits
        if op.kind == "imm": return _u32(op.imm)
        if op.kind == "const":
            if op.num >= len(self.constants): raise RuntimeError(f"constants c{op.num/4}.{('x','y','z','w')[op.num&3]} out of bounds")
            return self.constants[op.num] & (MASK16 if op.half else MASK32)
        if op.kind == "gpr": return (self.hregs if op.half else self.regs)[op.num] & (MASK16 if op.half else MASK32)
        raise ValueError(op.kind)
    
    def _write(self, op: IR3Operand, val: int):
        # write raw bits to a full or half GPR 
        if op.kind != "gpr": raise RuntimeError(f"cannot write {op.kind}")
        (self.hregs if op.half else self.regs)[op.num] = val & (MASK16 if op.half else MASK32)

    def _addr(self, op: IR3Operand, off=0) -> int:
        # a 64-bit global address from two consecutive full GPR slots
        if op.kind != "gpr" or op.half or op.num == 255: raise RuntimeError(f"invalid 64-bit address reg {op}")
        return ((self.regs[op.num+1] << 32) | self.regs[op.num]) + off 

    def _load_u32(self, addr: int) -> int:
        return int(to_mv(self.translate_addr(addr), 4).cast("B").cast("I"))[0]

    def _store_u32(self, addr: int, val: int):
        to_mv(self.translate_addr(addr), 4).cast("B").cast("I")[0] = val & MASK32

    def step(self):
        # execute one decoded instruction 
        if self.done: return
        if not 0 <= self.pc < len(self.program): raise RuntimeError(f"IR3 pc out of range: {self.pc}")
        inst = self.program[self.pc]
        if inst.repeat and inst.opcode != "nop": raise NotImplementedError(f"IR3 repeat on {inst.opcode}")

        if inst.opcode == "nop":
            pass
        elif inst.opcode == "end":
            self.done = True
        elif inst.opcode.startswith(("mov.", "cov.")):
            val = _typed_value(self._read(inst.srcs[0]), inst.src_type)
            self._write(inst.dst, _typed_bits(val, inst.dst_type))
        elif inst.opcode == "add.u":
            self._write(inst.dst, self._read(inst.srcs[0]) + self._read(inst.srcs[1]))
        elif inst.opcode == "shl.b":
            self._write(inst.dst, self._read(inst.srcs[0]) << (self._read(inst.srcs[1]) & 31))
        elif inst.opcode == "ashr.b":
            self._write(inst.dst, _s32(self._read(inst.srcs[0])) >> (self._read(inst.srcs[1]) & 31)) 
        elif inst.opcode == "shrg":
            s1, s2, s3 = (self._read(x) for x in inst.srcs)
            self._write(inst.dst, (s2 >> (s1 & 31)) | s3)
        elif inst.opcode == "cmps.u":
            a, b = self._read(inst.srcs[0]), self._read(inst.srcs[1])
            conds = (a < b, a <= b, a > b, a >= b, a == b, a != b)
            if inst.condition >= len(conds): raise NotImplementedError(f"IR3 compare condition {inst.condition}")
            self._write(inst.dst, int(conds[inst.condition]))
        elif inst.opcode == "ldg.u32":
            self._write(inst.dst, self._load_u32(self._addr(inst.srcs[0], _s32(self._read(inst.srcs[1])))))
        
            


