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
    # cat0: flow/control 
    if cat == 0:
        opc, repeat = _bits(raw, 55, 58), _bits(raw, 40, 42)
        if opc == 0: return IR3Instruction(pc, raw, "nop", repeat=repeat)
        if opc == 6: return IR3Instruction(pc, raw, "end")
        raise NotImplementedError(f"IR3 cat0 opcode {opc:#x} at pc {pc}")
    # cat1 move and scalar conversions
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
    # TODO: add more 
