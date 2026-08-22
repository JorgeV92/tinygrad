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

