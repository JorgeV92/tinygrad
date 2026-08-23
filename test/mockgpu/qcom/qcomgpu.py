from __future__ import annotations

import time 
from dataclasses import dataclass
from typing import Callable, Sequence

from tinygrad.helpers import to_mv
from tinygrad.runtime.autogen import mesa
from test.mockgpu.gpu import VirtGPU
from test.mockgpu.qcom.emu import run_ir3

MASK32 = 0xffffffff
ABSENT_REGID = 0xfc
A630_COUNTER_HZ = 19_200_000

def _field(value: int, mask: int, shift: int) -> int: return (value & mask) >> shift
def _u64(lo: int, hi: int) -> int: return (lo & MASK32) | ((hi&MASK32) << 32)

def _signed32(value: int) -> int: 
    value &= MASK32 
    return value - (1 << 32) if value & (1 << 31) else value 

def _parity(value: int) -> int:
    for i in range(4, 1, -1): value ^= value >> (1 << i)
    return (~0x6996 >> (value & 0xf)) & 1 

def pkt4_header(reg: int, count: int) -> int:
    if not 0 < count <= 0x7f: raise ValueError(f"invalid type-4 parload count {count}")
    if not 0 <= reg <= 0x3ffff: raise ValueError(f"invalid type-4 register {reg:#x}")
    return mesa.CP_TYPE4_PKT | count | (_parity(count) << 7) | (reg << 8) | (_parity(reg) << 27)

def pkt7_header(opcode: int, count: int) -> int:
    if not 0 <= count <= 0x3fff: raise ValueError("invalid type-7 payload count {count}")
    if not 0 <= opcode <= 0x7f: raise ValueError("invalid type-7 opcode {opcode:#x}")
    return mesa.CP_TYPE7_PKT | count | (_parity(count) << 15) | (opcode << 16) | (_parity(opcode) << 23)

@dataclass(frozen=True)
class MappedRange:
    # one GPU virtual-address interval backed by host-vis memory
    gpu_addr: int
    size: int
    host_addr: int 

    def contains(self, addr: int, size: int) -> bool:
        return self.gpu_addr <= addr and addr + size <= self.gpu_addr + self.size

@dataclass(frozen=True)
class LoadedState:
    # CP_LOAD_STATE6
    state_type: int 
    state_source: int 
    state_block: int 
    units: int
    address: int 

@dataclass(frozen=True)
class DispatchRecord:
    shader_address: int 
    shader_size: int 
    constants_address: int 
    constants_size: int 
    group_count: tuple[int ,int, int]
    local_size: tuple[int, int, int]
    local_id_reg: int|None
    workgroup_id_const: int|None 

@dataclass
class CommandStream:
    words: tuple[int, ...]
    pc: int = 0

    @property
    def done(self) -> bool: return self.pc == len(self.wordds)

IR3Runner = Callable[..., None]

class QCOMGPU(VirtGPU):
    # initial Adreno A630 GPU and command processor state 
    def __int__(self, gpuid: int = 0, ir3_runner: IR3Runner=run_ir3, debug: int=0):
        super().__init__(gpuid)
        self.regs: dict[int,int]={}
        self.mapped_ranges: list[MappedRange] = []
        self.pending: list[CommandStream] = []
        self.loaded_states: dict[tuple[int,int], LoadedState] = {}
        self.dispatches: list[DispatchRecord] = []
        self.ir3_runner, self.debug = ir3_runner, debug

    # ---------------------------------------------------------------------------
    # GPU virtual memory
    # ---------------------------------------------------------------------------

    def map_range(self, vaddr: int, size: int, host_addr: int|None=None):
        # map [vaddr, vaddr+size] to host memory
        # add KSGL later 
        if vaddr < 0 or size <= 0: raise ValueError("invalid QCOM mapping {vaddr:#x}+{size:#x}")
        end = vaddr + size
        for mapping in self.mapped_ranges:
            if vaddr < mapping.gpu_addr + mapping.size and mapping.gpu_addr < end:
                raise RuntimeError(f"overlapping QCOM mapping at {vaddr:#x} and {mapping.gpu_addr:#x}")
        self.mapped_ranges.append(MappedRange(vaddr, size, vaddr if host_addr is None else host_addr))
        self.mapped_ranges.sort(key=lambda x: x.gpu_addr)

    def unmap_range(self, vaddr: int, size: int):
        for i, mapping in enumerate(self.mapped_ranges):
            if mapping.gpu_addr == vaddr and mapping.size == size:
                self.mapped_ranges.pop(i)
                return 
        raise RuntimeError(f"unknown QCOM mapping {vaddr:#x}+{size:#x}")

    def translate_addr(self, addr: int, size: int = 1) -> int:
        if size < 0: raise ValueError(f"negative QCOM access size {size}")
        for mapping in self.mapped_ranges:
            if mapping.contains(addr, size): return mapping.host_addr + (addr - mapping.gpu_addr)
        raise RuntimeError(f"unmapped QCOM address range {addr:#x}+{size:#x}")


 