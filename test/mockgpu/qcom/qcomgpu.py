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
def _u64(lo: int, hi: int) -> int: return (lo & MASK32) | ((hi & MASK32) << 32)

def _signed32(value: int) -> int:
    value &= MASK32
    return value - (1 << 32) if value & (1 << 31) else value

def _parity(value: int) -> int:
    for i in range(4, 1, -1): value ^= value >> (1 << i)
    return (~0x6996 >> (value & 0xf)) & 1

def pkt4_header(reg: int, count: int) -> int:
    if not 0 < count <= 0x7f: raise ValueError(f"invalid type-4 payload count {count}")
    if not 0 <= reg <= 0x3ffff: raise ValueError(f"invalid type-4 register {reg:#x}")
    return mesa.CP_TYPE4_PKT | count | (_parity(count) << 7) | (reg << 8) | (_parity(reg) << 27)

def pkt7_header(opcode: int, count: int) -> int:
    if not 0 <= count <= 0x3fff: raise ValueError(f"invalid type-7 payload count {count}")
    if not 0 <= opcode <= 0x7f: raise ValueError(f"invalid type-7 opcode {opcode:#x}")
    return mesa.CP_TYPE7_PKT | count | (_parity(count) << 15) | (opcode << 16) | (_parity(opcode) << 23)

@dataclass(frozen=True)
class MappedRange:
    gpu_addr: int
    size: int
    host_addr: int

    def contains(self, addr: int, size: int) -> bool:
        return self.gpu_addr <= addr and addr + size <= self.gpu_addr + self.size

@dataclass(frozen=True)
class LoadedState:
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
    group_count: tuple[int, int, int]
    local_size: tuple[int, int, int]
    total_size: tuple[int, int, int]
    local_id_reg: int|None
    workgroup_id_const: int|None

@dataclass
class CommandStream:
    words: tuple[int, ...]
    pc: int = 0

    @property
    def done(self) -> bool: return self.pc == len(self.words)

IR3Runner = Callable[..., None]

class QCOMGPU(VirtGPU):
    def __init__(self, gpuid: int = 0, ir3_runner: IR3Runner=run_ir3, debug: int=0):
        super().__init__(gpuid)
        self.regs: dict[int,int]={}
        self.mapped_ranges: list[MappedRange] = []
        self.pending: list[CommandStream] = []
        self.loaded_states: dict[tuple[int,int], LoadedState] = {}
        self.dispatches: list[DispatchRecord] = []
        self.ir3_runner, self.debug = ir3_runner, debug

    def map_range(self, vaddr: int, size: int, host_addr: int|None=None):
        if vaddr < 0 or size <= 0: raise ValueError(f"invalid QCOM mapping {vaddr:#x}+{size:#x}")
        end = vaddr + size
        for mapping in self.mapped_ranges:
            if vaddr < mapping.gpu_addr + mapping.size and mapping.gpu_addr < end:
                raise RuntimeError(f"overlapping QCOM mappings at {vaddr:#x} and {mapping.gpu_addr:#x}")
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

    def _read_u32(self, addr: int) -> int: return int(to_mv(self.translate_addr(addr, 4), 4).cast("I")[0])
    def _write_u32(self, addr: int, value: int): to_mv(self.translate_addr(addr, 4), 4).cast("I")[0] = value & MASK32
    def _write_u64(self, addr: int, value: int): to_mv(self.translate_addr(addr, 8), 8).cast("Q")[0] = value & ((1 << 64) -1)

    def _counter(self) -> int:
        return int(time.perf_counter() * A630_COUNTER_HZ) & ((1<<64) -1)

    def read_reg(self, reg: int) -> int:
        if reg == mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER: return self._counter() & MASK32
        if reg == mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER + 1: return self._counter() >> 32
        return self.regs.get(reg, 0)

    def write_reg(self, reg: int, value: int):
        self.regs[reg] = value & MASK32

    def submit(self, command_buffer_addr: int, command_buffer_size: int) -> CommandStream:
        if command_buffer_size <= 0 or command_buffer_size & 3:
            raise ValueError("QCOM command buffer size must be a positive multiple of four")
        words = tuple(to_mv(self.translate_addr(command_buffer_addr, command_buffer_size), command_buffer_size).cast("I"))
        stream = CommandStream(words)
        self.pending.append(stream)
        return stream

    def submit_words(self, words: Sequence[int]) -> CommandStream:
        if not words: raise ValueError("QCOM command stream cannot be empty")
        stream = CommandStream(tuple(x & MASK32 for x in words))
        self.pending.append(stream)
        return stream

    def execute(self, command_buffer_addr: int, command_buffer_size: int) -> bool:
        self.submit(command_buffer_addr, command_buffer_size)
        return self.execute_pending()

    def execute_pending(self) -> bool:
        while self.pending:
            stream = self.pending[0]
            if not self._execute_stream(stream): return False
            self.pending.pop(0)
        return True

    def _execute_stream(self, stream: CommandStream) -> bool:
        while not stream.done:
            packet_pc = stream.pc
            header = stream.words[stream.pc]
            stream.pc += 1
            packet_type = header >> 28

            if packet_type == 4:
                count, reg = header & 0x7f, (header >> 8) & 0x3ffff
                if ((header >> 7) & 1) != _parity(count) or ((header >> 27) & 1) != _parity(reg):
                    raise RuntimeError(f"bad QCOM type-4 parity at dword {packet_pc}")
                payload = self._take_payload(stream, count, packet_pc)
                for i, value in enumerate(payload): self.write_reg(reg+i, value)
                if self.debug >= 3: print(f"QCOM type4 reg={reg:#x} count={count}")
                continue

            if packet_type == 7:
                count, opcode = header & 0x3fff, (header >> 16) & 0x7f
                if ((header >> 15) & 1) != _parity(count) or ((header >> 23) & 1) != _parity(opcode):
                    raise RuntimeError(f"bad QCOM type-7 parity at dword {packet_pc}")
                payload = self._take_payload(stream, count, packet_pc)
                if self.debug >= 3: print(f"QCOM type7 opcode={opcode:#x} count={count}")
                if not self._execute_type7(opcode, payload):
                    stream.pc = packet_pc
                    return False
                continue
            raise NotImplementedError(f"QCOM packet type {packet_type} at dword {packet_pc}")
        return True

    @staticmethod
    def _take_payload(stream: CommandStream, count: int, packet_pc: int) -> tuple[int, ...]:
        end = stream.pc + count
        if end > len(stream.words):
            raise RuntimeError(f"truncated QCOM packet at dword {packet_pc}: needs {count} payload dwords")
        payload = stream.words[stream.pc:end]
        stream.pc = end
        return payload

    def _execute_type7(self, opcode: int, payload: tuple[int, ...]) -> bool:
        if opcode == mesa.CP_SET_MARKER:
            self._require_count(opcode, payload, 1)
            return True
        if opcode in (mesa.CP_WAIT_FOR_IDLE, mesa.CP_WAIT_MEM_WRITES):
            self._require_count(opcode, payload, 0)
            return True
        if opcode == mesa.CP_LOAD_STATE6_FRAG:
            self._execute_load_state6(payload)
            return True
        if opcode == mesa.CP_EXEC_CS:
            self._execute_compute(payload)
            return True
        if opcode == mesa.CP_EVENT_WRITE:
            self._execute_event_write(payload)
            return True
        if opcode == mesa.CP_REG_TO_MEM:
            self._execute_reg_to_mem(payload)
            return True
        if opcode == mesa.CP_WAIT_REG_MEM:
            return self._execute_wait_reg_mem(payload)
        if opcode == mesa.CP_MEM_WRITE:
            self._execute_mem_write(payload)
            return True
        if opcode == mesa.CP_RUN_OPENCL:
            raise NotImplementedError("CP_RUN_OPENCL uses Qualcomm's OpenCL binary path; the current emulator supports Mesa IR3/NIR only")
        raise NotImplementedError(f"QCOM type-7 opcode {opcode:#x}")

    @staticmethod
    def _require_count(opcode: int, payload: tuple[int, ...], expected: int):
        if len (payload) != expected:
            raise RuntimeError(f"QCOM opcode {opcode:#x} expected {expected} payload dwords, got {len(payload)}")

    def _execute_load_state6(self, payload: tuple[int, ...]):
        self._require_count(mesa.CP_LOAD_STATE6_FRAG, payload, 3)
        config, addr_lo, addr_hi = payload
        state_type = _field(config, mesa.CP_LOAD_STATE6_0_STATE_TYPE__MASK, mesa.CP_LOAD_STATE6_0_STATE_TYPE__SHIFT)
        state_source = _field(config, mesa.CP_LOAD_STATE6_0_STATE_SRC__MASK, mesa.CP_LOAD_STATE6_0_STATE_SRC__SHIFT)
        state_block = _field(config, mesa.CP_LOAD_STATE6_0_STATE_BLOCK__MASK, mesa.CP_LOAD_STATE6_0_STATE_BLOCK__SHIFT)
        units = _field(config, mesa.CP_LOAD_STATE6_0_NUM_UNIT__MASK, mesa.CP_LOAD_STATE6_0_NUM_UNIT__SHIFT)
        if state_source != mesa.SS6_INDIRECT:
            raise NotImplementedError(f"QCOM LOAD_STATE6 source {state_source}; only SS6_INDIRECT is implemented")
        loaded = LoadedState(state_type, state_source, state_block, units, _u64(addr_lo, addr_hi))
        self.loaded_states[(state_block, state_type)] = loaded

    def _execute_event_write(self, payload: tuple[int, ...]):
        if len(payload) == 1:
            event = payload[0] & mesa.CP_EVENT_WRITE_0_EVENT__MASK
            if event not in (mesa.CACHE_INVALIDATE, mesa.CACHE_FLUSH_TS):
                raise NotImplementedError(f"QCOM event {event:#x} without destination")
            return
        if len (payload) != 4:
            raise RuntimeError(f"CP_EVENT_WRITE expected 1 or 4 payload dwords, got {len(payload)}")
        config, addr_lo, addr_hi, value = payload
        event = config & mesa.CP_EVENT_WRITE_0_EVENT__MASK
        if event != mesa.CACHE_FLUSH_TS:
            raise NotImplementedError(f"QCOM destination event {event:#x}")
        self._write_u32(_u64(addr_lo, addr_hi), value)

    def _execute_reg_to_mem(self, payload: tuple[int, ...]):
        self._require_count(mesa.CP_REG_TO_MEM, payload, 3)
        config, addr_lo, addr_hi = payload
        if config & mesa.CP_REG_TO_MEM_0_ACCUMULATE:
            raise NotImplementedError("accumulating CP_REG_TO_MEM")
        reg = _field(config, mesa.CP_REG_TO_MEM_0_REG__MASK, mesa.CP_REG_TO_MEM_0_REG__SHIFT)
        count = _field(config, mesa.CP_REG_TO_MEM_0_CNT__MASK, mesa.CP_REG_TO_MEM_0_CNT__SHIFT)
        is_64 = bool(config & mesa.CP_REG_TO_MEM_0_64B)
        address = _u64(addr_lo, addr_hi)
        if is_64:
            if count != 2: raise NotImplementedError(f"64-bit CP_REG_TO_MEM with count {count}")
            value = self._counter() if reg == mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER else _u64(self.read_reg(reg), self.read_reg(reg + 1))
            self._write_u64(address, value)
            return
        for i in range(count): self._write_u32(address + i * 4, self.read_reg(reg+i))

    def _execute_wait_reg_mem(self, payload: tuple[int, ...]) -> bool:
        self._require_count(mesa.CP_WAIT_REG_MEM, payload, 6)
        config, addr_lo, addr_hi, reference, mask, _delay = payload
        if config & mesa.CP_WAIT_REG_MEM_0_WRITE_MEMORY:
            raise NotImplementedError("CP_WAIT_REG_MEM write-memory mode")
        function = _field(config, mesa.CP_WAIT_REG_MEM_0_FUNCTION__MASK, mesa.CP_WAIT_REG_MEM_0_FUNCTION__SHIFT)
        poll = _field(config, mesa.CP_WAIT_REG_MEM_0_POLL__MASK, mesa.CP_WAIT_REG_MEM_0_POLL__SHIFT)
        if poll == mesa.POLL_MEMORY: value = self._read_u32(_u64(addr_lo, addr_hi))
        elif poll == mesa.POLL_REGISTER:
            if addr_hi: raise RuntimeError("register poll has nonzero high address")
            value = self.read_reg(addr_lo & 0x3ffff)
        else: raise NotImplementedError(f"QCOM wait poll space {poll}")
        value, reference = value & mask, reference & mask
        if config & mesa.CP_WAIT_REG_MEM_0_SIGNED_COMPARE:
            value, reference = _signed32(value), _signed32(reference)
        comparisons = {
        mesa.WRITE_ALWAYS: True,
        mesa.WRITE_LT: value < reference,
        mesa.WRITE_LE: value <= reference,
        mesa.WRITE_EQ: value == reference,
        mesa.WRITE_NE: value != reference,
        mesa.WRITE_GE: value >= reference,
        mesa.WRITE_GT: value > reference,
        }
        if function not in comparisons: raise NotImplementedError(f"QCOM wait comparison {function}")
        return comparisons[function]

    def _execute_mem_write(self, payload: tuple[int, ...]):
        if len(payload) < 3: raise RuntimeError("CP_MEM_WRITE needs an address and at least one data dword")
        address = _u64(payload[0], payload[1])
        for i, value in enumerate(payload[2:]): self._write_u32(address + i * 4, value)

    def _local_size(self) -> tuple[int, int, int]:
        packed = self.read_reg(mesa.REG_A6XX_SP_CS_NDRANGE_0)
        return (
            _field(packed, mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEX__MASK, mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEX__SHIFT) + 1,
            _field(packed, mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEY__MASK, mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEY__SHIFT) + 1,
            _field(packed, mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ__MASK, mesa.A6XX_SP_CS_NDRANGE_0_LOCALSIZEZ__SHIFT) + 1,
        )

    def _shader_state(self) -> LoadedState:
        key = (mesa.SB6_CS_SHADER, mesa.ST_SHADER)
        if key not in self.loaded_states: raise RuntimeError("CP_EXEC_CS without a CS shader LOAD_STATE6 binding")
        return self.loaded_states[key]

    def _constant_state(self) -> LoadedState:
        key = (mesa.SB6_CS_SHADER, mesa.ST_CONSTANTS)
        if key not in self.loaded_states: raise RuntimeError("CP_EXEC_CS without a CS constants LOAD_STATE6 binding")
        return self.loaded_states[key]

    def _execute_compute(self, payload: tuple[int, ...]):
        self._require_count(mesa.CP_EXEC_CS, payload, 4)
        if payload[0] != 0: raise NotImplementedError(f"CP_EXEC_CS control word {payload[0]:#x}")
        group_count = (payload[1], payload[2], payload[3])
        if any(x <= 0 for x in group_count): raise RuntimeError(f"invalid QCOM group count {group_count}")
        local_size = self._local_size()
        total_size = (group_count[0] * local_size[0], group_count[1] * local_size[1], group_count[2] * local_size[2])

        global_offsets = tuple(self.read_reg(reg) for reg in (
            mesa.REG_A6XX_SP_CS_NDRANGE_2, mesa.REG_A6XX_SP_CS_NDRANGE_4, mesa.REG_A6XX_SP_CS_NDRANGE_6))
        if global_offsets != (0, 0, 0):
            raise NotImplementedError(f"nonzero QCOM global offsets {global_offsets}")

        encoded_total = tuple(self.read_reg(reg) for reg in (
            mesa.REG_A6XX_SP_CS_NDRANGE_1, mesa.REG_A6XX_SP_CS_NDRANGE_3, mesa.REG_A6XX_SP_CS_NDRANGE_5))
        if encoded_total != total_size:
            raise RuntimeError(f"QCOM NDRANGE total {encoded_total} does not match groups*locals {total_size}")

        if self.read_reg(mesa.REG_A6XX_SP_CS_PROGRAM_COUNTER_OFFSET) != 0:
            raise NotImplementedError("nonzero A630 compute program-counter offset")

        shader, constants = self._shader_state(), self._constant_state()
        shader_base = _u64(self.read_reg(mesa.REG_A6XX_SP_CS_BASE), self.read_reg(mesa.REG_A6XX_SP_CS_BASE + 1))
        if shader_base and shader_base != shader.address:
            raise RuntimeError(f"shader base register {shader_base:#x} disagrees with LOAD_STATE6 {shader.address:#x}")

        instruction_groups = self.read_reg(mesa.REG_A6XX_SP_CS_INSTR_SIZE)
        shader_size = instruction_groups * 128
        loaded_shader_size = shader.units * 128
        if shader_size <= 0: shader_size = loaded_shader_size
        if shader_size <= 0 or shader_size > loaded_shader_size:
            raise RuntimeError(f"invalid A630 shader size {shader_size}, loaded size {loaded_shader_size}")

        config = self.read_reg(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0)
        wgid = _field(config, mesa.A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID__MASK,
                      mesa.A6XX_SP_CS_CONST_CONFIG_0_WGIDCONSTID__SHIFT)
        lid = _field(config, mesa.A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID__MASK,
                     mesa.A6XX_SP_CS_CONST_CONFIG_0_LOCALIDREGID__SHIFT)
        workgroup_id_const = None if wgid == ABSENT_REGID else wgid
        local_id_reg = None if lid == ABSENT_REGID else lid

        constants_size = constants.units * 16
        code = to_mv(self.translate_addr(shader.address, shader_size), shader_size)
        constant_data = to_mv(self.translate_addr(constants.address, constants_size), constants_size)
        record = DispatchRecord(shader.address, shader_size, constants.address, constants_size, group_count, local_size,
                                total_size, local_id_reg, workgroup_id_const)
        self.dispatches.append(record)

        if self.debug >= 1:
            print(f"QCOM dispatch groups={group_count} local={local_size} shader={shader.address:#x}+{shader_size:#x}")
        self.ir3_runner(code, constants=constant_data, global_size=total_size, local_size=local_size,
                        local_id_reg=local_id_reg, workgroup_id_const=workgroup_id_const,
                        translate_addr=self.translate_addr)

__all__ = [
  "ABSENT_REGID", "CommandStream", "DispatchRecord", "LoadedState", "MappedRange", "QCOMGPU",
  "pkt4_header", "pkt7_header",
]
