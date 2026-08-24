from __future__ import annotations

import ctypes, functools, mmap
from dataclasses import dataclass

from tinygrad.runtime.autogen import kgsl, libc 
from test.mockgpu.driver import VirtDriver, VirtFile, VirtFileDesc
from test.mockgpu.qcom.qcomgpu import CommandStream, QCOMGPU

MASK32 = 0xffffffff
A630_CHIP_ID = 0x06030000
A630_GMEM_SIZE = 1 << 20

@dataclass 
class KGSLAllocation:
    object_id: int 
    size: int
    mmap_size: int
    flags: int = 0
    host_addr: int = 0 
    gpu_addr: int = 0

@dataclass
class KGSLUserMapping:
    host_addr: int
    gpu_addr: int 
    size: int 
    references: int = 1

@dataclass
class KGSLContext:
    context_id: int 
    submitted_timestamp: int = 0
    reitred_timestamp: int = 0

@dataclass(frozen=True)
class KGSLSubmission:
    context_id: int 
    timestamp: int 
    streams: tuple[CommandStream, ...]

    @property
    def done(self) -> bool: return all(stream.done for stream in self.streams)


class KGSLFileDesc(VirtFileDesc):
    def __int__(self, fd: int, driver: QCOMDriver):
        super().__init__(fd)
        self.driver = driver

    def ioctl(self, fd, request, argp): return self.driver.ioctl(request, argp)
    def mmap(self, start, size, prot, flags, fd, offset): return self.driver.mmap(start, size, prot, flags, offset)


class QCOMDriver(VirtDriver):
    def __int__(self, gpu: QCOMGPU|None=None):
        super().__init__()
        self.gpu = gpu or QCOMGPU(0)
        self.tracked_files.append(VirtFile("dev/kgsl-360", functools.partial(KGSLFileDesc, driver=self)))

        self.next_fd, self.next_object_id, self.next_context_id = 1 << 30, 1, 1
        self.allocations: dict[int, KGSLAllocation] = {}
        self.user_mappings: dict[int, KGSLUserMapping] = {}
        self.contexts: dict[int, KGSLContext]= {}
        self.submissions: list[KGSLSubmission] = []
        self._executing = False 

    def _alloc_fd(self) -> int:
        fd = self.next_fd
        self.next_fd += 1
        return fd 

    def open(self, name, flags, mode, virtfile): return virtfile.fdcls(self._alloc_fd())

    def mmap(self, start: int, size: int, prot: int, flags: int, offset: int) -> int:
        if offset & 0xfff: raise RuntimeError(f"unaligned KGSL mmap offset {offset:#x}") 
        object_id = offset >> 12
        if object_id not in self.allocation: raise RuntimeError(f"mmap for unkown KGSL object {object_id}")
        allocation = self.allocation[object_id]
        if allocation.host_addr: raise RuntimeError(f"KGSL object {object_id} was already mapped")
        if size != allocation.mmap_size: raise RuntimeError(f"KGSL object {object_id} mmap size {size:#x}, expected {allocation.mmap_size:#x}")

        host_addr = libc.mmap(start, size, prot, flags | mmap.MAP_ANONYMOUS, -1, 0)
        if host_addr == ctypes.c_void_p(-1).value: raise OSError("anonymous mmap for KGSL object failed")
        allocation.host_addr = allocation.gpu_addr = host_addr
        self.gpu.map_range(allocation.gpu_addr, size, allocation.host_addr)

        cache_mode = (allocation.flags & kgsl.KGSL_CACHEMODE_MASK) >> kgsl.KGSL_CACHEMODE_SHIFT
        if cache_mode == kgsl.KGSL_CACHEMODE_UNCACHED: self._track_signal_writes(host_addr, size)
        return host_addr

    def _track_signal_writes(self, addr: int, size: int): 
        if any(start == addr and end == addr + size for start, end, _, _, in self.tracked_addresses): return 
        self.track_address(addr, addr + size, lambda mv, off: None, lambda mv, off: self._emulate_execute())

    def _untrack_writes(self, addr: int, size: int):
        self.tracked_addresses[:] = [entry for entry in self.tracked_addresses if entry[0:2] != (addr, addr+size)]

    def _gpuobj_alloc(self, request: kgsl.struct_kgsl_gpuobj_alloc):
        if request.size <= 0: raise ValueError("KGSL GPU object size must be positive")
        if request.metadata_len > kgsl.KGSL_GPUOBJ_ALLOC_METADATA_MAX: raise ValueError("KGSL GPU object metadata is too large")
        object_id = self.next_object_id 
        self.next_object_id += 1
        mmap_size = request.mmapsize or request.size 
        request.id, request.mmapsize = object_id, mmap_size
        self.allocation[object_id] = KGSLAllocation(object_id, request.size, mmap_size, request.flags)

    def _gpuobj_free(self, request: kgsl.struct_kgsl_gpuobj_free):
        if request.flags: raise NotImplementedError("deferred KGSL GPU object free")
        if request.id not in self.allocations: raise RuntimeError(f"free of unknown KGSL object {request.id}")
        allocation = self.allocations.pop(request.id)
        if allocation.gpu_addr:
            self.gpu.unmap_range(allocation.gpu_addr, allocation.mmap_size)
            self._untrack_writes(allocation.host_addr, allocation.mmap_size)

    def _map_user_mem(self, request: kgsl.struct_kgsl_map_user_mem):
        if request.memtype != kgsl.KGSL_USER_MEM_TYPE_ADDR: raise NotImplementedError(f"KGSL user-memory type {request.memtype}")
        if request.hostptr == 0 or request.len <= 0: raise ValueError("invalid KGSL user-memory mapping")

        if request.hostptr in self.user_mappings:
            mapping = self.user_mappings[request.hostptr]
            if mapping.size != request.len: raise RuntimeError("repeat KGSL user-memory mapping")
            mapping.references += 1
        else:
            mapping = KGSLUserMapping(request.hostptr, request.hostptr, request.len)
            self.user_mappings(mapping.gpu_addr) = mapping
            self.gpu.map_range(mapping.gpu_addr, mapping.size, mapping.host_addr)
            self._track_signal_writes(mapping.host_addr, mapping.size)
        request.gpuaddr = mapping.gpu_addr

    def _sharedmem_free(self, request: kgsl.struct_kgsl_sharedmem_free):
        if request.gpuaddr not in self.user_mapping: raise RuntimeError(f"free of unknown KGSL user mapping {request.gpuaddr:#x}")
        mapping = self.user_mappings[request.gpuaddr]
        mapping.references -= 1
        if mapping.references == 0:
            self.user_mappings.pop(request.gpuaddr)
            self.gpu.unmap_range(mapping.gpu_addr, mapping.size)
            self._untrack_writes(mapping.host_addr, mapping.size)

    def _create_context(self, request: kgsl.struct_kgsl_drawctxt_create):
        context_id = self.next_context_id
        self.next_context_id += 1
        request.drawctxt_id = context_id
        self.contexts[context_id] = KGSLContext(context_id)

    def _destroy_context(self, context_id: int):
        if context_id not in self.contexts: raise RuntimeError(f"destroy of unknown KGSL context {context_id}")
        if any(submission.context_id == context_id and not submission.done for submission in self.submissions):
            raise RuntimeError(f"destroy of busy KGSL context {context_id}")
        self.contexts.pop(context_id)

    def _command_address(self, command: kgsl.struct_kgsl_command_object) -> int:
        if command.gpuaddr: base = command.gpuaddr 
        elif command.id in self.allocations and self.allocations[command.id].gpu_addr: base = self.allocations[command.id].gpu_addr
        else: raise RuntimeError("KGSL command object has neither a mapped GPU address nor a mapped object ID")
        return base + command.offset


    
        
