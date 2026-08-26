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
    flags: int
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
    retired_timestamp: int = 0

@dataclass(frozen=True)
class KGSLSubmission:
    context_id: int
    timestamp: int
    streams: tuple[CommandStream, ...]

    @property
    def done(self) -> bool: return all(stream.done for stream in self.streams)


class KGSLFileDesc(VirtFileDesc):
    def __init__(self, fd: int, driver: QCOMDriver):
        super().__init__(fd)
        self.driver = driver

    def ioctl(self, fd, request, argp): return self.driver.ioctl(request, argp)
    def mmap(self, start, size, prot, flags, fd, offset): return self.driver.mmap(start, size, prot, flags, offset)


class QCOMDriver(VirtDriver):
    def __init__(self, gpu: QCOMGPU|None=None):
        super().__init__()
        self.gpu = gpu or QCOMGPU(0)
        self.tracked_files.append(VirtFile("/dev/kgsl-3d0", functools.partial(KGSLFileDesc, driver=self)))

        self.next_fd, self.next_object_id, self.next_context_id = 1 << 30, 1, 1
        self.allocations: dict[int, KGSLAllocation] = {}
        self.user_mappings: dict[int, KGSLUserMapping] = {}
        self.contexts: dict[int, KGSLContext] = {}
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
        if object_id not in self.allocations: raise RuntimeError(f"mmap for unknown KGSL object {object_id}")
        allocation = self.allocations[object_id]
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
        if any(start == addr and end == addr + size for start, end, _, _ in self.tracked_addresses): return
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
        self.allocations[object_id] = KGSLAllocation(object_id, request.size, mmap_size, request.flags)

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
            if mapping.size != request.len: raise RuntimeError("conflicting repeated KGSL user-memory mapping")
            mapping.references += 1
        else:
            mapping = KGSLUserMapping(request.hostptr, request.hostptr, request.len)
            self.user_mappings[mapping.gpu_addr] = mapping
            self.gpu.map_range(mapping.gpu_addr, mapping.size, mapping.host_addr)
            self._track_signal_writes(mapping.host_addr, mapping.size)
        request.gpuaddr = mapping.gpu_addr

    def _sharedmem_free(self, request: kgsl.struct_kgsl_sharedmem_free):
        if request.gpuaddr not in self.user_mappings: raise RuntimeError(f"free of unknown KGSL user mapping {request.gpuaddr:#x}")
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

    def _gpu_command(self, request: kgsl.struct_kgsl_gpu_command):
        if request.context_id not in self.contexts: raise RuntimeError(f"submission for unknown KGSL context {request.context_id}")
        if request.numsyncs: raise NotImplementedError("KGSL command syncpoints")
        if request.numcmds <= 0 or request.cmdlist == 0: raise ValueError("KGSL submission has no command buffers")
        command_size = ctypes.sizeof(kgsl.struct_kgsl_command_object)
        if request.cmdsize < command_size: raise RuntimeError(f"KGSL command stride {request.cmdsize}, expected at least {command_size}")

        commands, addresses = [], []
        for index in range(request.numcmds):
            command = kgsl.struct_kgsl_command_object.from_address(request.cmdlist + index * request.cmdsize)
            if not command.flags & kgsl.KGSL_CMDLIST_IB: raise NotImplementedError(f"KGSL command flags {command.flags:#x}")
            if command.size <= 0 or command.size & 3: raise ValueError("KGSL indirect-buffer size must be a positive multiple of four")
            address = self._command_address(command)
            self.gpu.translate_addr(address, command.size)
            commands.append(command)
            addresses.append(address)

        context = self.contexts[request.context_id]
        context.submitted_timestamp = (context.submitted_timestamp + 1) & MASK32
        if context.submitted_timestamp == 0: context.submitted_timestamp = 1
        request.timestamp = context.submitted_timestamp
        streams = tuple(self.gpu.submit(address, command.size) for address, command in zip(addresses, commands))
        self.submissions.append(KGSLSubmission(request.context_id, request.timestamp, streams))
        self._emulate_execute()

    def _retire_completed_submissions(self):
        while self.submissions and self.submissions[0].done:
            submission = self.submissions.pop(0)
            self.contexts[submission.context_id].retired_timestamp = submission.timestamp

    def _emulate_execute(self):
        if self._executing: return
        self._executing = True
        try:
            self.gpu.execute_pending()
            self._retire_completed_submissions()
        finally:
            self._executing = False

    def _wait_timestamp(self, request: kgsl.struct_kgsl_device_waittimestamp_ctxtid):
        if request.context_id not in self.contexts: raise RuntimeError(f"wait for unknown KGSL context {request.context_id}")
        context = self.contexts[request.context_id]
        if request.timestamp > context.submitted_timestamp: raise RuntimeError(f"wait for unsubmitted KGSL timestamp {request.timestamp}")
        self._emulate_execute()

    def _get_property(self, request: kgsl.struct_kgsl_device_getproperty):
        if request.type != kgsl.KGSL_PROP_DEVICE_INFO: raise NotImplementedError(f"KGSL property {request.type:#x}")
        if not request.value or request.sizebytes < ctypes.sizeof(kgsl.struct_kgsl_devinfo): raise ValueError("invalid KGSL device-info buffer")
        info = kgsl.struct_kgsl_devinfo.from_address(request.value)
        info.device_id = kgsl.KGSL_DEVICE_3D0
        info.chip_id = A630_CHIP_ID
        info.mmu_enabled = 1
        info.gmem_gpubaseaddr = 0
        info.gpu_id = 630
        info.gmem_sizebytes = A630_GMEM_SIZE

    def ioctl(self, request:int, argp:int) -> int:
        nr = request & 0xff
        if nr == 0x02:
            self._get_property(kgsl.struct_kgsl_device_getproperty.from_address(argp))
        elif nr == 0x07:
            self._wait_timestamp(kgsl.struct_kgsl_device_waittimestamp_ctxtid.from_address(argp))
        elif nr == 0x13:
            self._create_context(kgsl.struct_kgsl_drawctxt_create.from_address(argp))
        elif nr == 0x14:
            self._destroy_context(kgsl.struct_kgsl_drawctxt_destroy.from_address(argp).drawctxt_id)
        elif nr == 0x15:
            self._map_user_mem(kgsl.struct_kgsl_map_user_mem.from_address(argp))
        elif nr == 0x21:
            self._sharedmem_free(kgsl.struct_kgsl_sharedmem_free.from_address(argp))
        elif nr == 0x32:
            prop = kgsl.struct_kgsl_device_getproperty.from_address(argp)
            if prop.type != kgsl.KGSL_PROP_PWR_CONSTRAINT: raise NotImplementedError(f"KGSL set-property {prop.type:#x}")
        elif nr == 0x45:
            self._gpuobj_alloc(kgsl.struct_kgsl_gpuobj_alloc.from_address(argp))
        elif nr == 0x46:
            self._gpuobj_free(kgsl.struct_kgsl_gpuobj_free.from_address(argp))
        elif nr == 0x4a:
            self._gpu_command(kgsl.struct_kgsl_gpu_command.from_address(argp))
        else:
            raise NotImplementedError(f"unknown KGSL ioctl number {nr:#x} (request {request:#x})")
        return 0

