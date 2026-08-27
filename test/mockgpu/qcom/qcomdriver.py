from __future__ import annotations

import ctypes, functools, mmap, threading, time
from dataclasses import dataclass, field

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
    owner_fd: int
    host_addr: int = 0
    gpu_addr: int = 0

@dataclass
class KGSLUserMapping:
    host_addr: int
    gpu_addr: int
    size: int
    references: int = 1
    owners: dict[int, int] = field(default_factory=dict)

@dataclass
class KGSLContext:
    context_id: int
    owner_fd: int
    submitted_timestamp: int = 0
    retired_timestamp: int = 0
    failures: dict[int, Exception] = field(default_factory=dict)

@dataclass
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

    def ioctl(self, fd, request, argp): return self.driver.ioctl(request, argp, fd)
    def mmap(self, start, size, prot, flags, fd, offset): return self.driver.mmap(start, size, prot, flags, offset, fd)
    def close(self, fd): return self.driver.close(fd)


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
        self.open_fds: set[int] = set()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._executing = False

    def _alloc_fd(self) -> int:
        fd = self.next_fd
        self.next_fd += 1
        return fd

    def open(self, name, flags, mode, virtfile):
        with self._lock:
            fd = self._alloc_fd()
            self.open_fds.add(fd)
            return virtfile.fdcls(fd)

    def _require_fd(self, fd: int|None) -> int:
        if fd is None:
            if len(self.open_fds) != 1: raise RuntimeError("KGSL operation requires an open file descriptor")
            fd = next(iter(self.open_fds))
        if fd not in self.open_fds: raise RuntimeError(f"operation on closed or unknown KGSL file descriptor {fd}")
        return fd

    def _remove_submission(self, submission: KGSLSubmission, error: Exception|None=None):
        if submission in self.submissions: self.submissions.remove(submission)
        self.gpu.pending[:] = [stream for stream in self.gpu.pending if all(stream is not owned for owned in submission.streams)]
        if error is not None and submission.context_id in self.contexts:
            self.contexts[submission.context_id].failures[submission.timestamp] = error

    def _release_user_references(self, mapping: KGSLUserMapping, fd: int, count: int=1):
        owned = mapping.owners.get(fd, 0)
        if count <= 0 or owned < count: raise RuntimeError(f"KGSL file descriptor {fd} does not own mapping {mapping.gpu_addr:#x}")
        mapping.references -= count
        if owned == count: mapping.owners.pop(fd)
        else: mapping.owners[fd] = owned - count
        if mapping.references == 0:
            self.user_mappings.pop(mapping.gpu_addr)
            self.gpu.unmap_range(mapping.gpu_addr, mapping.size)
            self._untrack_writes(mapping.host_addr, mapping.size)

    def close(self, fd: int) -> int:
        with self._condition:
            self._require_fd(fd)
            owned_contexts = {context_id for context_id, context in self.contexts.items() if context.owner_fd == fd}
            for submission in tuple(self.submissions):
                if submission.context_id in owned_contexts: self._remove_submission(submission)
            for context_id in owned_contexts: self.contexts.pop(context_id)
            for object_id, allocation in tuple(self.allocations.items()):
                if allocation.owner_fd != fd: continue
                self.allocations.pop(object_id)
                if allocation.gpu_addr:
                    self.gpu.unmap_range(allocation.gpu_addr, allocation.mmap_size)
                    self._untrack_writes(allocation.host_addr, allocation.mmap_size)
                    libc.munmap(allocation.host_addr, allocation.mmap_size)
            for mapping in tuple(self.user_mappings.values()):
                if (count:=mapping.owners.get(fd, 0)): self._release_user_references(mapping, fd, count)
            self.open_fds.remove(fd)
            self._condition.notify_all()
            return 0

    def mmap(self, start: int, size: int, prot: int, flags: int, offset: int, fd: int|None=None) -> int:
        with self._lock:
            fd = self._require_fd(fd)
            if offset & 0xfff: raise RuntimeError(f"unaligned KGSL mmap offset {offset:#x}")
            object_id = offset >> 12
            if object_id not in self.allocations: raise RuntimeError(f"mmap for unknown KGSL object {object_id}")
            allocation = self.allocations[object_id]
            if allocation.owner_fd != fd: raise RuntimeError(f"KGSL object {object_id} belongs to another file descriptor")
            if allocation.host_addr: raise RuntimeError(f"KGSL object {object_id} was already mapped")
            if size != allocation.mmap_size: raise RuntimeError(f"KGSL object {object_id} mmap size {size:#x}, expected {allocation.mmap_size:#x}")

            host_addr = libc.mmap(start, size, prot, flags | mmap.MAP_ANONYMOUS, -1, 0)
            if host_addr == ctypes.c_void_p(-1).value: raise OSError("anonymous mmap for KGSL object failed")
            try: self.gpu.map_range(host_addr, size, host_addr)
            except Exception:
                libc.munmap(host_addr, size)
                raise
            allocation.host_addr = allocation.gpu_addr = host_addr
            cache_mode = (allocation.flags & kgsl.KGSL_CACHEMODE_MASK) >> kgsl.KGSL_CACHEMODE_SHIFT
            if cache_mode == kgsl.KGSL_CACHEMODE_UNCACHED: self._track_signal_writes(host_addr, size)
            return host_addr

    def _track_signal_writes(self, addr: int, size: int):
        if any(start == addr and end == addr + size for start, end, _, _ in self.tracked_addresses): return
        self.track_address(addr, addr + size, lambda mv, off: None, lambda mv, off: self._emulate_execute())

    def _untrack_writes(self, addr: int, size: int):
        self.tracked_addresses[:] = [entry for entry in self.tracked_addresses if entry[0:2] != (addr, addr+size)]

    def _gpuobj_alloc(self, request: kgsl.struct_kgsl_gpuobj_alloc, fd: int):
        if request.size <= 0: raise ValueError("KGSL GPU object size must be positive")
        if request.metadata_len > kgsl.KGSL_GPUOBJ_ALLOC_METADATA_MAX: raise ValueError("KGSL GPU object metadata is too large")
        if request.metadata_len and not request.metadata: raise ValueError("KGSL GPU object metadata has no address")
        if request.va_len and request.va_len < request.size: raise ValueError("KGSL GPU object virtual range is smaller than its allocation")
        object_id = self.next_object_id
        self.next_object_id += 1
        mmap_size = (request.size + 0xfff) & ~0xfff
        request.id, request.mmapsize = object_id, mmap_size
        self.allocations[object_id] = KGSLAllocation(object_id, request.size, mmap_size, request.flags, fd)

    def _gpuobj_free(self, request: kgsl.struct_kgsl_gpuobj_free, fd: int):
        if request.flags: raise NotImplementedError("deferred KGSL GPU object free")
        if request.id not in self.allocations: raise RuntimeError(f"free of unknown KGSL object {request.id}")
        allocation = self.allocations[request.id]
        if allocation.owner_fd != fd: raise RuntimeError(f"KGSL object {request.id} belongs to another file descriptor")
        self.allocations.pop(request.id)
        if allocation.gpu_addr:
            self.gpu.unmap_range(allocation.gpu_addr, allocation.mmap_size)
            self._untrack_writes(allocation.host_addr, allocation.mmap_size)

    def _map_user_mem(self, request: kgsl.struct_kgsl_map_user_mem, fd: int):
        if request.memtype != kgsl.KGSL_USER_MEM_TYPE_ADDR: raise NotImplementedError(f"KGSL user-memory type {request.memtype}")
        if request.hostptr == 0 or request.len <= 0: raise ValueError("invalid KGSL user-memory mapping")

        if request.hostptr in self.user_mappings:
            mapping = self.user_mappings[request.hostptr]
            if mapping.size != request.len: raise RuntimeError("conflicting repeated KGSL user-memory mapping")
            mapping.references += 1
            mapping.owners[fd] = mapping.owners.get(fd, 0) + 1
        else:
            mapping = KGSLUserMapping(request.hostptr, request.hostptr, request.len, owners={fd: 1})
            self.gpu.map_range(mapping.gpu_addr, mapping.size, mapping.host_addr)
            self.user_mappings[mapping.gpu_addr] = mapping
            self._track_signal_writes(mapping.host_addr, mapping.size)
        request.gpuaddr = mapping.gpu_addr

    def _sharedmem_free(self, request: kgsl.struct_kgsl_sharedmem_free, fd: int):
        if request.gpuaddr not in self.user_mappings: raise RuntimeError(f"free of unknown KGSL user mapping {request.gpuaddr:#x}")
        self._release_user_references(self.user_mappings[request.gpuaddr], fd)

    def _create_context(self, request: kgsl.struct_kgsl_drawctxt_create, fd: int):
        context_id = self.next_context_id
        self.next_context_id += 1
        request.drawctxt_id = context_id
        self.contexts[context_id] = KGSLContext(context_id, fd)

    def _destroy_context(self, context_id: int, fd: int):
        if context_id not in self.contexts: raise RuntimeError(f"destroy of unknown KGSL context {context_id}")
        if self.contexts[context_id].owner_fd != fd: raise RuntimeError(f"KGSL context {context_id} belongs to another file descriptor")
        if any(submission.context_id == context_id and not submission.done for submission in self.submissions):
            raise RuntimeError(f"destroy of busy KGSL context {context_id}")
        self.contexts.pop(context_id)

    def _command_address(self, command: kgsl.struct_kgsl_command_object, fd: int) -> int:
        if command.offset + command.size > 1 << 64:
            raise ValueError("KGSL command object range overflows 64 bits")
        if command.gpuaddr: base = command.gpuaddr
        elif command.id in self.allocations and self.allocations[command.id].gpu_addr:
            allocation = self.allocations[command.id]
            if allocation.owner_fd != fd: raise RuntimeError(f"KGSL object {command.id} belongs to another file descriptor")
            base = allocation.gpu_addr
        else: raise RuntimeError("KGSL command object has neither a mapped GPU address nor a mapped object ID")
        if base + command.offset + command.size > 1 << 64: raise ValueError("KGSL command GPU address overflows 64 bits")
        address = base + command.offset
        owned = False
        for allocation in self.allocations.values():
            if allocation.gpu_addr and allocation.gpu_addr <= address and address + command.size <= allocation.gpu_addr + allocation.mmap_size:
                if allocation.owner_fd != fd: raise RuntimeError("KGSL command buffer belongs to another file descriptor")
                owned = True
                break
        else:
            for mapping in self.user_mappings.values():
                if mapping.gpu_addr <= address and address + command.size <= mapping.gpu_addr + mapping.size:
                    if fd not in mapping.owners: raise RuntimeError("KGSL command buffer belongs to another file descriptor")
                    owned = True
                    break
        if not owned: raise RuntimeError("KGSL command buffer is not owned by this driver")
        return address

    def _gpu_command(self, request: kgsl.struct_kgsl_gpu_command, fd: int):
        if request.context_id not in self.contexts: raise RuntimeError(f"submission for unknown KGSL context {request.context_id}")
        context = self.contexts[request.context_id]
        if context.owner_fd != fd: raise RuntimeError(f"KGSL context {request.context_id} belongs to another file descriptor")
        if request.flags: raise NotImplementedError(f"KGSL submission flags {request.flags:#x}")
        if request.numsyncs: raise NotImplementedError("KGSL command syncpoints")
        if request.numobjs: raise NotImplementedError("KGSL submission object lists")
        if request.synclist or request.syncsize: raise ValueError("KGSL submission has a sync list without syncpoints")
        if request.objlist or request.objsize: raise ValueError("KGSL submission has an object list without objects")
        if request.numcmds <= 0 or request.cmdlist == 0: raise ValueError("KGSL submission has no command buffers")
        if request.numcmds > 4096: raise ValueError("KGSL submission has too many command buffers")
        command_size = ctypes.sizeof(kgsl.struct_kgsl_command_object)
        if request.cmdsize < command_size: raise RuntimeError(f"KGSL command stride {request.cmdsize}, expected at least {command_size}")

        commands, addresses = [], []
        for index in range(request.numcmds):
            command = kgsl.struct_kgsl_command_object.from_address(request.cmdlist + index * request.cmdsize)
            if not command.flags & kgsl.KGSL_CMDLIST_IB: raise NotImplementedError(f"KGSL command flags {command.flags:#x}")
            if command.flags & ~kgsl.KGSL_CMDLIST_IB: raise NotImplementedError(f"KGSL command flags {command.flags:#x}")
            if command.size <= 0 or command.size & 3: raise ValueError("KGSL indirect-buffer size must be a positive multiple of four")
            address = self._command_address(command, fd)
            self.gpu.translate_addr(address, command.size)
            commands.append(command)
            addresses.append(address)

        previous_timestamp = context.submitted_timestamp
        context.submitted_timestamp = (context.submitted_timestamp + 1) & MASK32
        if context.submitted_timestamp == 0: context.submitted_timestamp = 1
        request.timestamp = context.submitted_timestamp
        streams: list[CommandStream] = []
        submission: KGSLSubmission|None = None
        try:
            for address, command in zip(addresses, commands): streams.append(self.gpu.submit(address, command.size))
            self.submissions.append(submission:=KGSLSubmission(request.context_id, request.timestamp, tuple(streams)))
            self._emulate_execute()
        except Exception:
            if submission is not None: self._remove_submission(submission)
            else: self.gpu.pending[:] = [stream for stream in self.gpu.pending if all(stream is not owned for owned in streams)]
            context.submitted_timestamp = previous_timestamp
            context.failures.pop(request.timestamp, None)
            request.timestamp = 0
            raise

    def _retire_completed_submissions(self):
        while self.submissions and self.submissions[0].done:
            submission = self.submissions.pop(0)
            if submission.context_id in self.contexts: self.contexts[submission.context_id].retired_timestamp = submission.timestamp

    def _emulate_execute(self):
        with self._condition:
            if self._executing: return
            self._executing = True
            try: self.gpu.execute_pending()
            except Exception as error:
                failed_stream = self.gpu.pending[0] if self.gpu.pending else None
                if failed_stream is not None:
                    if (submission:=next((x for x in self.submissions if any(failed_stream is stream for stream in x.streams)), None)) is not None:
                        self._remove_submission(submission, error)
                    else: self.gpu.pending.pop(0)
                raise
            finally:
                self._retire_completed_submissions()
                self._executing = False
                self._condition.notify_all()

    @staticmethod
    def _timestamp_after(first: int, second: int) -> bool: return 0 < ((first - second) & MASK32) < (1 << 31)

    def _wait_timestamp(self, request: kgsl.struct_kgsl_device_waittimestamp_ctxtid, fd: int):
        if request.context_id not in self.contexts: raise RuntimeError(f"wait for unknown KGSL context {request.context_id}")
        context = self.contexts[request.context_id]
        if context.owner_fd != fd: raise RuntimeError(f"KGSL context {request.context_id} belongs to another file descriptor")
        if self._timestamp_after(request.timestamp, context.submitted_timestamp):
            raise RuntimeError(f"wait for unsubmitted KGSL timestamp {request.timestamp}")
        if request.timestamp == 0 or not self._timestamp_after(request.timestamp, context.retired_timestamp): return
        deadline = None if request.timeout == MASK32 else time.monotonic() + request.timeout / 1000
        while self._timestamp_after(request.timestamp, context.retired_timestamp):
            if fd not in self.open_fds or request.context_id not in self.contexts:
                raise RuntimeError(f"KGSL context {request.context_id} was closed while waiting")
            if (error:=context.failures.get(request.timestamp)) is not None:
                raise RuntimeError(f"KGSL submission {request.context_id}:{request.timestamp} failed") from error
            self._emulate_execute()
            if not self._timestamp_after(request.timestamp, context.retired_timestamp): return
            if request.timeout == 0 or (deadline is not None and (remaining:=deadline-time.monotonic()) <= 0):
                raise TimeoutError(f"timeout waiting for KGSL timestamp {request.context_id}:{request.timestamp}")
            self._condition.wait(0.01 if deadline is None else min(0.01, remaining))

    def _get_property(self, request: kgsl.struct_kgsl_device_getproperty):
        if request.type != kgsl.KGSL_PROP_DEVICE_INFO: raise NotImplementedError(f"KGSL property {request.type:#x}")
        if not request.value or request.sizebytes < ctypes.sizeof(kgsl.struct_kgsl_devinfo): raise ValueError("invalid KGSL device-info buffer")
        info_address = ctypes.cast(request.value, ctypes.c_void_p).value
        assert info_address is not None
        info = kgsl.struct_kgsl_devinfo.from_address(info_address)
        info.device_id = kgsl.KGSL_DEVICE_3D0
        info.chip_id = A630_CHIP_ID
        info.mmu_enabled = 1
        info.gmem_gpubaseaddr = 0
        info.gpu_id = 630
        info.gmem_sizebytes = A630_GMEM_SIZE

    def ioctl(self, request:int, argp:int, fd:int|None=None) -> int:
        with self._condition:
            fd = self._require_fd(fd)
            if not argp: raise ValueError("KGSL ioctl has a null payload")
            if request > 0xff and ((request >> 8) & 0xff) != kgsl.KGSL_IOC_TYPE: raise ValueError(f"invalid KGSL ioctl type in request {request:#x}")
            expected_types: dict[int, type] = {
              0x02: kgsl.struct_kgsl_device_getproperty, 0x07: kgsl.struct_kgsl_device_waittimestamp_ctxtid,
              0x13: kgsl.struct_kgsl_drawctxt_create, 0x14: kgsl.struct_kgsl_drawctxt_destroy,
              0x15: kgsl.struct_kgsl_map_user_mem, 0x21: kgsl.struct_kgsl_sharedmem_free,
              0x32: kgsl.struct_kgsl_device_getproperty, 0x45: kgsl.struct_kgsl_gpuobj_alloc,
              0x46: kgsl.struct_kgsl_gpuobj_free, 0x4a: kgsl.struct_kgsl_gpu_command}
            return self._ioctl(request, argp, fd, expected_types)

    def _ioctl(self, request: int, argp: int, fd: int, expected_types: dict[int, type]) -> int:
        nr = request & 0xff
        if request > 0xffff and nr in expected_types and (encoded_size:=(request >> 16) & 0x3fff) != ctypes.sizeof(expected_types[nr]):
            raise ValueError(f"invalid KGSL ioctl payload size {encoded_size} for request {nr:#x}")
        if nr == 0x02:
            self._get_property(kgsl.struct_kgsl_device_getproperty.from_address(argp))
        elif nr == 0x07:
            self._wait_timestamp(kgsl.struct_kgsl_device_waittimestamp_ctxtid.from_address(argp), fd)
        elif nr == 0x13:
            self._create_context(kgsl.struct_kgsl_drawctxt_create.from_address(argp), fd)
        elif nr == 0x14:
            self._destroy_context(kgsl.struct_kgsl_drawctxt_destroy.from_address(argp).drawctxt_id, fd)
        elif nr == 0x15:
            self._map_user_mem(kgsl.struct_kgsl_map_user_mem.from_address(argp), fd)
        elif nr == 0x21:
            self._sharedmem_free(kgsl.struct_kgsl_sharedmem_free.from_address(argp), fd)
        elif nr == 0x32:
            prop = kgsl.struct_kgsl_device_getproperty.from_address(argp)
            if prop.type != kgsl.KGSL_PROP_PWR_CONSTRAINT: raise NotImplementedError(f"KGSL set-property {prop.type:#x}")
        elif nr == 0x45:
            self._gpuobj_alloc(kgsl.struct_kgsl_gpuobj_alloc.from_address(argp), fd)
        elif nr == 0x46:
            self._gpuobj_free(kgsl.struct_kgsl_gpuobj_free.from_address(argp), fd)
        elif nr == 0x4a:
            self._gpu_command(kgsl.struct_kgsl_gpu_command.from_address(argp), fd)
        else:
            raise NotImplementedError(f"unknown KGSL ioctl number {nr:#x} (request {request:#x})")
        return 0
