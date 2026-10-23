"""Process identities independent of wall-clock/psutil boot-time conversions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sys

import psutil


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    source: str
    started: str
    boot_id: str | None = None

    def to_dict(self):
        return asdict(self)


def process_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    if sys.platform == 'linux':
        try:
            # comm may contain spaces and parentheses; field 22 is starttime in
            # kernel ticks, not an epoch timestamp derived from cached boot time.
            fields = Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()
        except FileNotFoundError:
            raise psutil.NoSuchProcess(pid) from None
        except PermissionError:
            raise psutil.AccessDenied(pid) from None
        if fields[0] in {'Z', 'X'}:
            raise psutil.NoSuchProcess(pid)
        try:
            boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except OSError:
            boot_id = None
        return ProcessIdentity(pid, 'linux_start_ticks', fields[19], boot_id)
    process = psutil.Process(pid)
    if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
        raise psutil.NoSuchProcess(pid)
    return ProcessIdentity(pid, 'native_create_time', str(process.create_time()))


def process_matches(identity):
    """True=same process, False=gone/reused, None=identity cannot be verified."""
    if isinstance(identity, dict):
        identity = ProcessIdentity(**identity)
    try:
        if identity.source == 'native_create_time':
            # Also supports old checkpoint records. On Linux this fallback is
            # conservative cleanup only; new watchdogs always use kernel ticks.
            process = psutil.Process(identity.pid)
            return (process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                    and str(process.create_time()) == identity.started)
        current = process_identity(identity.pid)
        if current.source != identity.source or current.started != identity.started:
            return False
        if identity.boot_id and not current.boot_id:
            return None
        return not identity.boot_id or current.boot_id == identity.boot_id
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except psutil.AccessDenied:
        return None


def worker_record(pid):
    return dict(pid=pid, identity=process_identity(pid).to_dict())


def recorded_process_matches(record):
    identity = record.get('identity')
    if identity is None:  # Checkpoints written before stable identities.
        identity = ProcessIdentity(record['pid'], 'native_create_time', str(record['created']))
    elif identity['pid'] != record['pid']:
        return False
    return process_matches(identity)
