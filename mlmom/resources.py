"""Non-privileged process supervision and resource-aware admission control."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
import psutil
from .persistence import atomic_json
from .pipeline_contracts import BudgetSnapshot, RunConfig


def capacity(config: RunConfig):
    cpu = len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(),'cpu_affinity') else (os.cpu_count() or 1)
    memory = psutil.virtual_memory().total
    for p in ['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes']:
        try:
            text = Path(p).read_text().strip()
            if text != 'max': memory = min(memory,int(text))
        except (OSError,ValueError): pass
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max': cpu = min(cpu,max(1,int(int(quota)/int(period))))
    except (OSError,ValueError): pass
    return max(1,min(8,cpu)-config.reserve_cpus), max(.25,min(32.,memory/1024**3)-config.reserve_memory_gb)


class Budget:
    def __init__(self, config: RunConfig, started: float, *, deadline=None):
        self.config = config; self.started = started
        self.deadline = deadline or started+config.total_seconds
        self.reserve = config.final_reserve_seconds
        self.cpus, self.memory_gb = capacity(config)
        self.monotonic_start = time.monotonic()
        self.wall_start = time.time()

    def now(self):
        return self.wall_start+(time.monotonic()-self.monotonic_start)

    def snapshot(self):
        remaining = max(0.,self.deadline-self.now())
        return BudgetSnapshot(remaining_seconds=remaining,research_seconds=max(0.,remaining-self.reserve),
            available_cpus=self.cpus,available_memory_gb=self.memory_gb,spent_seconds=max(0.,self.now()-self.started))


def terminate_tree(pid: int):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for process in reversed(children):
            try: process.kill()
            except psutil.Error: pass
        try: parent.kill()
        except psutil.Error: pass
        psutil.wait_procs(children+[parent],timeout=1)
    except psutil.Error: pass


@dataclass
class Job:
    process: subprocess.Popen
    directory: Path
    started: float
    seconds: float
    memory_gb: float
    cpus: int
    stdout: object
    stderr: object
    peak_memory_gb: float = 0.
    reason: str | None = None
    descendants: set = None
    cpu_ids: list = None

    def poll(self):
        try:
            parent = psutil.Process(self.process.pid)
            children = parent.children(recursive=True)
            self.descendants.update((p.pid,p.create_time()) for p in children)
            rss = sum(p.memory_info().rss for p in [parent]+children if p.is_running())/1024**3
            self.peak_memory_gb = max(self.peak_memory_gb,rss)
            if rss > self.memory_gb: self.reason = 'memory_limit'
        except psutil.Error: pass
        if time.monotonic()-self.started > self.seconds: self.reason = 'time_limit'
        if self.reason and self.process.poll() is None: terminate_tree(self.process.pid)
        return self.process.poll()

    def finish(self):
        self.process.wait()
        self.stdout.close(); self.stderr.close()
        for pid,created in self.descendants:
            try:
                process = psutil.Process(pid)
                if process.create_time() == created: terminate_tree(pid)
            except psutil.Error: pass
        result_path = self.directory/'result.json'
        limit_path = self.directory/'limit.json'
        if limit_path.exists(): self.reason=json.loads(limit_path.read_text(encoding='utf-8'))['reason']
        result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
        if self.reason or self.process.returncode != 0 or not result:
            result = dict(result,status='failed',stop_reason=self.reason or 'worker_error',
                          error=result.get('error') or (self.directory/'stderr.log').read_text(encoding='utf-8',errors='replace')[-6000:])
        result.update(duration_seconds=time.monotonic()-self.started,peak_memory_gb=self.peak_memory_gb)
        return result


class SandboxManager:
    def __init__(self): self.jobs = []
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def start(self, request: dict, directory: Path, *, seconds: float, memory_gb: float, cpus: int):
        directory.mkdir(parents=True,exist_ok=True)
        for name in ['result.json','limit.json']:
            (directory/name).unlink(missing_ok=True)
        request = dict(request, parent_pid=os.getpid(),parent_created=psutil.Process().create_time(),
                       worker_seconds=seconds,worker_cpus=cpus,worker_memory_gb=memory_gb)
        atomic_json(directory/'request.json',request)
        root = Path(__file__).resolve().parent.parent
        allowed={'PATH','SystemRoot','SYSTEMROOT','WINDIR','COMSPEC','PATHEXT','LD_LIBRARY_PATH'}
        if os.name=='nt':allowed|={'SystemDrive','SYSTEMDRIVE','ProgramData','PROGRAMDATA','HOMEDRIVE','HOMEPATH','LOCALAPPDATA','APPDATA'}
        env = {k:v for k,v in os.environ.items() if k in allowed}
        env.update(PYTHONPATH=str(root),PYTHONIOENCODING='utf-8',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
            HOME=str(directory),USERPROFILE=str(directory),TEMP=str(directory),TMP=str(directory),
            CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            LANGCHAIN_TRACING_V2='false',LANGSMITH_TRACING='false')
        if os.name=='nt' and 'USERPROFILE' in os.environ:env['USERPROFILE']=os.environ['USERPROFILE']
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
            env[key] = str(cpus)
        out = (directory/'stdout.log').open('w',encoding='utf-8')
        err = (directory/'stderr.log').open('w',encoding='utf-8')
        kwargs = {'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {'start_new_session':True}
        process = subprocess.Popen([sys.executable,'-m','mlmom.worker',str(directory/'request.json')],
            cwd=directory,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,**kwargs)
        selected=[]
        try:
            parent = psutil.Process()
            if hasattr(parent,'cpu_affinity'):
                available = parent.cpu_affinity()
                used={cpu for job in self.jobs if job.process.poll() is None for cpu in (job.cpu_ids or [])}
                free=[cpu for cpu in available if cpu not in used]
                if len(free)<cpus:
                    terminate_tree(process.pid)
                    raise ValueError('No disjoint CPU affinity allocation remains')
                selected=free[-cpus:]
                psutil.Process(process.pid).cpu_affinity(selected)
        except psutil.Error: pass
        job = Job(process,directory,time.monotonic(),seconds,memory_gb,cpus,out,err,descendants=set(),cpu_ids=selected)
        self.jobs.append(job)
        return job

    def run(self, request, directory, **limits):
        job = self.start(request,directory,**limits)
        while job.poll() is None: time.sleep(.05)
        return job.finish()

    def close(self):
        for job in self.jobs:
            if job.process.poll() is None: terminate_tree(job.process.pid)
            if not job.stdout.closed: job.stdout.close()
            if not job.stderr.closed: job.stderr.close()
