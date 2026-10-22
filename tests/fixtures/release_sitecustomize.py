"""External offline test instrumentation; never included in the submission ZIP."""
import os
from pathlib import Path
import sys

if os.getenv('MLMOM_RELEASE_OFFLINE')=='1' and Path(sys.argv[0]).name=='solution.py':
    sys.path.insert(0,os.getcwd())
    import solution  # Bind production modules to the extracted ZIP first.
    import mlmom.pipeline as pipeline
    from tests.fakes import OfflineLLM
    original=pipeline.Pipeline

    class OfflinePipeline(original):
        def __init__(self,*args,**kwargs):
            kwargs['client']=OfflineLLM(seconds=60,cpus=2)
            super().__init__(*args,**kwargs)

    pipeline.Pipeline=OfflinePipeline
