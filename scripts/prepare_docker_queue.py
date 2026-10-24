"""Prepare fresh inputs and a frozen runtime without copying API credentials."""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlmom.persistence import atomic_json, file_hash
from mlmom.pipeline_contracts import RunConfig
from scripts.package_solution import package


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--downloads', type=Path, default=Path.home() / 'Downloads')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / 'runtime'
    runtime.mkdir()
    for folder in ['mlmom', 'skills', 'profiles', 'scripts']:
        shutil.copytree(ROOT / folder, runtime / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['solution.py', 'requirements.txt']:
        shutil.copyfile(ROOT / name, runtime / name)
    package(ROOT, output / 'mlmom-solution.zip')
    inputs = []
    for name, filename in [
        ('regression', 'Регрессия-20261020T130119Z-1-001.zip'),
        ('multiclass', 'Многоклассовая классификация-20261020T130139Z-1-001.zip'),
    ]:
        archive_path = args.downloads / filename
        task = output / name / 'task'
        task.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            # Both supplied archives have one enclosing directory; preserve file bytes.
            for member in archive.infolist():
                if member.is_dir():
                    continue
                parts = Path(member.filename).parts
                if len(parts) != 2 or parts[-1] in {'.', '..'}:
                    raise ValueError('Unexpected organizer archive layout')
                (task / parts[-1]).write_bytes(archive.read(member))
        with zipfile.ZipFile(output / 'mlmom-solution.zip') as archive:
            for name_in_zip in archive.namelist():
                if '/' in name_in_zip or '\\' in name_in_zip:
                    raise ValueError('Solution ZIP must be flat')
                (task / name_in_zip).write_bytes(archive.read(name_in_zip))
        inputs.append(dict(case=name, archive=filename, archive_sha256=file_hash(archive_path),
                           task_hashes={p.name: file_hash(p) for p in task.iterdir() if p.is_file()}))
    config = RunConfig(timeouts_enabled=False, profile='platform', total_seconds=3540,
        final_reserve_seconds=600, first_submission_seconds=600, experiment_seconds=810,
        initial_experiment_seconds=120, coding_seconds=300, prediction_seconds=45,
        smoke_seconds=30, max_rounds=None, plateau_rounds=None, cycle_full_refit=True,
        target_completed_rounds=0, fixed_outer_split=False,
        initial_boosting_sweep=True, max_workers=3, reserve_cpus=2, reserve_memory_gb=2, llm_timeout_seconds=300,
        hypothesis_seconds=210, planner_seconds=210, validator_seconds=210)
    atomic_json(output / 'config.json', config)
    atomic_json(output / 'input_manifest.json', dict(inputs=inputs,
        solution_zip_sha256=file_hash(output / 'mlmom-solution.zip'),
        runtime_hashes={p.relative_to(runtime).as_posix(): file_hash(p)
                        for p in runtime.rglob('*') if p.is_file()},
        seconds_per_task=3600, order=['regression', 'multiclass']))
    print(json.dumps(dict(output=str(output), runtime_files=len(list(runtime.rglob('*.py'))),
                          solution_zip_bytes=(output / 'mlmom-solution.zip').stat().st_size)))


if __name__ == '__main__':
    main()
