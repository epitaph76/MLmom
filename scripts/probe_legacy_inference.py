"""Verify a historical portable model with a new archive, without editing history."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', default='mlmom-ml:test')
    args = parser.parse_args()
    source, output = args.legacy.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('code', 'model'):
        shutil.copytree(source / name, output / name)
    for name in ('features.csv', 'reference.parquet', 'verify.py'):
        shutil.copyfile(source / name, output / name)
    settings = json.loads((source / 'settings.json').read_text(encoding='utf-8'))
    legacy_hash = settings['archive_sha256']
    archive_hash = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    settings['archive_sha256'] = archive_hash
    (output / 'settings.json').write_text(json.dumps(settings), encoding='utf-8')
    with zipfile.ZipFile(args.archive) as archive:
        for name in ('solution.py', 'mlmom_bundle.py', 'requirements.txt'):
            (output / name).write_bytes(archive.read(name))
    result = subprocess.run([
        'docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python',
        '--mount', f'type=bind,source={output},target=/portable',
        '--workdir', '/portable', args.image, 'verify.py',
    ], capture_output=True, text=True)
    (output / 'verification.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    report = dict(status='failed', archive_sha256=archive_hash,
                  legacy_archive_sha256=legacy_hash, exit_code=result.returncode,
                  network_enabled=False, original_artifacts_mounted=False)
    if result.returncode == 0:
        report.update(json.loads((output / 'report.json').read_text(encoding='utf-8')))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
