"""Pass parsed dotenv credentials through the process environment, never argv."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--image', default='mlmom-ml:test')
    parser.add_argument('--seconds', type=int, default=0, help='0 disables the external task deadline')
    parser.add_argument('--model', default='deepseek/deepseek-v4.1-flash-1017')
    args = parser.parse_args()
    environment = os.environ.copy()
    local = dotenv_values(ROOT / '.env')
    for name in ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL']:
        value = local.get(name) or environment.get(name)
        if not value:
            raise ValueError(f'Missing {name}')
        environment[name] = value
    environment['OPENAI_MODEL'] = args.model
    if environment['OPENAI_BASE_URL'].rstrip('/') != 'https://routerai.ru/api/v1':
        raise ValueError('The requested run must use the configured RouterAI endpoint')
    output = args.output.resolve(strict=True)
    command = ['docker', 'run', '-d', '--name', args.name, '--cpus', '8',
        '--memory', '6g', '--memory-swap', '6g', '--pids-limit', '512', '--init',
        '--env', 'OPENAI_API_KEY', '--env', 'OPENAI_BASE_URL', '--env', 'OPENAI_MODEL',
        '--env', 'OPENBLAS_NUM_THREADS=1', '--env', 'OMP_NUM_THREADS=4',
        '--env', 'MKL_NUM_THREADS=4', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--mount', f'type=bind,source={output},target=/work',
        '--workdir', '/work/runtime', '--entrypoint', 'python', args.image,
        '-u', 'scripts/run_docker_queue.py', '--output', '/work', '--seconds', str(args.seconds)]
    result = subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
    print(json.dumps(dict(container=args.name, id=result.stdout.strip(), output=str(output),
        model=environment['OPENAI_MODEL'], base_url=environment['OPENAI_BASE_URL'])))


if __name__ == '__main__':
    main()
