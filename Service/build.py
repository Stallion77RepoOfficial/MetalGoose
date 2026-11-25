#!/usr/bin/env python3
"""Build MetalGoose's headless service from the shared app sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [repo / p for p in json.loads((repo / 'Service/sources.json').read_text())]
    shaders = sorted((repo / 'MetalGoose/Shaders').glob('*.metal'))
    inputs = sources + shaders + sorted((repo / 'MetalGoose/Shaders').glob('*.h')) + [Path(__file__)]
    fingerprint = hashlib.sha256()
    for p in inputs:
        fingerprint.update(str(p.relative_to(repo)).encode() + b'\0' + p.read_bytes())
    fingerprint.update(subprocess.check_output(['xcrun', 'swiftc', '--version']))
    key = fingerprint.hexdigest()
    stamp = output / 'source.sha256'
    total = len(shaders) + 2

    def progress(done):
        print('@metalgoose ' + json.dumps({'completed': done, 'total': total}), flush=True)

    if (stamp.is_file() and stamp.read_text().strip() == key
            and (output / 'metalgoose-backend').is_file() and (output / 'default.metallib').is_file()):
        progress(total)
        return
    progress(0)
    subprocess.run(['xcrun', 'swiftc', '-O', '-wmo', '-parse-as-library', '-swift-version', '6',
                    '-target', 'arm64-apple-macos27.0', '-import-objc-header',
                    str(repo / 'MetalGoose/Shaders/ShaderTypes.h'), '-o', str(output / 'metalgoose-backend'),
                    *map(str, sources)], check=True)
    progress(1)
    air = output / 'air'
    air.mkdir(exist_ok=True)
    objects = []
    for i, shader in enumerate(shaders, 2):
        obj = air / (shader.stem + '.air')
        subprocess.run(['xcrun', 'metal', '-c', '-O2', '-fmetal-math-mode=fast',
                        '-target', 'air64-apple-macos27.0', '-std=metal4.1',
                        '-o', str(obj), str(shader)], check=True)
        objects.append(str(obj))
        progress(i)
    subprocess.run(['xcrun', 'metallib', '-o', str(output / 'default.metallib'), *objects], check=True)
    stamp.write_text(key + '\n')
    progress(total)


if __name__ == '__main__':
    main()
