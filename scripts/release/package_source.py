#!/usr/bin/env python3
"""Export a committed ASTR source candidate without changing the checkout."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
DOC_MANIFEST = 'documents/ASTR_RELEASE_DOCUMENT_MANIFEST.txt'
PRODUCTION = ['CMakeLists.txt', 'src', 'src_gpu', 'user_define_module', 'examples']
EXCLUDED = {'examples/Lid-Driven-Cavity/reference_data.opj',
            'examples/Lid-Driven-Cavity/results.pptx'}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def payload(revision):
    commit = git('rev-parse', '--verify', revision+'^{commit}').decode().strip()
    docs = git('show', commit+':'+DOC_MANIFEST).decode().splitlines()
    paths = sorted(set(PRODUCTION + ['LICENSE', DOC_MANIFEST] + docs))
    archive = git('archive', '--format=tar', commit, '--', *paths)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for entry in source:
            if entry.isdir() or entry.name in EXCLUDED:
                continue
            path = PurePosixPath(entry.name)
            if not entry.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Unsupported archive entry: {entry.name}')
            files[entry.name] = (entry.mode, source.extractfile(entry).read())
    for required in docs + ['LICENSE', 'src_gpu/checkpoint_gpu.cuf']:
        if required not in files:
            raise ValueError(f'Missing required release file: {required}')
    metadata = dict(schema=1, source_commit=commit, status='local-source-candidate',
                    target_platform_qualified=False, default_air5=False,
                    configure_required=['-DBUILD_TESTING=OFF', '-DCHEMISTRY=OFF'],
                    excluded_tracked_artifacts=sorted(EXCLUDED),
                    note='Historical acceptance documents describe their recorded revision; '
                         'this manifest identifies the packaged commit. No binaries included.')
    files['RELEASE.json'] = (0o644, (json.dumps(metadata, indent=2)+'\n').encode())
    checksums = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
                        for name, (_, data) in sorted(files.items()))
    files['SHA256SUMS'] = (0o644, checksums.encode())
    return commit, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    commit, files = payload(args.revision)
    prefix = 'astr-source-'+commit[:12]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as target:
                for name, (mode, data) in sorted(files.items()):
                    entry = tarfile.TarInfo(prefix+'/'+name)
                    entry.mode, entry.size, entry.mtime = mode, len(data), 0
                    target.addfile(entry, io.BytesIO(data))
    with tarfile.open(args.output) as archive:
        for entry in archive:
            name = entry.name.removeprefix(prefix+'/')
            if archive.extractfile(entry).read() != files[name][1]:
                raise ValueError(f'Archive round-trip mismatch: {name}')
    print(json.dumps(dict(commit=commit, files=len(files), archive=str(args.output.resolve()),
                         sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
