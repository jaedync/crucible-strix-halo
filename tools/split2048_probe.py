#!/usr/bin/env python3
"""Prepare a bench-only 2048-cut qualification. Serving source stays unchanged."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROWS = '    int R = ConversationRows(i, ids.size(), head ? std::min(step, kMaxRows) : step, schedule_boundary);'
SIGNATURE = '    const ConversationResume& resume = {}, ConversationHooks* conversation = nullptr) {'


def schedule(length, boundary, cut=0):
    if length < 1 or boundary < 0 or boundary > length:
        raise ValueError('invalid prompt or boundary')
    result, at, found = [], 0, False
    while at < length:
        end = min(at + 4096, boundary if at < boundary else length, length)
        if at < cut < end:
            if (cut - at) % 2048 or min(cut - at, end - cut) <= 8:
                raise ValueError('cut changes core intervals or creates a small-row piece')
            result.extend((cut - at, end - cut))
            found = True
        else:
            result.append(end - at)
        at = end
    if cut and not found:
        raise ValueError('cut must be an interior core boundary')
    return result


def adapt_prompt(source):
    start = source.index('template <class Engine, class Copy, class Mtp, class Final>')
    end = source.index('// Finish the synchronous snapshot', start)
    body = source[start:end]
    if body.count(ROWS) != 1 or body.count(SIGNATURE) != 1:
        raise ValueError('production prompt source changed')
    body = body.replace('RunPrompt(', 'RunSplitPrompt(', 1)
    body = body.replace('int mtp_prefix = 0,', 'int mtp_prefix,', 1)
    body = body.replace(SIGNATURE, '    const ConversationResume& resume, ConversationHooks* conversation,\n'
                        '    const std::vector<int>& schedule) {\n  size_t call = 0;', 1)
    body = body.replace(ROWS, '    int R = schedule.at(call++);\n'
                        '    if (R <= 0 || R > EngineLaunchRows(eng) || i + R > ids.size()) throw std::runtime_error("invalid probe schedule");', 1)
    return '#pragma once\nnamespace crucible {\n' + body + '}\n'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare_case(root, vocabulary=248320, modes=('whole', 'split'), expected_cut=None):
    inventory = {'state.bin', 'state.json', 'logits.f32', 'ids.i32'}
    records, restores = {}, {}
    for mode in modes:
        directory = Path(root) / mode
        expected = inventory | ({'restore.json'} if mode == 'restored' else set())
        if set(p.name for p in directory.iterdir()) != expected:
            raise ValueError('unexpected capture inventory')
        if any(not p.is_file() or p.is_symlink() for p in directory.iterdir()):
            raise ValueError('invalid capture file')
        meta = json.loads((directory / 'state.json').read_text())
        if (directory / 'state.bin').stat().st_size != meta['bytes'] or meta['bytes'] <= 0:
            raise ValueError('invalid state size')
        if (directory / 'logits.f32').stat().st_size != vocabulary * 4:
            raise ValueError('invalid logit size')
        if (directory / 'ids.i32').stat().st_size != 8 * 4:
            raise ValueError('invalid continuation size')
        if mode == 'restored':
            marker = json.loads((directory / 'restore.json').read_text())
            if (expected_cut is None or marker.get('length') != expected_cut or
                    not isinstance(marker.get('bytes'), int) or marker['bytes'] <= 0):
                raise ValueError('invalid restore marker')
            restores[mode] = {'marker': marker, 'sha256': digest(directory / 'restore.json')}
        records[mode] = {'meta': meta, 'hashes': {name: digest(directory / name) for name in sorted(inventory)}}
    equal = all(record == records[modes[0]] for record in records.values())
    return {'equal': equal, 'captures': records, 'restores': restores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'split2048_prompt.h').write_text(adapt_prompt((args.source / 'engine/runtime/prompt.h').read_text()))
    cases = [('code', 4096, 2048), ('chat', 8192, 6144), ('spanish', 20480, 18432)]
    manifest = []
    for name, length, cut in cases:
        original = args.source / f'bench/suite/kl/{name}-4096.ids'
        ids = [int(v) for v in original.read_text().replace(',', ' ').split()]
        if len(ids) != 4096:
            raise ValueError('expected 4096 source IDs')
        path = args.output / f'{name}-{length}.ids'
        path.write_text(','.join(map(str, (ids * ((length + 4095) // 4096))[:length])) + '\n')
        manifest.append({'name': name, 'length': length, 'cut': cut, 'ids': str(path.resolve()),
                         'source_sha256': digest(original), 'sha256': digest(path),
                         'whole': schedule(length, 0), 'split': schedule(length, 0, cut)})
    (args.output / 'cases.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if args.build:
        files = sorted([*args.source.glob('engine/**/*'), *args.source.glob('kernels/**/*'),
                        args.source / 'engine/exl3/safetensors.h', args.source / 'bench/split2048.hip',
                        Path(__file__), args.source / 'tools/run_split2048.py', args.source / 'tools/gates.py',
                        args.output / 'split2048_prompt.h', args.output / 'cases.json',
                        *(Path(case['ids']) for case in manifest)])
        hashes = {str(path.resolve()): digest(path) for path in files if path.is_file()}
        binary = args.output.resolve() / 'split2048'
        command = ['/opt/rocm/bin/hipcc', '-O3', '-std=c++20', '--offload-arch=gfx1151',
                   '--rocm-path=/opt/rocm', '-I/opt/rocm/include', '-I' + str(args.output.resolve()),
                   '-o', str(binary), str((args.source / 'bench/split2048.hip').resolve()), '-lpthread']
        with (args.output / 'build.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if any(digest(path) != value for path, value in hashes.items()):
            raise ValueError('build inputs changed')
        record = {'source_commit': args.source_commit, 'command': command,
                  'source_hashes': hashes, 'binary_sha256': digest(binary)}
        (args.output / 'build.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
