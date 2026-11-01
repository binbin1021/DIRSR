#!/usr/bin/env python3
"""Prepare a fixed, leak-checked HR split for learned-offset IDE training.

Only reads source images. Writes new manifests inside --out, never modifies
the source datasets or existing manifests. Compatible with Python 3.8+.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import random
import re
import sys

from PIL import Image


DATASET_ORDER = ('M3FD', 'HMTIR')
TRAIN_NAME = re.compile(r'^(\d+)_(M3FD|HMTIR)$')
TEST_NAME = re.compile(r'^\d+$')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def image_record(path, dataset, frame_id, split):
    # verify() checks the encoded PNG, while a second open reads its metadata.
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
        image_format = image.format
    if image_format != 'PNG':
        raise ValueError('Expected actual PNG image: {}'.format(path))
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions: {}'.format(path))
    return dict(path=str(path.resolve()), filename=path.name,
                dataset=dataset, frame_id=frame_id,
                canonical_id='{}:{}'.format(dataset, int(frame_id)),
                sha256=sha256(path), width=width, height=height, mode=mode,
                size_bytes=path.stat().st_size, split=split)


def images_in(root):
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError('HR directory does not exist: {}'.format(root))
    paths = sorted((path for path in root.iterdir()
                    if path.is_file() and path.suffix.lower() == '.png'),
                   key=lambda path: path.name)
    if not paths:
        raise ValueError('HR directory contains no PNG files: {}'.format(root))
    nested = [path for path in root.iterdir() if path.is_dir() and
              not path.name.startswith('.')]
    if nested:
        raise ValueError('Expected flat HR directory; found subdirectories: {}'.format(nested))
    return paths


def ensure_unique(records, label):
    for key in ('canonical_id', 'sha256'):
        grouped = defaultdict(list)
        for record in records:
            grouped[record[key]].append(record['path'])
        duplicates = {value: paths for value, paths in grouped.items() if len(paths) > 1}
        if duplicates:
            raise ValueError('{} contains duplicate {}: {}'.format(
                label, key, json.dumps(duplicates, ensure_ascii=False)))


def prepare(args):
    out = Path(args.out).resolve()
    output_names = ('train.txt', 'val.txt', 'manifest.json')
    occupied = [str(out / name) for name in output_names if (out / name).exists()]
    if occupied:
        raise FileExistsError('Refusing to overwrite existing manifests: {}'.format(occupied))
    fraction = Decimal(str(args.val_fraction))
    if not Decimal('0') < fraction < Decimal('1'):
        raise ValueError('--val-fraction must be between zero and one')
    training_pool = []
    for path in images_in(Path(args.hr_root)):
        match = TRAIN_NAME.fullmatch(path.stem)
        if not match:
            raise ValueError('Unknown training frame format; expected numeric '
                             '_M3FD or _HMTIR suffix: {}'.format(path.name))
        frame_id, dataset = match.groups()
        training_pool.append(image_record(path, dataset, frame_id, 'unassigned'))
    ensure_unique(training_pool, 'Training HR pool')
    grouped = {dataset: [record for record in training_pool
                        if record['dataset'] == dataset] for dataset in DATASET_ORDER}
    if any(not records for records in grouped.values()):
        raise ValueError('Training pool must contain both M3FD and HMTIR')
    test_records = []
    tests = {'M3FD': Path(args.test_m3fd), 'HMTIR': Path(args.test_hmtir)}
    for dataset in DATASET_ORDER:
        for path in images_in(tests[dataset]):
            if not TEST_NAME.fullmatch(path.stem):
                raise ValueError('Expected numeric canonical test frame name: {}'.format(path.name))
            test_records.append(image_record(path, dataset, path.stem, 'test_only'))
    ensure_unique(test_records, 'Canonical test HR pool')
    if args.expected_test_m3fd is not None:
        count = sum(record['dataset'] == 'M3FD' for record in test_records)
        if count != args.expected_test_m3fd:
            raise ValueError('M3FD test count {} != expected {}'.format(count, args.expected_test_m3fd))
    if args.expected_test_hmtir is not None:
        count = sum(record['dataset'] == 'HMTIR' for record in test_records)
        if count != args.expected_test_hmtir:
            raise ValueError('HMTIR test count {} != expected {}'.format(count, args.expected_test_hmtir))
    overlaps = {}
    for key in ('canonical_id', 'sha256'):
        training_values = {record[key] for record in training_pool}
        test_values = {record[key] for record in test_records}
        overlaps[key] = sorted(training_values & test_values)
        if overlaps[key]:
            raise ValueError('Training/test {} overlap: {}'.format(key, overlaps[key]))
    # One RNG, a stated fixed dataset order, and filename-sorted inputs fully
    # specify this split. Lists written below are sorted for easy inspection.
    rng = random.Random(args.seed)
    per_dataset = {}
    for dataset in DATASET_ORDER:
        shuffled = sorted(grouped[dataset], key=lambda record: record['filename'])
        rng.shuffle(shuffled)
        val_count = int((fraction * len(shuffled)).to_integral_value(rounding=ROUND_CEILING))
        if val_count >= len(shuffled):
            raise ValueError('Validation fraction leaves no training images for {}'.format(dataset))
        for index, record in enumerate(shuffled):
            record['split'] = 'val' if index < val_count else 'train'
        per_dataset[dataset] = dict(total=len(shuffled), train=len(shuffled) - val_count,
                                   val=val_count)
    records = sorted(training_pool, key=lambda record: (record['dataset'], record['filename']))
    train = [record for record in records if record['split'] == 'train']
    val = [record for record in records if record['split'] == 'val']
    assert len(train) + len(val) == len(records)
    assert not ({record['canonical_id'] for record in train} &
                {record['canonical_id'] for record in val})
    assert not ({record['sha256'] for record in train} &
                {record['sha256'] for record in val})
    paths = {'train': ''.join(record['path'] + '\n' for record in train),
             'val': ''.join(record['path'] + '\n' for record in val)}
    # MM-RealSR/BasicSR meta-info parsers split lines by spaces; explicitly
    # reject whitespace so each absolute path remains its first field.
    if any(any(char.isspace() for char in record['path']) for record in records):
        raise ValueError('Absolute paths cannot contain whitespace for BasicSR meta-info')
    manifest = dict(
        schema_version=1, purpose='Learned-offset IDE HR training and internal validation',
        hr_root=str(Path(args.hr_root).resolve()),
        test_roots={dataset: str(root.resolve()) for dataset, root in tests.items()},
        seed=args.seed, validation_fraction=str(fraction),
        split_policy=dict(dataset_name_format='<numeric frame>_M3FD.png or <numeric frame>_HMTIR.png',
            dataset_order=list(DATASET_ORDER),
            algorithm='Filename sort within dataset, one random.Random(seed), sequential shuffle in dataset_order, first ceil(n*fraction) validation.',
            output_order='Sorted by dataset then filename',
            test_policy='Canonical test HR used only for overlap exclusion; never included in train or internal validation.'),
        counts=dict(total=len(records), train=len(train), val=len(val),
                    per_dataset=per_dataset,
                    canonical_test=dict(Counter(record['dataset'] for record in test_records))),
        checks=dict(source_pngs_verified=True, duplicate_training_ids=0,
                    duplicate_training_sha256=0, duplicate_test_ids=0,
                    duplicate_test_sha256=0, train_val_id_overlap=0,
                    train_val_sha256_overlap=0, training_test_id_overlap=0,
                    training_test_sha256_overlap=0),
        list_format='UTF-8, one absolute HR image path per line; no header.',
        list_sha256={split: hashlib.sha256(content.encode('utf-8')).hexdigest()
                     for split, content in paths.items()},
        records=records, canonical_test_records=test_records)
    # No output is created until all source, leakage and split checks pass.
    out.mkdir(parents=True, exist_ok=True)
    for split, content in paths.items():
        with (out / '{}.txt'.format(split)).open('x', encoding='utf-8') as stream:
            stream.write(content)
    with (out / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return dict(out=str(out), counts=manifest['counts'], checks=manifest['checks'],
                train_list=str(out / 'train.txt'), val_list=str(out / 'val.txt'),
                list_sha256=manifest['list_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hr-root', required=True)
    parser.add_argument('--test-m3fd', required=True)
    parser.add_argument('--test-hmtir', required=True)
    parser.add_argument('--expected-test-m3fd', type=int, required=True)
    parser.add_argument('--expected-test-hmtir', type=int, required=True)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--val-fraction', type=float, default=.05)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = prepare(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
