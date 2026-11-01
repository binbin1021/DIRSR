"""Native MM-RealSR contrast/anchor synthesis for standalone infrared IDE training.

The official dataset class and feed_data function are compiled from their source
AST without changing any function body. Package initializers and the dataset's
registry decorator are excluded, avoiding a conflicting DEResNet registration.
Only the queue allocation is made device aware so CPU validation is possible.
No original repository files or installed packages are modified.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import importlib.util
from pathlib import Path
import random
import sys
from types import ModuleType

import numpy as np
import torch
from torch.nn import functional as F
import yaml


_COMPONENTS = {}
_SOURCE_FILES = {
    'dataset': 'mmrealsr/data/mmrealsr_dataset.py',
    'feed': 'mmrealsr/models/mmrealsrnet_model.py',
    'degradations': 'mmrealsr/data/degradations.py',
    'transforms': 'mmrealsr/data/transforms.py',
}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _body_hash(node):
    body = ast.Module(body=copy.deepcopy(node.body), type_ignores=[])
    return _sha256(ast.dump(body, include_attributes=False).encode())


def _ensure_basicsr(mmroot):
    """Reuse an existing dependency or locate the neighboring project checkout."""
    if importlib.util.find_spec('basicsr') is None:
        candidates = [mmroot.parent / 'S3Diff-new', mmroot.parent / 'BasicSR',
                      Path(__file__).resolve().parents[2]]
        location = next((p for p in candidates
                         if (p / 'basicsr/__init__.py').is_file()), None)
        if location is None:
            raise ImportError('BasicSR is required; expose the existing project checkout on PYTHONPATH')
        sys.path.insert(0, str(location))
    return importlib.import_module('basicsr')


def _load_source_module(path, name):
    """Import one native helper file directly, bypassing mmrealsr.__init__."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _components(mmroot):
    mmroot = Path(mmroot).resolve()
    paths = {key: mmroot / relative for key, relative in _SOURCE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing native MM-RealSR source: ' + ', '.join(missing))
    sources = {key: path.read_bytes() for key, path in paths.items()}
    hashes = {key: _sha256(data) for key, data in sources.items()}
    cache_key = (str(mmroot), tuple(sorted(hashes.items())))
    if cache_key in _COMPONENTS:
        return _COMPONENTS[cache_key]

    basicsr = _ensure_basicsr(mmroot)
    from basicsr.utils.img_process_util import filter2D
    suffix = _sha256(repr(cache_key).encode())[:12]
    deg = _load_source_module(paths['degradations'], '_dirsr_mmreal_deg_' + suffix)
    transforms = _load_source_module(paths['transforms'], '_dirsr_mmreal_crop_' + suffix)

    dataset_tree = ast.parse(sources['dataset'], filename=str(paths['dataset']))
    dataset_nodes = [node for node in dataset_tree.body
                     if isinstance(node, ast.ClassDef) and node.name == 'MMRealSRGANDataset']
    if len(dataset_nodes) != 1:
        raise RuntimeError('Expected one native MMRealSRGANDataset class')
    dataset_source = dataset_nodes[0]
    dataset_node = copy.deepcopy(dataset_source)
    expected_decorator = 'DATASET_REGISTRY.register()'
    if (len(dataset_node.decorator_list) != 1
            or ast.unparse(dataset_node.decorator_list[0]) != expected_decorator):
        raise RuntimeError('Unexpected native dataset decorator; cannot safely isolate registry')
    dataset_node.decorator_list = []
    if _body_hash(dataset_source) != _body_hash(dataset_node):
        raise RuntimeError('Native dataset body changed during AST extraction')
    imports = []
    excluded_imports = []
    for node in dataset_tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ImportFrom) and node.level:
            if (node.level != 1 or node.module != 'degradations'
                    or {alias.name for alias in node.names}
                    != {'two_random_mixed_kernels', 'three_random_mixed_kernels'}):
                raise RuntimeError('Unexpected relative dataset import')
            excluded_imports.append(ast.unparse(node))
        else:
            imports.append(copy.deepcopy(node))
    if len(excluded_imports) != 1:
        raise RuntimeError('Expected one dataset helper import to bind directly')
    dataset_module = ModuleType('_dirsr_mmreal_dataset_' + suffix)
    dataset_module.__file__ = str(paths['dataset'])
    dataset_module.__dict__.update(
        two_random_mixed_kernels=deg.two_random_mixed_kernels,
        three_random_mixed_kernels=deg.three_random_mixed_kernels)
    sys.modules[dataset_module.__name__] = dataset_module
    tree = ast.fix_missing_locations(ast.Module(body=imports + [dataset_node], type_ignores=[]))
    exec(compile(tree, str(paths['dataset']), 'exec'), dataset_module.__dict__)

    feed_tree = ast.parse(sources['feed'], filename=str(paths['feed']))
    model_nodes = [node for node in feed_tree.body
                   if isinstance(node, ast.ClassDef) and node.name == 'MMRealSRNetModel']
    if len(model_nodes) != 1:
        raise RuntimeError('Expected one native MMRealSRNetModel class')
    feed_nodes = [node for node in model_nodes[0].body
                  if isinstance(node, ast.FunctionDef) and node.name == 'feed_data']
    if len(feed_nodes) != 1:
        raise RuntimeError('Expected one native feed_data method')
    feed_source = feed_nodes[0]
    feed_node = copy.deepcopy(feed_source)
    if _body_hash(feed_source) != _body_hash(feed_node):
        raise RuntimeError('Native feed_data body changed during AST extraction')
    feed_namespace = dict(
        __name__='_dirsr_mmreal_feed_' + suffix, __file__=str(paths['feed']),
        torch=torch, np=np, random=random, F=F, filter2D=filter2D,
        custom_random_add_gaussian_noise_pt_anchor=deg.custom_random_add_gaussian_noise_pt_anchor,
        custom_random_add_poisson_noise_pt_anchor=deg.custom_random_add_poisson_noise_pt_anchor,
        paired_random_crop=transforms.paired_random_crop)
    tree = ast.fix_missing_locations(ast.Module(body=[feed_node], type_ignores=[]))
    exec(compile(tree, str(paths['feed']), 'exec'), feed_namespace)
    audit = dict(
        mmroot=str(mmroot), native_source_paths={key: str(path) for key, path in paths.items()},
        native_source_sha256=hashes,
        dataset_class_body_sha256=_body_hash(dataset_source),
        feed_data_body_sha256=_body_hash(feed_source),
        function_bodies_unchanged=True,
        removed_registry_decorators=[expected_decorator],
        directly_bound_relative_imports=excluded_imports,
        package_initializers_bypassed=['mmrealsr/__init__.py', 'mmrealsr/archs/__init__.py',
                                      'mmrealsr/data/__init__.py', 'mmrealsr/models/__init__.py'],
        basicsr_source=str(Path(basicsr.__file__).resolve()),
        queue_adaptation='Native enqueue/dequeue/copy/shuffle order; buffer device and dtype follow input; CPU randperm preserved as in the source',
        views=['base', 'stronger_blur', 'stronger_noise', 'minimum_anchor', 'maximum_anchor'])
    result = dict(dataset=dataset_module.MMRealSRGANDataset,
                  feed=feed_namespace['feed_data'], audit=audit)
    _COMPONENTS[cache_key] = result
    return result


def load_native_components(mmroot, options_path):
    """Return (official dataset class, source options plus serializable audit).

    Set datasets.train.dataroot_gt/meta_info to the desired HR split before
    constructing the returned class. Each meta_info line is only an image path;
    pass a deepcopy of dataset options because FileClient initialization mutates
    the io_backend mapping.
    """
    native = _components(mmroot)
    options_path = Path(options_path).resolve()
    raw = options_path.read_bytes()
    options = yaml.safe_load(raw)
    if not isinstance(options, dict) or options.get('scale') != 4:
        raise ValueError('Expected an x4 MM-RealSR option mapping')
    options['_native_mmreal'] = copy.deepcopy(native['audit'])
    options['_native_mmreal'].update(options_path=str(options_path),
                                    options_sha256=_sha256(raw))
    return native['dataset'], options


class NativeSynthesizer:
    """Device-aware host for the unchanged native high-order feed_data method.

    feed(data) returns (lq[B,3,C,48,48], anchor[B,2,C,48,48]) for gt_size=192,
    scale=4. Concatenate views along dimension one before flattening B*5 inputs.
    Validation should use a separate instance with use_queue=False.
    """
    view_order = ('base', 'stronger_blur', 'stronger_noise',
                  'minimum_anchor', 'maximum_anchor')

    def __init__(self, options, device, queue_size=90, use_queue=True):
        if '_native_mmreal' not in options:
            raise ValueError('Obtain options through load_native_components first')
        from basicsr.utils import DiffJPEG, USMSharp
        native = _components(options['_native_mmreal']['mmroot'])
        if native['audit']['native_source_sha256'] != options['_native_mmreal']['native_source_sha256']:
            raise RuntimeError('Native MM-RealSR source changed after option loading')
        self.opt = copy.deepcopy(options)
        self.device = torch.device(device)
        self.is_train = True
        self.opt['high_order_degradation'] = True
        self.queue_size = int(queue_size)
        if self.queue_size < 1:
            raise ValueError('queue_size must be positive')
        self.opt['queue_size'] = self.queue_size
        self.use_queue = bool(use_queue)
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        self.usm_sharpener = USMSharp().to(self.device)
        self.gaussian_noise_diff_range = self.opt.get('gaussian_noise_diff_range', (1, 21))
        self.poisson_noise_diff_range = self.opt.get('poisson_noise_diff_range', (0.05, 2.05))
        self.idx = 1
        self.feed_data = native['feed'].__get__(self, type(self))
        self.audit = copy.deepcopy(self.opt['_native_mmreal'])

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        if not self.use_queue:
            return
        # Mirrors mmrealsrnet_model.py:38-74. The original allocation hardcodes
        # .cuda(); shape, clone/copy order and CPU permutation are retained.
        b, _, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            if self.queue_size % b:
                raise ValueError('queue_size must be divisible by source batch size')
            kwargs = dict(device=self.lq.device, dtype=self.lq.dtype)
            self.queue_lr = torch.zeros(self.queue_size, 3, c, h, w, **kwargs)
            self.queue_anchor = torch.zeros(self.queue_size, 2, c, h, w, **kwargs)
            c, h, w = self.gt.size()[-3:]
            self.queue_gt = torch.zeros(self.queue_size, 5, c, h, w,
                                        device=self.gt.device, dtype=self.gt.dtype)
            self.queue_ptr = 0
            self.queue_source_batch = b
        elif b != self.queue_source_batch:
            raise ValueError('Source batch size changed after native queue initialization')
        if self.queue_ptr == self.queue_size:
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_anchor = self.queue_anchor[idx]
            self.queue_gt = self.queue_gt[idx]
            lq_dequeue = self.queue_lr[0:b].clone()
            anchor_dequeue = self.queue_anchor[0:b].clone()
            gt_dequeue = self.queue_gt[0:b].clone()
            self.queue_lr[0:b] = self.lq.clone()
            self.queue_anchor[0:b] = self.anchor.clone()
            self.queue_gt[0:b] = self.gt.clone()
            self.lq, self.anchor, self.gt = lq_dequeue, anchor_dequeue, gt_dequeue
        else:
            self.queue_lr[self.queue_ptr:self.queue_ptr + b] = self.lq.clone()
            self.queue_anchor[self.queue_ptr:self.queue_ptr + b] = self.anchor.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b] = self.gt.clone()
            self.queue_ptr += b

    def feed(self, data):
        self.feed_data(data)
        return self.lq, self.anchor

    def queue_state(self):
        """Return an independent CPU snapshot for exact training resumption."""
        initialized = hasattr(self, 'queue_lr')
        result = dict(queue_size=self.queue_size, use_queue=self.use_queue,
                      initialized=initialized,
                      queue_ptr=int(self.queue_ptr) if initialized else 0,
                      source_batch=int(self.queue_source_batch) if initialized else None,
                      gt_size=int(self.opt['gt_size']), scale=int(self.opt['scale']),
                      view_order=self.view_order)
        if initialized:
            for name in ('queue_lr', 'queue_anchor', 'queue_gt'):
                result[name] = getattr(self, name).detach().cpu().clone()
        return result

    @torch.no_grad()
    def load_queue_state(self, state):
        """Strictly restore native buffers onto this synthesizer's device."""
        if not isinstance(state, dict):
            raise TypeError('Native queue state must be a dictionary')
        required = {'queue_size', 'use_queue', 'initialized', 'queue_ptr',
                    'source_batch', 'gt_size', 'scale', 'view_order'}
        if required - set(state):
            raise ValueError('Incomplete native queue metadata')
        if (state['queue_size'] != self.queue_size
                or state['use_queue'] != self.use_queue
                or state['gt_size'] != self.opt['gt_size']
                or state['scale'] != self.opt['scale']
                or tuple(state['view_order']) != self.view_order):
            raise ValueError('Native queue state does not match the current configuration')
        if not isinstance(state['initialized'], bool):
            raise ValueError('Native queue initialized flag must be Boolean')
        ptr = state['queue_ptr']
        if not isinstance(ptr, int) or isinstance(ptr, bool) or not 0 <= ptr <= self.queue_size:
            raise ValueError('Invalid native queue pointer')
        names = ('queue_lr', 'queue_anchor', 'queue_gt')
        if not state['initialized']:
            if ptr != 0 or state['source_batch'] is not None or any(name in state for name in names):
                raise ValueError('Unexpected buffers in uninitialized native queue state')
            for name in names + ('queue_ptr', 'queue_source_batch'):
                if hasattr(self, name):
                    delattr(self, name)
            return
        if not self.use_queue:
            raise ValueError('A disabled native queue cannot contain saved buffers')
        batch = state['source_batch']
        if (not isinstance(batch, int) or isinstance(batch, bool) or batch < 1
                or self.queue_size % batch or ptr % batch):
            raise ValueError('Invalid native queue source batch or pointer alignment')
        if any(not torch.is_tensor(state.get(name)) for name in names):
            raise ValueError('Native queue buffers must all be tensors')
        lr, anchor, gt = (state[name] for name in names)
        if lr.ndim != 5 or lr.shape[2] < 1:
            raise ValueError('Invalid native LR queue shape')
        channels = lr.shape[2]
        patch = self.opt['gt_size'] // self.opt['scale']
        expected = {
            'queue_lr': (self.queue_size, 3, channels, patch, patch),
            'queue_anchor': (self.queue_size, 2, channels, patch, patch),
            'queue_gt': (self.queue_size, 5, channels, self.opt['gt_size'], self.opt['gt_size']),
        }
        for name in names:
            tensor = state[name]
            if tuple(tensor.shape) != expected[name] or tensor.dtype != torch.float32:
                raise ValueError(f'Native queue shape or FP32 dtype mismatch: {name}')
            if tensor.device.type != 'cpu':
                raise ValueError(f'Native queue snapshots must use CPU tensors: {name}')
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f'Non-finite native queue tensor: {name}')
        for name in names:
            setattr(self, name, state[name].to(self.device).clone())
        self.queue_ptr, self.queue_source_batch = ptr, batch

    def metadata(self):
        return dict(**copy.deepcopy(self.audit),
                    device=str(self.device), source_batch=getattr(self, 'queue_source_batch', None),
                    queue_size=self.queue_size, use_queue=self.use_queue,
                    gt_size=self.opt['gt_size'], scale=self.opt['scale'],
                    input_patch_size=self.opt['gt_size'] // self.opt['scale'],
                    source_stage='Public MMRealSRNet joint-training synthesis reused for separate IDE training')
