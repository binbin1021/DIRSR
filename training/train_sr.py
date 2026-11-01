#!/usr/bin/env python3
"""DIRSR continuation with a frozen learned-offset IDE and DAPE.

A synchronized cycle retains G-reconstruction, G-GAN, D-real, D-fake updates.
Full optimizer, sampler and RNG states are saved separately from SR exports.
"""
import argparse
import contextlib
import datetime
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


class Progress:
    def __init__(self, config_sha):
        self.data = dict(local_step=0, epoch=0, microbatches=0,
                         generator_updates=0, discriminator_updates=0,
                         elapsed_seconds=0., config_sha256=config_sha)

    def state_dict(self):
        return self.data.copy()

    def load_state_dict(self, state):
        if state['config_sha256'] != self.data['config_sha256']:
            raise ValueError('Resume configuration differs from saved run')
        self.data = state.copy()


def main(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.utils import set_seed
    from diffusers.optimization import get_scheduler
    from tensorboardX import SummaryWriter
    from omegaconf import OmegaConf
    from transformers import BertTokenizer
    import lpips
    import pyiqa

    config_path = Path(args.config).resolve()
    project = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project / 'scripts'))
    from _common import resolve_train_config
    cfg = resolve_train_config(json.loads(config_path.read_text()))
    root = Path(cfg['experiment_dir'])
    os.chdir(project)
    sys.path.insert(0, str(project / 'src'))
    sys.path.insert(0, str(Path(__file__).parent))
    from ide import IDE
    from dirsr import DIRSR
    from my_utils.region_aware_loss import PhysicsInformedRegionAwareLoss
    from dataset_helpers import make_datasets, degradation_proc
    from sampler_helpers import make_stateful_train_loader, assert_permutation_state
    from ram import inference_ram
    import vision_aided_loss

    for name in ('checkpoints', 'states', 'eval', 'tb_logs', 'data_manifest'):
        (root / name).mkdir(exist_ok=True, parents=True)
    cfg_hash = sha256(config_path)
    progress = Progress(cfg_hash)
    run_started = time.monotonic()
    prior_elapsed = 0.
    stop_requested = []
    writer = None
    status_path = root / 'training_status.json'
    last_checkpoint = None
    initial_gate = {}

    def request_stop(signum, frame):
        stop_requested.append(signum)
        print('Stop requested; finishing the current synchronized cycle.', flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def status(state, **extra):
        value = dict(progress.data)
        value.update(state=state, updated_at=now(), pid=os.getpid(),
                     parent_step=cfg['parent_step'],
                     cumulative_step=cfg['parent_step'] + progress.data['local_step'],
                     target_additional_steps=cfg['additional_steps'],
                     latest_native_checkpoint=last_checkpoint,
                     elapsed_seconds=prior_elapsed + time.monotonic() - run_started)
        value.update(extra)
        atomic_json(status_path, value)

    try:
        status('initializing')
        if cfg['mixed_precision'] != 'no' or int(cfg['num_processes']) != 1:
            raise ValueError('This run is single-process FP32')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(cfg['physical_gpu']):
            raise ValueError('Physical GPU visibility differs from configuration')
        if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
            raise RuntimeError('Expected one healthy visible CUDA GPU')
        set_seed(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32 = cfg['allow_tf32']
        torch.backends.cudnn.allow_tf32 = cfg['allow_tf32']
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
            mixed_precision='no',
            dataloader_config=DataLoaderConfiguration(use_stateful_dataloader=True))
        if accelerator.num_processes != 1:
            raise RuntimeError('Unexpected distributed process count')
        device = accelerator.device

        # Asset identity is checked before any training update.
        hashes = {}
        for key in ('parent_checkpoint', 'ide_checkpoint', 'ram_checkpoint', 'dape_checkpoint'):
            hashes[key] = sha256(cfg[key])
        source_hashes = {str(path.relative_to(project)): sha256(path) for path in (
            Path(__file__), Path(__file__).with_name('dataset_helpers.py'),
            Path(__file__).with_name('sampler_helpers.py'),
            project / 'src/dirsr.py', project / 'src/ide.py',
            project / 'src/my_utils/region_aware_loss.py', project / 'src/my_utils/training_utils.py')}
        train_set, val_set, data_metadata = make_datasets(cfg, root / 'data_manifest')
        atomic_json(root / 'data_manifest/dataset_metadata.json', data_metadata)
        native_config = OmegaConf.create(cfg['datasets'])
        # Resolve current contents rather than pinning a private historical file.
        ignored = {'project_root', 'experiment_dir', 'parent_checkpoint', 'ide_checkpoint',
                   'sd_path', 'ram_checkpoint', 'dape_checkpoint', 'bert_path', 'datasets',
                   'train_manifest', 'val_manifest', 'physical_gpu'}
        settings = {k: v for k, v in cfg.items() if k not in ignored}
        identity = {'settings': settings, 'asset_sha256': hashes,
                    'source_sha256': source_hashes, 'data_sha256': data_metadata['content_sha256']}
        cfg_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        progress.data['config_sha256'] = cfg_hash
        train_dl = make_stateful_train_loader(
            train_set, batch_size=cfg['batch_size'], seed=cfg['seed'], pin_memory=True)
        # Fixed validation monitor; it does not select a checkpoint.
        val_names = [Path(str(p)).name for p in val_set.paths_gt]
        hmtir = [i for i, name in enumerate(val_names) if 'HMTIR' in name or 'HM-TIR' in name]
        m3fd = [i for i in range(len(val_names)) if i not in set(hmtir)]
        n = min(cfg['val_samples'], len(val_set))
        if hmtir:
            count_hm = min(len(hmtir), round(n * len(hmtir) / len(val_set)))
            monitor_indices = sorted(m3fd[:n-count_hm] + hmtir[:count_hm])
        else:
            monitor_indices = list(range(n))
        val_dl = torch.utils.data.DataLoader(
            torch.utils.data.Subset(val_set, monitor_indices), batch_size=1,
            shuffle=False, num_workers=0)
        atomic_json(root / 'data_manifest/monitor.json', {
            'indices': monitor_indices, 'names': [val_names[i] for i in monitor_indices],
            'purpose': 'fixed validation monitoring; no checkpoint selection'})

        ide = IDE(num_in_ch=3, num_degradation=2)
        ide.load_new_checkpoint(cfg['ide_checkpoint'])
        ide = ide.to(device, dtype=torch.float32).eval().requires_grad_(False)
        ide_initial = {k: v.detach().cpu().clone() for k, v in ide.state_dict().items()}
        if not any('offset' in k and torch.count_nonzero(v).item() for k, v in ide_initial.items()):
            raise ValueError('IDE checkpoint does not contain learned nonzero offsets')

        parent = torch.load(cfg['parent_checkpoint'], map_location='cpu')
        sr = DIRSR(sd_path=cfg['sd_path'], pretrained_path=cfg['parent_checkpoint'],
                    lora_rank_unet=parent['rank_unet'], lora_rank_vae=parent['rank_vae'])
        # Check all exported SR fields, including conditioning and W, exactly.
        checked = 0
        for field, module in (
            ('state_dict_unet', sr.unet), ('state_dict_vae', sr.vae),
            ('state_dict_vae_de_mlp', sr.vae_de_mlp),
            ('state_dict_unet_de_mlp', sr.unet_de_mlp),
            ('state_dict_vae_block_mlp', sr.vae_block_mlp),
            ('state_dict_unet_block_mlp', sr.unet_block_mlp),
            ('state_dict_vae_fuse_mlp', sr.vae_fuse_mlp),
            ('state_dict_unet_fuse_mlp', sr.unet_fuse_mlp)):
            current = module.state_dict()
            for key, tensor in parent[field].items():
                if not torch.equal(current[key].detach().cpu(), tensor):
                    raise ValueError('Parent SR tensor was not restored: ' + field + ':' + key)
                checked += 1
        if not torch.equal(sr.W.detach().cpu(), parent['w'].detach().cpu()):
            raise ValueError('Parent Fourier W was not restored')
        for key, module in (('state_dict_vae_block', sr.vae_block_embeddings),
                            ('state_dict_unet_block', sr.unet_block_embeddings)):
            for name, tensor in parent['state_embeddings'][key].items():
                if not torch.equal(module.state_dict()[name].detach().cpu(), tensor):
                    raise ValueError('Parent block embeddings were not restored')
                checked += 1
        initial_gate['parent_tensor_count_exact'] = checked + 1
        del parent

        # Explicitly freeze the base and enable exactly the native optimizer set.
        selected = []
        conditioning = []
        for name in ('vae_block_embeddings', 'unet_block_embeddings', 'vae_de_mlp',
                     'unet_de_mlp', 'vae_block_mlp', 'unet_block_mlp',
                     'vae_fuse_mlp', 'unet_fuse_mlp'):
            conditioning.extend(getattr(sr, name).parameters())
        selected.extend(conditioning)
        selected.extend(p for name, p in sr.unet.named_parameters() if 'lora' in name)
        selected.extend(sr.unet.conv_in.parameters())
        selected.extend(p for name, p in sr.vae.named_parameters() if 'lora' in name)
        if len(selected) != len({id(p) for p in selected}):
            raise ValueError('Duplicate optimizer parameters')
        sr.requires_grad_(False)
        for p in selected:
            p.requires_grad_(True)
        sr.set_train()
        sr.text_encoder.eval()
        sr.ram_embed_proj.eval().requires_grad_(False)
        if {id(p) for p in sr.parameters() if p.requires_grad} != {id(p) for p in selected}:
            raise ValueError('Unexpected trainable SR parameters outside native optimizer')
        if cfg['enable_xformers']:
            sr.unet.enable_xformers_memory_efficient_attention()
        if cfg['gradient_checkpointing']:
            sr.unet.enable_gradient_checkpointing()
        sr.to(device, dtype=torch.float32)
        update_probes = [(sr.unet.conv_in.weight, sr.unet.conv_in.weight.detach().cpu().clone()),
                         (sr.unet_de_mlp[0].weight, sr.unet_de_mlp[0].weight.detach().cpu().clone())]

        disc = vision_aided_loss.Discriminator(
            cv_type='dino', output_type='conv_multi_level',
            loss_type='multilevel_sigmoid_s', device=str(device)).to(device)
        disc.train().requires_grad_(True)
        disc.cv_ensemble.requires_grad_(False)
        for name, module in disc.named_modules():
            if 'attn' in name:
                module.fused_attn = False
        perceptual = lpips.LPIPS(net='vgg').to(device).eval().requires_grad_(False)
        dual = PhysicsInformedRegionAwareLoss(
            w_fg_edge=cfg['w_fg_edge'], w_fg_percep=cfg['w_fg_percep'],
            w_bg_smooth=cfg['w_bg_smooth'], w_bg_tv=cfg['w_bg_tv'], device=device
        ).to(device).eval().requires_grad_(False)

        # Freeze infrared-adapted DAPE; tokenizer is fully local.
        dape_module = importlib.import_module('ram.models.ram_lora')

        def local_tokenizer():
            tokenizer = BertTokenizer.from_pretrained(cfg['bert_path'], local_files_only=True)
            tokenizer.add_special_tokens({'bos_token': '[DEC]'})
            tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
            tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
            return tokenizer

        dape_module.init_tokenizer = local_tokenizer
        tagger = dape_module.ram(pretrained=cfg['ram_checkpoint'],
                                pretrained_condition=cfg['dape_checkpoint'],
                                image_size=384, vit='swin_l')
        tagger.to(device).eval().requires_grad_(False)
        mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)

        @torch.no_grad()
        def positives(raw_lr):
            tagged = (F.interpolate(raw_lr, size=(384, 384), mode='bilinear',
                                    align_corners=False) - mean) / std
            return [f"{cfg['pos_prompt']}, {inference_ram(tagged[i:i+1], tagger)[0]},"
                    for i in range(raw_lr.shape[0])]

        adam_kwargs = dict(lr=cfg['learning_rate'],
                           betas=(cfg['adam_beta1'], cfg['adam_beta2']),
                           weight_decay=cfg['weight_decay'], eps=cfg['epsilon'])
        optimizer = torch.optim.AdamW(selected, **adam_kwargs)
        optimizer_d = torch.optim.AdamW(disc.parameters(), **adam_kwargs)
        scheduler = get_scheduler('constant', optimizer=optimizer)
        scheduler_d = get_scheduler('constant', optimizer=optimizer_d)
        sr, disc, optimizer, optimizer_d, train_dl, scheduler, scheduler_d = accelerator.prepare(
            sr, disc, optimizer, optimizer_d, train_dl, scheduler, scheduler_d)
        accelerator.register_for_checkpointing(progress)
        if {id(p) for p in ide.parameters()} & {id(p) for p in selected}:
            raise ValueError('Frozen IDE is in SR optimizer')
        provenance = dict(config_sha256=cfg_hash, asset_sha256=hashes,
                          source_sha256=source_hashes, initialized_at=now(),
                          parent_restore=initial_gate.copy(), train_count=len(train_set),
                          continuation_holdout_count=len(val_set), monitor_count=len(monitor_indices),
                          sr_trainable_parameters=sum(p.numel() for p in selected),
                          ide_frozen=True, dape_frozen=True, dtype='float32',
                          optimizer_and_discriminator='fresh for weight-only parent checkpoint',
                          cycle_definition='one synchronized native cycle: G-recon, G-GAN, D-real, D-fake',
                          normalization='IDE receives raw LR [0,1], no additional (x+1)/2 transform')
        # Construct metrics before restoring RNG: LPIPS/VGG construction itself
        # consumes RNG even though its final pretrained weights are fixed.
        metrics = {
            'psnr': pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device).eval(),
            'ssim': pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device).eval(),
            'lpips': pyiqa.create_metric('lpips').to(device).eval()}
        for metric in metrics.values():
            metric.requires_grad_(False)
        writer = SummaryWriter(str(root / 'tb_logs'))
        if args.resume:
            state_dir = Path(args.resume)
            complete = json.loads((state_dir / 'complete.json').read_text())
            if complete['config_sha256'] != cfg_hash:
                raise ValueError('Resume state uses a different config')
            sampler_evidence = assert_permutation_state(torch.load(
                state_dir / 'dl_state_dict.bin', map_location='cpu'))
            accelerator.load_state(str(state_dir), map_location='on_device')
            prior_elapsed = progress.data['elapsed_seconds']
            provenance['resumed_from'] = str(state_dir)
            last_checkpoint = str(root / complete['native_checkpoint'])
            opt_steps = {}
            for name, opt in [('generator', optimizer), ('discriminator', optimizer_d)]:
                actual = sorted({int(state['step'].item()) for state in opt.state.values() if 'step' in state})
                if not actual or max(actual) != 2 * progress.data['local_step']:
                    raise ValueError('Restored Adam step count differs from saved training progress')
                opt_steps[name] = actual
            atomic_json(root / 'resume_checks.json', {
                'loaded_from': str(state_dir), 'local_step': progress.data['local_step'],
                'adam_steps': opt_steps, 'sampler_state': sampler_evidence,
                'rng_restored_after_all_network_construction': True, 'at': now()})
        elif (root / 'latest_state.json').exists():
            raise RuntimeError('Existing run found; explicitly resume its full state')
        atomic_json(root / 'provenance.json', provenance)

        def export_native():
            nonlocal last_checkpoint
            cumulative = cfg['parent_step'] + progress.data['local_step']
            path = root / 'checkpoints' / f'model_{cumulative}.pkl'
            temporary = path.with_name(path.name + '.tmp')
            accelerator.unwrap_model(sr).save_model(str(temporary))
            os.replace(temporary, path)
            last_checkpoint = str(path)
            return path

        def save_full():
            if not accelerator.sync_gradients:
                raise RuntimeError('Full state requires a synchronized cycle boundary')
            path = export_native()
            progress.data['elapsed_seconds'] = prior_elapsed + time.monotonic() - run_started
            destination = root / 'states' / f"step_{progress.data['local_step']:05d}"
            temporary = destination.with_name(destination.name + '.tmp')
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir()
            accelerator.save_state(str(temporary), safe_serialization=False)
            sampler_evidence = assert_permutation_state(torch.load(
                temporary / 'dl_state_dict.bin', map_location='cpu'))
            required = ['pytorch_model.bin', 'pytorch_model_1.bin', 'optimizer.bin',
                        'optimizer_1.bin', 'scheduler.bin', 'scheduler_1.bin',
                        'random_states_0.pkl', 'dl_state_dict.bin', 'custom_checkpoint_0.pkl']
            for name in required:
                if not (temporary / name).is_file():
                    raise RuntimeError('Incomplete full state: ' + name)
            atomic_json(temporary / 'complete.json', {
                'config_sha256': cfg_hash, 'progress': progress.data.copy(),
                'native_checkpoint': str(path.relative_to(root)), 'saved_at': now(),
                'sampler_state': sampler_evidence,
                'files': {p.name: p.stat().st_size for p in temporary.iterdir() if p.is_file()}})
            if destination.exists():
                # Only a synchronized stop immediately after a periodic save can
                # reach this case; the completed state is already sufficient.
                shutil.rmtree(temporary)
            else:
                os.replace(temporary, destination)
            atomic_json(root / 'latest_state.json', {
                'state_dir': str(destination.relative_to(root)), 'native_checkpoint': str(path.relative_to(root)),
                'local_step': progress.data['local_step'], 'saved_at': now()})
            old = sorted(p for p in (root / 'states').glob('step_[0-9]*')
                         if p.is_dir() and (p / 'complete.json').is_file())
            for directory in old[:-2]:
                shutil.rmtree(directory)
            print('Saved complete continuation state:', destination, flush=True)

        @contextlib.contextmanager
        def preserve_rng():
            states = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
            try:
                yield
            finally:
                random.setstate(states[0])
                np.random.set_state(states[1])
                torch.set_rng_state(states[2])
                torch.cuda.set_rng_state_all(states[3])

        @torch.no_grad()
        def validate():
            unwrapped = accelerator.unwrap_model(sr)
            modes = {module: module.training for module in unwrapped.modules()}
            unwrapped.eval()
            results = []
            try:
                with preserve_rng():
                    for i, batch in enumerate(val_dl):
                        source, target, raw = degradation_proc(native_config, batch, device)
                        score = ide(raw.detach().float())
                        pred = unwrapped(source, score, positives(raw))
                        if pred.shape[-2:] != target.shape[-2:]:
                            target = F.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)
                        out = (pred * .5 + .5).clamp(0, 1)
                        gt = (target * .5 + .5).clamp(0, 1)
                        row = {name: float(metric(out, gt).mean()) for name, metric in metrics.items()}
                        row['name'] = Path(batch['gt_path'][0]).name
                        results.append(row)
                        if i < 3:
                            from torchvision.transforms.functional import to_pil_image
                            folder = root / 'eval' / f"step_{progress.data['local_step']:05d}"
                            folder.mkdir(exist_ok=True, parents=True)
                            to_pil_image(out[0].cpu()).save(folder / row['name'])
            finally:
                for module, mode in modes.items():
                    module.training = mode
            avg = {key: float(np.mean([r[key] for r in results])) for key in metrics}
            record = dict(local_step=progress.data['local_step'], at=now(), samples=len(results), mean=avg, per_image=results)
            with (root / 'validation.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            for key, value in avg.items():
                writer.add_scalar('Val/' + key, value, progress.data['local_step'])
            print('Continuation holdout monitor:', avg, flush=True)
            return avg

        print('Initialized parent SR exactly; fixed new IDE and DAPE. Starting additional',
              cfg['additional_steps'], 'native cycles.', flush=True)
        if not args.resume:
            status('baseline_validation')
            validate()
        status('training', initialization_checks=initial_gate)
        report_started = time.monotonic()
        report_initial_step = progress.data['local_step']
        micro_losses = []
        while progress.data['local_step'] < cfg['additional_steps'] and not stop_requested:
            for batch in train_dl:
                with accelerator.accumulate(sr, disc):
                    source, target, raw = degradation_proc(native_config, batch, device)
                    mask = batch['mask'].to(device)
                    if progress.data['microbatches'] == 0:
                        initial_gate.update(raw_lr_range=[float(raw.min()), float(raw.max())],
                                            mask_range=[float(mask.min()), float(mask.max())],
                                            lr_shape=list(raw.shape), hr_shape=list(target.shape))
                        if raw.min() < 0 or raw.max() > 1 or mask.min() < 0 or mask.max() > 1:
                            raise ValueError('LR/mask range differs from expected [0,1]')
                    with torch.no_grad():
                        score = ide(raw.detach().float()).detach()
                        pos = positives(raw)
                    if not torch.isfinite(score).all():
                        raise FloatingPointError('Nonfinite IDE scores')
                    probs = torch.rand(source.shape[0], device=device)
                    prompts = [cfg['neg_prompt'] if p < cfg['neg_prob'] else text for p, text in zip(probs, pos)]
                    mixed_target = torch.where(probs.reshape(-1, 1, 1, 1) < cfg['neg_prob'], source, target)
                    pred = sr(source.detach(), score, prompts)
                    if pred.shape[-2:] != mixed_target.shape[-2:]:
                        mixed_target = F.interpolate(mixed_target, size=pred.shape[-2:], mode='bilinear', align_corners=False)
                    l2 = F.mse_loss(pred.float(), mixed_target.detach().float()) * cfg['lambda_l2']
                    lp = perceptual(pred.float(), mixed_target.detach().float()).mean() * cfg['lambda_lpips']
                    reward = dual(pred.float(), mixed_target.detach().float(), mask.float()) * cfg['lambda_dual']
                    recon = l2 + lp + reward
                    if not torch.isfinite(recon):
                        raise FloatingPointError('Nonfinite reconstruction loss')
                    accelerator.backward(recon)
                    if progress.data['local_step'] == 0 and accelerator.sync_gradients:
                        gradients = [p.grad for p in selected if p.grad is not None]
                        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                            raise FloatingPointError('Missing/nonfinite generator gradients')
                        initial_gate['nonzero_conditioning_gradients'] = sum(
                            p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in conditioning)
                        if not initial_gate['nonzero_conditioning_gradients']:
                            raise RuntimeError('Conditioning MLPs/embeddings are not learning')
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(selected, cfg['max_grad_norm'])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    pred = sr(source.detach(), score, pos)
                    gan = disc(pred, for_G=True).mean() * cfg['lambda_gan']
                    if not torch.isfinite(gan):
                        raise FloatingPointError('Nonfinite generator GAN loss')
                    accelerator.backward(gan)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(selected, cfg['max_grad_norm'])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    real = disc(target.detach(), for_real=True).mean() * cfg['lambda_gan']
                    if not torch.isfinite(real):
                        raise FloatingPointError('Nonfinite real discriminator loss')
                    accelerator.backward(real)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(disc.parameters(), cfg['max_grad_norm'])
                    optimizer_d.step()
                    scheduler_d.step()
                    optimizer_d.zero_grad(set_to_none=True)
                    fake = disc(pred.detach(), for_real=False).mean() * cfg['lambda_gan']
                    if not torch.isfinite(fake):
                        raise FloatingPointError('Nonfinite fake discriminator loss')
                    accelerator.backward(fake)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(disc.parameters(), cfg['max_grad_norm'])
                    optimizer_d.step()
                    optimizer_d.zero_grad(set_to_none=True)
                    micro_losses.append({'l2': float(l2.detach()), 'lpips': float(lp.detach()),
                                         'dual': float(reward.detach()), 'reconstruction': float(recon.detach()),
                                         'gan_g': float(gan.detach()), 'gan_d': float((real + fake).detach())})
                    progress.data['microbatches'] += 1
                    del pred, mixed_target, recon, l2, lp, reward, gan, real, fake

                if accelerator.sync_gradients:
                    progress.data['local_step'] += 1
                    progress.data['generator_updates'] += 2
                    progress.data['discriminator_updates'] += 2
                    step = progress.data['local_step']
                    losses = {key: float(np.mean([r[key] for r in micro_losses])) for key in micro_losses[0]}
                    micro_losses.clear()
                    record = dict(local_step=step, cumulative_step=cfg['parent_step'] + step,
                                  at=now(), loss=losses, lr=scheduler.get_last_lr()[0])
                    with (root / 'training.jsonl').open('a') as f:
                        f.write(json.dumps(record) + '\n')
                    for key, value in losses.items():
                        writer.add_scalar('Train/' + key, value, step)
                    writer.add_scalar('Train/LR', record['lr'], step)
                    if step == 1:
                        if any(p.grad is not None or p.requires_grad for p in ide.parameters()):
                            raise RuntimeError('IDE was accidentally trained')
                        if not all(torch.equal(v.detach().cpu(), ide_initial[k]) for k, v in ide.state_dict().items()):
                            raise RuntimeError('Frozen IDE weights changed')
                        initial_gate['ide_unchanged_after_first_cycle'] = True
                        initial_gate['ide_scores_first_cycle'] = score.cpu().tolist()
                        initial_gate['first_cycle_finite'] = True
                        initial_gate['sr_parameter_probe_max_changes'] = [
                            float((p.detach().cpu() - previous).abs().max()) for p, previous in update_probes]
                        if not all(v > 0 for v in initial_gate['sr_parameter_probe_max_changes']):
                            raise RuntimeError('SR parameters did not change after first update cycle')
                        atomic_json(root / 'initialization_checks.json', initial_gate)
                    if step == 1 or step % cfg['log_interval'] == 0:
                        elapsed = time.monotonic() - report_started
                        seconds_per_cycle = elapsed / max(1, step - report_initial_step)
                        eta = seconds_per_cycle * (cfg['additional_steps'] - step)
                        status('training', latest_losses=losses, seconds_per_cycle=seconds_per_cycle,
                               estimated_remaining_seconds=eta, learning_rate=record['lr'],
                               gpu_memory_peak_gib=torch.cuda.max_memory_allocated() / 2**30)
                        print(f"step={step}/{cfg['additional_steps']} cumulative={cfg['parent_step']+step} "
                              f"loss={losses} seconds/cycle={seconds_per_cycle:.2f} eta_hours={eta/3600:.2f}", flush=True)
                    final = step >= cfg['additional_steps']
                    if step == 1 or step % cfg['full_state_interval'] == 0 or final:
                        save_full()
                    elif step % cfg['checkpoint_interval'] == 0:
                        export_native()
                    if step % cfg['val_interval'] == 0 or final:
                        status('validation')
                        validate()
                        writer.flush()
                        gc.collect()
                        torch.cuda.empty_cache()
                        status('training')
                    if final or stop_requested:
                        break
            if progress.data['local_step'] >= cfg['additional_steps'] or stop_requested:
                break
            progress.data['epoch'] += 1
        if stop_requested:
            if progress.data['local_step'] > 0:
                save_full()
            status('stopped', reason='requested signal at synchronized cycle boundary')
        else:
            status('completed', final_checkpoint=last_checkpoint)
            print('Completed exactly', progress.data['local_step'], 'additional native cycles.', flush=True)
        if writer:
            writer.flush()
            writer.close()
    except BaseException as exc:
        status('failed', error=repr(exc), traceback=traceback.format_exc())
        if writer:
            writer.close()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', help='Complete accelerator state directory')
    main(parser.parse_args())
