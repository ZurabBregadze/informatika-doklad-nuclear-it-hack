"""Warm latency measurements without disk I/O, plots or quality metrics."""
import copy
import os
import platform
import time

import numpy as np
import torch

from .fresnel_baseline import synthesize
from .metrics import quantized_reconstruction
from .model import Optics


def timed_call(function, device):
    """Wall latency includes Python dispatch and waits for CUDA completion."""
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = function()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return result, time.perf_counter()-started


def latency_summary(rows, key):
    values = [row[key] for row in rows]
    return dict(count=len(values), median_seconds=float(np.median(values)),
                p10_seconds=float(np.percentile(values, 10)),
                p90_seconds=float(np.percentile(values, 90)),
                min_seconds=min(values), max_seconds=max(values))


@torch.inference_mode()
def benchmark(model, data, gs_iterations, gs_seed, *, samples=3, repeats=3, warmup=2):
    if any(type(value) is not int or value < 1 for value in (samples, repeats, warmup)):
        raise ValueError('Timing samples, repeats and warmup must be positive.')
    device = next(model.parameters()).device
    model.eval()
    # Always include a CPU neural measurement beside the CPU GS measurement.
    cpu_model = model if device.type == 'cpu' else copy.deepcopy(model).cpu().eval()
    methods = [('neural_cpu', cpu_model, torch.device('cpu'))]
    if device.type == 'cuda':
        methods.append(('neural_cuda', model, device))
    methods.append(('gs_cpu', None, torch.device('cpu')))
    precise = Optics(dtype=torch.float64)
    targets = [data[index] for index in range(min(samples, len(data)))]
    ids = [folder.name for folder in data.folders[:len(targets)]]
    report = dict(
        samples=ids, repeats=repeats, neural_warmup=warmup, gs_warmup=1,
        torch_version=str(torch.__version__), cpu=platform.processor(),
        platform=platform.platform(), cpu_count=os.cpu_count(), torch_threads=torch.get_num_threads(),
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        core='Targets resident on method device -> continuous phase mask; GS setup included, history disabled.',
        pipeline='CPU targets -> synthesis -> CPU 8-bit phase decoding -> four CPU float64 Fresnel reconstructions -> uint8 arrays.',
        excluded='Dataset reads, model loading, warmup, PNG encoding/writes, metrics, plots and checkpoint/backup I/O.',
        interpretation='CPU comparison uses neural float32 and GS float64. CUDA-vs-CPU ratio describes these deployments, not equal hardware.',
        methods={})
    for name, network, method_device in methods:
        rows = []
        for sample_index, target in enumerate(targets):
            resident = target.double() if network is None else target[None].to(method_device)

            def core():
                if network is None:
                    return synthesize(resident, iterations=gs_iterations, seed=gs_seed, record_history=False)[0]
                return torch.angle(network(resident))[0]

            def pipeline():
                if network is None:
                    phase = synthesize(target.double(), iterations=gs_iterations,
                                       seed=gs_seed, record_history=False)[0]
                else:
                    phase = torch.angle(network(target[None].to(method_device)))[0]
                return quantized_reconstruction(phase, precise)

            if sample_index == 0:
                for _ in range(1 if network is None else warmup):
                    core()
                    pipeline()
                if method_device.type == 'cuda':
                    torch.cuda.synchronize(method_device)
            for repeat in range(repeats):
                _, core_seconds = timed_call(core, method_device)
                _, pipeline_seconds = timed_call(pipeline, method_device)
                rows.append(dict(sample=ids[sample_index], repeat=repeat+1,
                                 core_seconds=core_seconds, pipeline_seconds=pipeline_seconds))
        report['methods'][name] = dict(device=str(method_device),
                                      dtype='float64' if network is None else 'float32', rows=rows,
                                      core=latency_summary(rows, 'core_seconds'),
                                      pipeline=latency_summary(rows, 'pipeline_seconds'))
    gs = report['methods']['gs_cpu']
    report['gs_over_neural_cpu_latency_ratio'] = {
        key:gs[key]['median_seconds']/report['methods']['neural_cpu'][key]['median_seconds']
        for key in ('core', 'pipeline')}
    if 'neural_cuda' in report['methods']:
        report['gs_cpu_over_neural_cuda_deployment_latency_ratio'] = {
            key:gs[key]['median_seconds']/report['methods']['neural_cuda'][key]['median_seconds']
            for key in ('core', 'pipeline')}
    return report
