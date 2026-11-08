"""Четырёхплоскостной Gerchberg–Saxton: расчёт маски, PNG и метрики."""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

from .fresnel import FresnelSingleFFT
from .phase import encode_phase, decode_phase

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISTANCES = (0.0241, 0.0253, 0.0265, 0.0277)


def new_output(root, name):
    """Каждый запуск получает новую папку; прежние результаты сохраняются."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(root) / f"{timestamp}_{name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def read_target(path):
    """Прочитать и проверить одну целевую амплитуду."""
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != (128, 128):
            raise ValueError(f"{path}: expected 128x128 8-bit grayscale PNG (mode L)")
        if "transparency" in image.info or getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"{path}: expected a static PNG without transparency")
        amplitude = np.asarray(image, dtype=np.float64) / 255.0
    if not np.any(amplitude):
        raise ValueError(f"{path}: zero image has no amplitude constraint")
    return torch.from_numpy(amplitude)


def read_targets(paths):
    """Прочитать четыре grayscale PNG как целевые амплитуды от 0 до 1."""
    if len(paths) != 4:
        raise ValueError("exactly four input PNG paths required, ordered by distance")
    return torch.stack([read_target(path) for path in paths])


def image_metrics(a, b):
    """Pearson и глобальный SSIM для двух изображений в диапазоне [0, 1]."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("images must be torch tensors")
    if not a.is_floating_point() or not b.is_floating_point():
        raise TypeError("images must have real floating point dtype")
    if a.shape != b.shape or a.device != b.device or a.numel() == 0:
        raise ValueError("images must have matching nonempty shapes and devices")

    for image in (a, b):
        if not torch.isfinite(image).all():
            raise ValueError("images must be finite and lie in [0,1]")
        if (image < 0).any() or (image > 1).any():
            raise ValueError("images must be finite and lie in [0,1]")

    # Метрики нужны для отчёта, а не для backward; считаем их в float64.
    a = a.detach().to(torch.float64)
    b = b.detach().to(torch.float64)
    mean_a = a.mean()
    mean_b = b.mean()
    centered_a = a - mean_a
    centered_b = b - mean_b
    variance_a = centered_a.square().mean()
    variance_b = centered_b.square().mean()
    covariance = (centered_a * centered_b).mean()

    # У постоянной картинки Pearson не определён. Проверка через минимум
    # и максимум не зависит от ошибки округления среднего.
    constant_a = a.min() == a.max()
    constant_b = b.min() == b.max()
    pearson = None

    if not (constant_a or constant_b):
        normal_variances = (
            variance_a > 0
            and variance_b > 0
            and variance_a * variance_b >= torch.finfo(torch.float64).tiny
        )
        if normal_variances:
            correlation = covariance / torch.sqrt(variance_a * variance_b)
        else:
            # Очень малые значения сначала масштабируем: Pearson от этого
            # не меняется, а квадраты перестают обнуляться численно.
            scaled_a = a / a.max()
            scaled_b = b / b.max()
            scaled_a = scaled_a - scaled_a.mean()
            scaled_b = scaled_b - scaled_b.mean()
            scaled_covariance = (scaled_a * scaled_b).mean()
            scaled_variance_a = scaled_a.square().mean()
            scaled_variance_b = scaled_b.square().mean()
            correlation = scaled_covariance / torch.sqrt(
                scaled_variance_a * scaled_variance_b
            )
        pearson = float(correlation.clamp(-1, 1))

    # Глобальная формула SSIM из постановки, data_range = 1.
    brightness_numerator = 2 * mean_a * mean_b + 0.01**2
    structure_numerator = 2 * covariance + 0.03**2
    brightness_denominator = mean_a**2 + mean_b**2 + 0.01**2
    structure_denominator = variance_a + variance_b + 0.03**2
    ssim = (brightness_numerator * structure_numerator) / (
        brightness_denominator * structure_denominator
    )

    return {"pearson": pearson, "ssim_global": float(ssim)}


def amplitude_codes(amplitude):
    """Normalize each amplitude image and encode to the same 8-bit PNG levels."""
    peak = amplitude.amax((-2, -1), keepdim=True)
    return torch.round(255 * (amplitude / peak.clamp_min(1e-300))).clamp(0, 255).byte()


def _check_targets(targets):
    """Baseline работает с четырьмя ненулевыми амплитудами CPU float64."""
    message = "four finite nonnegative nonzero CPU float64 amplitudes required"
    if not isinstance(targets, torch.Tensor):
        raise ValueError(message)
    if targets.shape != (4, 128, 128):
        raise ValueError(message)
    if targets.dtype != torch.float64 or targets.device.type != "cpu":
        raise ValueError(message)
    if not torch.isfinite(targets).all() or (targets < 0).any():
        raise ValueError(message)
    if (targets.amax((-2, -1)) <= 0).any():
        raise ValueError(message)


def _amplitude_error(phase, operators, target_amplitudes):
    """Средняя нормированная ошибка амплитуды по четырём плоскостям."""
    field = torch.exp(1j * phase)
    errors = []
    for operator, target in zip(operators, target_amplitudes):
        prediction = operator(field).abs()
        squared_error = (prediction - target).square().sum()
        target_energy = target.square().sum()
        errors.append(squared_error / target_energy)
    return float(torch.stack(errors).mean())


def gs_step(phase, operators, target_amplitudes):
    """Одна параллельная итерация GS. Цели заданы в физических амплитудах."""
    field = torch.exp(1j * phase)
    backward_fields = []
    for operator, target in zip(operators, target_amplitudes):
        output_phase = torch.angle(operator(field))
        corrected_field = target * torch.exp(1j * output_phase)
        backward_fields.append(operator.inverse(corrected_field))
    return torch.angle(torch.stack(backward_fields).mean(0))


@torch.no_grad()
def synthesize(targets, *, iterations=200, seed=0, record_history=True):
    """Рассчитать одну фазовую маску для четырёх целевых амплитуд."""
    _check_targets(targets)
    if type(iterations) is not int or iterations < 0:
        raise ValueError("nonnegative integer iterations required")

    # По одному оператору для каждого расстояния.
    operators = []
    for distance in DISTANCES:
        operator = FresnelSingleFFT((128, 128), distance=distance)
        operators.append(operator)

    # Одинаковое единичное освещение по всей входной апертуре.
    incident = torch.ones((128, 128), dtype=torch.float64, device="cpu")
    incoming_power = incident.square().sum() * operators[0].dx * operators[0].dy

    # Нормируем мощность каждой цели, учитывая её выходной шаг.
    target_amplitudes = []
    for target, operator in zip(targets, operators):
        maximum = target.max()
        if maximum < 1e-100 or maximum > 1e100:
            target = target / maximum

        target_power = target.square().sum() * operator.dx_out * operator.dy_out
        amplitude = target * torch.sqrt(incoming_power / target_power)
        target_amplitudes.append(amplitude)
    target_amplitudes = torch.stack(target_amplitudes)

    # Локальный генератор делает запуск воспроизводимым по seed.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_values = torch.rand(
        (128, 128), dtype=torch.float64, device="cpu", generator=generator
    )
    phase = 2 * math.pi * random_values - math.pi

    history = []
    start_time = time.perf_counter()
    if record_history:
        initial_error = _amplitude_error(phase, operators, target_amplitudes)
        history.append({"iteration": 0, "amplitude_nmse": initial_error})

    for step in range(1, iterations + 1):
        phase = gs_step(phase, operators, target_amplitudes)
        if record_history:
            error = _amplitude_error(phase, operators, target_amplitudes)
            history.append({"iteration": step, "amplitude_nmse": error})

    seconds = time.perf_counter() - start_time
    return phase, operators, history, seconds


def _reconstruct(phase, operators):
    """Получить четыре физических модуля поля из одной фазы ДОЭ."""
    field = torch.exp(1j * phase)
    amplitudes = []
    for operator in operators:
        amplitudes.append(operator(field).abs())
    return torch.stack(amplitudes)


def run(paths, output="outputs", iterations=200, seed=0):
    """Полный запуск: прочитать цели, рассчитать маску и сохранить результат."""
    torch.set_num_threads(1)
    targets = read_targets(paths)
    phase, operators, history, seconds = synthesize(
        targets, iterations=iterations, seed=seed
    )
    root = new_output(output, "fresnel_four_plane_gs")

    encoded_phase = encode_phase(phase)
    Image.fromarray(encoded_phase.numpy()).save(root / "doe_phase.png")

    # Проверяем именно экспортированную PNG-маску, читая её заново.
    with Image.open(root / "doe_phase.png") as image:
        saved_codes = torch.from_numpy(np.array(image))
    decoded_phase = decode_phase(saved_codes)

    amplitudes = _reconstruct(decoded_phase, operators)
    continuous_amplitudes = _reconstruct(phase, operators)

    rows = []
    figure, axes = plt.subplots(4, 2, figsize=(7, 12), constrained_layout=True)
    for index, (target, amplitude, operator) in enumerate(
        zip(targets, amplitudes, operators)
    ):
        number = index + 1

        # Нормировка для PNG; физические модули сохраняются отдельно в NPZ.
        reconstruction_codes = amplitude_codes(amplitude)
        target_codes = torch.round(target * 255).to(torch.uint8)
        Image.fromarray(reconstruction_codes.numpy()).save(
            root / f"reconstruction_{number}.png"
        )
        Image.fromarray(target_codes.numpy()).save(root / f"target_{number}.png")

        with Image.open(root / f"reconstruction_{number}.png") as image:
            restored = torch.from_numpy(np.array(image)).to(torch.float64) / 255
        metrics = image_metrics(target / target.max(), restored)
        continuous = continuous_amplitudes[index]
        continuous_metrics = image_metrics(target / target.max(), continuous / continuous.max())
        row = {
            "distance_m": operator.distance,
            "dx_out_m": operator.dx_out,
            "dy_out_m": operator.dy_out,
            "pearson": metrics["pearson"],
            "ssim_global": metrics["ssim_global"],
            "continuous_phase_metrics": continuous_metrics,
        }
        rows.append(row)

        axes[index, 0].imshow(target.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[index, 0].set_title(f"Target {number}")
        axes[index, 1].imshow(restored.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[index, 1].set_title(
            f"z={operator.distance * 1000:g} mm, quantized DOE"
        )
    figure.savefig(root / "comparison.png", dpi=140)
    plt.close(figure)

    iteration_numbers = []
    errors = []
    for entry in history:
        iteration_numbers.append(entry["iteration"])
        errors.append(entry["amplitude_nmse"])
    figure, axis = plt.subplots()
    axis.plot(iteration_numbers, errors)
    axis.set(xlabel="Iteration", ylabel="Mean amplitude NMSE")
    figure.savefig(root / "convergence.png", dpi=140)
    plt.close(figure)

    np.savez_compressed(
        root / "fields.npz",
        phase=phase.numpy(),
        decoded_phase=decoded_phase.numpy(),
        amplitude=amplitudes.numpy(),
        continuous_amplitude=continuous_amplitudes.numpy(),
        targets=targets.numpy(),
    )

    source_paths = []
    for path in paths:
        source_paths.append(str(Path(path).resolve()))
    report = {
        "method": "parallel multi-plane GS, equal plane weights",
        "iterations": iterations,
        "seed": seed,
        "seconds": seconds,
        "wavelength_m": 532e-9,
        "input_pitch_m": 10e-6,
        "shape": [128, 128],
        "illumination": "uniform unit amplitude, full square aperture",
        "input_interpretation": "PNG gray values are target amplitudes",
        "png_reconstruction_scaling": "each modulus divided by its own maximum, rounded to uint8",
        "ssim": "global population SSIM, range=1, K1=0.01 K2=0.03 (not local-window SSIM)",
        "phase_mapping": "0 -> -pi, 255 -> +pi; round to nearest, 256 codes",
        "source_paths": source_paths,
        "planes": rows,
        "history": history,
    }
    report_text = json.dumps(report, indent=2, allow_nan=False)
    (root / "result.json").write_text(report_text + "\n")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    result_folder = run(args.images, args.output, args.iterations, args.seed)
    print(result_folder)


if __name__ == "__main__":
    main()
