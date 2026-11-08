"""Одно-FFT преобразование Френеля. Все длины задаются в метрах."""

import math
from numbers import Real

import torch
from torch import nn


def centered_fft(field, inverse=False):
    """Перенести начало координат, выполнить FFT и вернуть центр в середину."""
    shifted_field = torch.fft.ifftshift(field, dim=(-2, -1))

    if inverse:
        spectrum = torch.fft.ifft2(shifted_field, norm="ortho")
    else:
        spectrum = torch.fft.fft2(shifted_field, norm="ortho")

    return torch.fft.fftshift(spectrum, dim=(-2, -1))


def _make_chirp(height, width, dx, dy, wavelength_distance):
    """Квадратичный фазовый множитель exp(i*pi*(x²+y²)/(lambda*z))."""
    y = torch.arange(height, dtype=torch.float64, device="cpu")
    x = torch.arange(width, dtype=torch.float64, device="cpu")
    y = (y - height // 2) * dy
    x = (x - width // 2) * dx

    # Столбец y и строка x образуют двумерную сетку при сложении.
    y = y.reshape(height, 1)
    x = x.reshape(1, width)
    argument = 1j * math.pi * (y**2 + x**2) / wavelength_distance

    if not torch.isfinite(argument).all():
        raise ValueError(
            "quadratic chirp phase overflows float64 for these optical parameters"
        )

    return torch.exp(argument)


def _check_positive_number(name, value):
    """Проверить параметр, введённый пользователем."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real scalar")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real scalar")


def _check_derived_number(name, value):
    """Проверить, что результат вычислений не переполнился и не обнулился."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} is not representable as a finite positive float64 value"
        )


class FresnelSingleFFT(nn.Module):
    """Прямой оператор Френеля на естественной выходной сетке.

    dx_out = wavelength * distance / (width * dx).
    dy_out = wavelength * distance / (height * dy).

    nn.Module нужен для совместимости с PyTorch: register_buffer хранит
    постоянные множители, которые можно переносить на другое устройство.
    Обучаемых параметров у этого оператора нет.
    """

    def __init__(
        self,
        shape,
        wavelength=532e-9,
        dx=10e-6,
        dy=10e-6,
        distance=0.0241,
        *,
        dtype=torch.float64,
        device="cpu",
    ):
        super().__init__()

        # 1. Проверяем входные параметры.
        try:
            valid_shape = len(shape) == 2
            if valid_shape:
                for size in shape:
                    if type(size) is not int or size <= 0:
                        valid_shape = False
        except TypeError:
            valid_shape = False

        if not valid_shape:
            raise ValueError("positive 2D integer shape required")

        _check_positive_number("wavelength", wavelength)
        _check_positive_number("dx", dx)
        _check_positive_number("dy", dy)
        _check_positive_number("distance", distance)

        # Даже NumPy float32 сначала переводим в двойную точность.
        wavelength = float(wavelength)
        dx = float(dx)
        dy = float(dy)
        distance = float(distance)

        if dtype not in (torch.float32, torch.float64):
            raise ValueError("float32 or float64 required")

        self.shape = tuple(shape)
        self.wavelength = wavelength
        self.distance = distance
        self.dx = dx
        self.dy = dy
        height, width = shape

        # 2. Находим физический шаг выходной сетки.
        wavelength_distance = wavelength * distance
        _check_derived_number("wavelength * distance", wavelength_distance)
        _check_derived_number("input pixel area", dx * dy)
        _check_derived_number("input width", width * dx)
        _check_derived_number("input height", height * dy)

        self.dx_out = wavelength_distance / (width * dx)
        self.dy_out = wavelength_distance / (height * dy)
        _check_derived_number("output dx", self.dx_out)
        _check_derived_number("output dy", self.dy_out)
        _check_derived_number("output pixel area", self.dx_out * self.dy_out)

        # 3. Строим входной и выходной квадратичные фазовые множители.
        if dtype == torch.float64:
            complex_dtype = torch.complex128
        else:
            complex_dtype = torch.complex64

        input_chirp = _make_chirp(height, width, dx, dy, wavelength_distance)
        output_chirp = _make_chirp(
            height, width, self.dx_out, self.dy_out, wavelength_distance
        )
        input_chirp = input_chirp.to(device=device, dtype=complex_dtype)
        output_chirp = output_chirp.to(device=device, dtype=complex_dtype)
        self.register_buffer("input_chirp", input_chirp)
        self.register_buffer("output_chirp", output_chirp)

        # 4. Общий коэффициент интеграла и нормировки FFT.
        carrier_phase = 2 * math.pi * distance / wavelength
        if not math.isfinite(carrier_phase):
            raise ValueError(
                "global carrier phase overflows float64 for these optical parameters"
            )

        carrier = complex(math.cos(carrier_phase), math.sin(carrier_phase)) / 1j
        scale = carrier * dx * dy * math.sqrt(height * width) / wavelength_distance
        coefficient = torch.tensor(scale, dtype=complex_dtype, device="cpu")
        reciprocal = coefficient.reciprocal()

        if not torch.isfinite(coefficient) or coefficient == 0:
            raise ValueError(
                f"Fresnel scale or its inverse is not representable in {complex_dtype}"
            )
        if not torch.isfinite(reciprocal) or reciprocal == 0:
            raise ValueError(
                f"Fresnel scale or its inverse is not representable in {complex_dtype}"
            )

        self.register_buffer("factor", coefficient.to(device=device))

    def _check(self, field):
        """Поле должно соответствовать настройкам созданного оператора."""
        if not isinstance(field, torch.Tensor):
            raise TypeError("field must be a torch tensor")

        valid_layout = field.layout == torch.strided
        valid_shape = field.shape[-2:] == self.shape
        valid_dtype = field.dtype == self.input_chirp.dtype
        valid_device = field.device == self.input_chirp.device

        if not (valid_layout and valid_shape and valid_dtype and valid_device):
            raise ValueError(
                "field must be dense with shape, complex precision and device matching operator"
            )

    def forward(self, field):
        """Входной chirp → один FFT → выходной chirp и общий коэффициент."""
        self._check(field)
        prepared_field = field * self.input_chirp
        spectrum = centered_fft(prepared_field)
        return self.factor * self.output_chirp * spectrum

    def inverse(self, field):
        """Вернуть поле с выходной сетки на исходную входную сетку."""
        self._check(field)
        prepared_field = field * self.output_chirp.conj() / self.factor
        restored_field = centered_fft(prepared_field, inverse=True)
        return self.input_chirp.conj() * restored_field

    def adjoint(self, field):
        """Сопряжённый оператор для обычного дискретного скалярного произведения."""
        self._check(field)
        prepared_field = field * self.output_chirp.conj()
        restored_field = centered_fft(prepared_field, inverse=True)
        return self.factor.conj() * self.input_chirp.conj() * restored_field
