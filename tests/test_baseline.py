"""Strict public contracts and quantized four-plane baseline regression checks."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from doe_p0.fresnel_baseline import (
    decode_phase, encode_phase, image_metrics, read_targets, run, synthesize,
)

pytestmark=pytest.mark.software


@pytest.mark.numerical
def test_normalized_amplitude_loss_is_nonincreasing():
    generator=torch.Generator().manual_seed(8)
    targets=torch.rand((4,128,128),dtype=torch.float64,generator=generator)
    _,_,history,_=synthesize(targets,iterations=50,seed=5)
    losses=np.array([row['amplitude_nmse'] for row in history])
    assert np.all(np.diff(losses)<=1e-12)


def test_all_constant_png_levels_have_undefined_pearson():
    other=torch.linspace(0,1,128*128,dtype=torch.float64).reshape(128,128)
    for code in range(256):
        image=torch.full_like(other,code/255)
        assert image_metrics(image,other)['pearson'] is None
        assert image_metrics(other,image)['pearson'] is None
        assert image_metrics(image,image)=={'pearson':None,'ssim_global':1.0}


def test_metrics_reject_broadcasting_and_invalid_data():
    image=torch.rand((8,8),dtype=torch.float64)
    for other in (image[0],torch.empty((0,8),dtype=torch.float64)):
        with pytest.raises(ValueError,match='matching nonempty'):
            image_metrics(image,other)
    for other in (image.to(torch.uint8),image.to(torch.complex128)):
        with pytest.raises(TypeError,match='real floating'):
            image_metrics(image,other)
    for other in (image*float('nan'),image*float('inf'),-image,image+1):
        with pytest.raises(ValueError,match='finite and lie'):
            image_metrics(image,other)
        with pytest.raises(ValueError,match='finite and lie'):
            image_metrics(other,image)
    with pytest.raises(ValueError,match='matching nonempty'):
        image_metrics(torch.empty(0),torch.empty(0))


def test_float32_metrics_use_double_statistics():
    a=torch.linspace(0,1,128*128,dtype=torch.float32).reshape(128,128)
    b=a.flip(0)
    assert image_metrics(a,b)==image_metrics(a.double(),b.double())


@pytest.mark.parametrize('scale',[1e-80,1e-200])
def test_pearson_for_nonconstant_tiny_amplitudes_does_not_underflow(scale):
    base=torch.linspace(0,1,128,dtype=torch.float64)
    a=base*scale
    assert image_metrics(a,a)['pearson']==pytest.approx(1.)
    assert image_metrics(a,a.flip(0))['pearson']==pytest.approx(-1.)
    assert image_metrics(a,base.square()*scale)['pearson']==pytest.approx(image_metrics(base,base.square())['pearson'],abs=1e-14)


def test_pearson_at_smallest_subnormal_amplitude():
    a=torch.tensor([0.,torch.finfo(torch.float64).smallest_normal],dtype=torch.float64)
    a[1]=torch.nextafter(torch.tensor(0.,dtype=torch.float64),torch.tensor(1.,dtype=torch.float64))
    assert image_metrics(a,a.flip(0))['pearson']==pytest.approx(-1.)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_phase_endpoint_codes(dtype):
    phase=torch.tensor([-math.pi,math.pi],dtype=dtype)
    assert torch.equal(encode_phase(phase),torch.tensor([0,255],dtype=torch.uint8))


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_phase_codec_promotes_before_complex_exponential(dtype):
    phase=torch.tensor([-math.pi,math.pi,4.],dtype=dtype)
    encoded=encode_phase(phase)
    assert encoded[:2].tolist()==[0,255]
    decoded=decode_phase(encoded)
    assert (torch.exp(1j*decoded)-torch.exp(1j*phase.double())).abs().max()<math.pi/255+1e-6


def test_cpu_baseline_ignores_global_default_device():
    target=torch.ones((4,128,128),dtype=torch.float64,device='cpu')
    with torch.device('meta'):
        phase,_,history,_=synthesize(target,iterations=1)
    assert phase.device.type=='cpu'
    assert np.isfinite([row['amplitude_nmse'] for row in history]).all()


def test_phase_codec_all_codes_and_invalid_data():
    codes=torch.arange(256,dtype=torch.uint8)
    assert torch.equal(encode_phase(decode_phase(codes)),codes)
    for dtype in (torch.float32,torch.float64):
        huge=torch.tensor([-1e8,1e8],dtype=dtype)
        decoded=decode_phase(encode_phase(huge))
        # Quantization has a maximum half-bin phase error, apart from roundoff.
        assert (torch.exp(1j*decoded)-torch.exp(1j*huge).to(torch.complex128)).abs().max()<math.pi/255+1e-6
    for values in ([float('nan')],[float('inf')],[-float('inf')]):
        with pytest.raises(ValueError,match='finite'):
            encode_phase(torch.tensor(values))
    for value in (torch.tensor([1]),torch.tensor([1j])):
        with pytest.raises(TypeError,match='real floating'):
            encode_phase(value)
    for value in (torch.tensor([-1,256]),torch.tensor([0.,255.]),np.array([0,255],dtype=np.uint8)):
        with pytest.raises(TypeError,match='uint8 tensor'):
            decode_phase(value)


@pytest.mark.parametrize('scale',[1e-200,1e200])
def test_extreme_amplitude_scale_is_finite_and_equivalent(scale):
    base=torch.linspace(.1,1,4*128*128,dtype=torch.float64).reshape(4,128,128)
    phase,_,history,_=synthesize(base,iterations=2,seed=6)
    scaled,_,scaled_history,_=synthesize(base*scale,iterations=2,seed=6)
    assert torch.isfinite(scaled).all()
    assert torch.allclose(torch.exp(1j*phase),torch.exp(1j*scaled),atol=1e-12,rtol=1e-12)
    assert np.isfinite([r['amplitude_nmse'] for r in scaled_history]).all()
    assert np.allclose([r['amplitude_nmse'] for r in history],[r['amplitude_nmse'] for r in scaled_history],rtol=1e-12,atol=1e-12)


def test_zero_target_plane_rejected():
    target=torch.ones((4,128,128),dtype=torch.float64)
    target[2]=0
    with pytest.raises(ValueError,match='nonzero CPU float64'):
        synthesize(target)


def test_png_input_contract(tmp_path):
    valid=tmp_path/'valid.png'
    Image.fromarray(np.full((128,128),3,dtype=np.uint8)).save(valid)
    targets=read_targets([valid]*4)
    assert targets.shape==(4,128,128) and targets.dtype==torch.float64
    for name,array in [('rgb',np.ones((128,128,3),dtype=np.uint8)),
                       ('small',np.ones((127,128),dtype=np.uint8)),
                       ('16bit',np.ones((128,128),dtype=np.uint16)),
                       ('zero',np.zeros((128,128),dtype=np.uint8))]:
        path=tmp_path/f'{name}.png';Image.fromarray(array).save(path)
        with pytest.raises(ValueError):read_targets([path]*4)
    with pytest.raises(ValueError,match='exactly four'):read_targets([valid]*3)


def test_transparent_or_animated_grayscale_png_rejected(tmp_path):
    image=Image.fromarray(np.full((128,128),255,dtype=np.uint8))
    transparent=tmp_path/'transparent.png'
    image.save(transparent,transparency=255)
    with Image.open(transparent) as saved:
        assert saved.mode=='L' and 'transparency' in saved.info
    with pytest.raises(ValueError,match='static PNG without transparency'):
        read_targets([transparent]*4)
    animated=tmp_path/'animated.png'
    second=Image.fromarray(np.full((128,128),127,dtype=np.uint8))
    image.save(animated,save_all=True,append_images=[second],duration=100,loop=0)
    with Image.open(animated) as saved:
        assert saved.mode=='L' and saved.n_frames==2
    with pytest.raises(ValueError,match='static PNG without transparency'):
        read_targets([animated]*4)


def test_exported_png_roundtrip_matches_saved_fields_and_metrics(tmp_path):
    generator=np.random.default_rng(5)
    paths=[]
    for i in range(4):
        path=tmp_path/f'input{i}.png'
        Image.fromarray(generator.integers(0,256,(128,128),dtype=np.uint8)).save(path)
        paths.append(path)
    output=run(paths,tmp_path/'outputs',iterations=2,seed=3)
    report=json.loads((output/'result.json').read_text())
    with Image.open(output/'doe_phase.png') as image:
        assert image.mode=='L' and image.size==(128,128)
        phase=decode_phase(torch.from_numpy(np.array(image)))
    from doe_p0.fresnel import FresnelSingleFFT
    targets=read_targets(paths)
    with np.load(output/'fields.npz') as arrays:
        assert np.array_equal(arrays['decoded_phase'],phase.numpy())
        for i,row in enumerate(report['planes']):
            op=FresnelSingleFFT((128,128),distance=row['distance_m'])
            amplitude=op(torch.exp(1j*phase)).abs()
            assert np.array_equal(amplitude.numpy(),arrays['amplitude'][i])
            with Image.open(output/f'reconstruction_{i+1}.png') as image:
                assert image.mode=='L' and image.size==(128,128)
                restored=torch.from_numpy(np.array(image))
            assert torch.equal(restored,torch.round(255*amplitude/amplitude.max()).to(torch.uint8))
            measured=image_metrics(targets[i],restored.double()/255)
            assert measured['pearson']==row['pearson']
            assert measured['ssim_global']==row['ssim_global']


def test_example_against_independent_numpy_propagation(tmp_path):
    """Audit a fresh temporary export with independent NumPy propagation."""
    root=Path(__file__).resolve().parents[1]
    inputs=[root/f'examples/fresnel_targets/{i}.png' for i in range(1,5)]
    output=run(inputs,tmp_path/'outputs',iterations=2,seed=0)
    report=json.loads((output/'result.json').read_text())
    n=128;dx=10e-6;wavelength=532e-9
    coordinates=(np.arange(n)-n//2)*dx
    with Image.open(output/'doe_phase.png') as image:
        assert image.mode=='L' and image.size==(n,n)
        codes=np.array(image)
    phase=codes.astype(np.float64)*(2*np.pi/255)-np.pi
    with np.load(output/'fields.npz') as saved:
        assert set(saved.files)=={'phase','decoded_phase','amplitude','continuous_amplitude','targets'}
        for key in saved.files:assert np.isfinite(saved[key]).all()
        assert np.array_equal(phase,saved['decoded_phase'])
        assert saved['phase'].shape==(n,n)
        assert np.max(np.abs(saved['phase']))<=np.pi
        expected_codes=np.rint((saved['phase']+np.pi)*(255/(2*np.pi))).clip(0,255).astype(np.uint8)
        assert np.array_equal(codes,expected_codes)
        errors=[]
        for i,(row,z) in enumerate(zip(report['planes'],(.0241,.0253,.0265,.0277))):
            pitch=wavelength*z/(n*dx)
            assert row['distance_m']==z and row['dx_out_m']==pitch and row['dy_out_m']==pitch
            out_coordinates=(np.arange(n)-n//2)*pitch
            input_chirp=np.exp(1j*np.pi*(coordinates[:,None]**2+coordinates[None,:]**2)/(wavelength*z))
            output_chirp=np.exp(1j*np.pi*(out_coordinates[:,None]**2+out_coordinates[None,:]**2)/(wavelength*z))
            coefficient=dx**2*n*np.exp(2j*np.pi*z/wavelength)/(1j*wavelength*z)
            def propagate(p):
                spectrum=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.exp(1j*p)*input_chirp),norm='ortho'))
                return np.abs(coefficient*output_chirp*spectrum)
            predicted=propagate(phase)
            continuous=propagate(saved['phase'])
            assert np.allclose(predicted,saved['amplitude'][i],rtol=1e-12,atol=1e-12)
            assert np.allclose(continuous,saved['continuous_amplitude'][i],rtol=1e-12,atol=1e-12)
            assert predicted.shape==(n,n)
            assert np.square(predicted).sum()*pitch**2==pytest.approx(n*n*dx**2,rel=1e-12)
            with Image.open(root/f'examples/fresnel_targets/{i+1}.png') as image:target_codes=np.array(image)
            with Image.open(output/f'target_{i+1}.png') as image:assert np.array_equal(np.array(image),target_codes)
            target=target_codes.astype(np.float64)/255
            assert np.array_equal(target,saved['targets'][i])
            with Image.open(output/f'reconstruction_{i+1}.png') as image:
                assert image.mode=='L' and image.size==(n,n)
                restored_codes=np.array(image)
            assert np.array_equal(restored_codes,np.rint(255*predicted/predicted.max()).clip(0,255).astype(np.uint8))
            restored=restored_codes.astype(np.float64)/255
            ac=target-target.mean();bc=restored-restored.mean()
            va=(ac**2).mean();vb=(bc**2).mean();cov=(ac*bc).mean()
            correlation=cov/np.sqrt(va*vb)
            ssim=((2*target.mean()*restored.mean()+.01**2)*(2*cov+.03**2))/((target.mean()**2+restored.mean()**2+.01**2)*(va+vb+.03**2))
            assert row['pearson']==pytest.approx(correlation,abs=1e-12)
            assert row['ssim_global']==pytest.approx(ssim,abs=1e-12)
            normalized=target*np.sqrt((n*n*dx**2)/((target**2).sum()*pitch**2))
            errors.append(((continuous-normalized)**2).sum()/(normalized**2).sum())
        assert len(report['planes'])==4
        assert report['history'][-1]['amplitude_nmse']==pytest.approx(np.mean(errors),abs=1e-12)
    assert [row['iteration'] for row in report['history']]==list(range(report['iterations']+1))
    assert np.isfinite([row['amplitude_nmse'] for row in report['history']]).all()
    for name in ('comparison.png','convergence.png'):
        with Image.open(output/name) as image:image.verify()
