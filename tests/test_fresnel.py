import math
import numpy as np
import pytest
import torch
from doe_p0.fresnel import FresnelSingleFFT
from doe_p0.fresnel_baseline import encode_phase,decode_phase,image_metrics,synthesize

@pytest.mark.numerical
@pytest.mark.parametrize('shape',[(5,6),(6,5),(8,8)])
def test_fresnel_direct_integral(shape):
    h,w=shape;dx=10e-6;dy=12e-6;z=.0241;lam=532e-9
    op=FresnelSingleFFT(shape,lam,dx,dy,z)
    u=torch.randn(shape,dtype=torch.complex128)
    y=(np.arange(h)-h//2)*dy;x=(np.arange(w)-w//2)*dx
    yo=(np.arange(h)-h//2)*op.dy_out;xo=(np.arange(w)-w//2)*op.dx_out
    oracle=np.zeros(shape,dtype=complex)
    for i,b in enumerate(yo):
        for j,a in enumerate(xo):
            oracle[i,j]=np.sum(u.numpy()*np.exp(1j*np.pi*((a-x[None,:])**2+(b-y[:,None])**2)/(lam*z)))*dx*dy*np.exp(2j*np.pi*z/lam)/(1j*lam*z)
    assert np.allclose(op(u).numpy(),oracle,rtol=1e-10,atol=1e-10)

@pytest.mark.numerical
@pytest.mark.parametrize('distance',[.0241,.0253,.0265,.0277])
def test_inverse_energy_adjoint(distance):
    op=FresnelSingleFFT((128,128),distance=distance)
    u=torch.randn((2,128,128),dtype=torch.complex128);v=torch.randn_like(u)
    assert torch.allclose(op.inverse(op(u)),u,atol=1e-13)
    assert torch.allclose(op(u).abs().square().sum()*op.dx_out*op.dy_out,u.abs().square().sum()*op.dx*op.dy,rtol=1e-13)
    assert torch.allclose(torch.vdot(op(u).flatten(),v.flatten()),torch.vdot(u.flatten(),op.adjoint(v).flatten()),atol=1e-10)

@pytest.mark.numerical
def test_fresnel_gradients():
    op=FresnelSingleFFT((4,5))
    phase=torch.randn((4,5),dtype=torch.float64,requires_grad=True)
    weights=torch.rand_like(phase)
    assert torch.autograd.gradcheck(lambda p:(op(torch.exp(1j*p)).abs().square()*weights).sum(),(phase,))

@pytest.mark.physical
@pytest.mark.parametrize('z',[.0241,.0253,.0265,.0277])
def test_fresnel_analytic_complex_gaussian(z):
    n=128;dx=10e-6;lam=532e-9;waist=100e-6;f=.025
    op=FresnelSingleFFT((n,n),distance=z)
    x=(torch.arange(n,dtype=torch.float64)-n//2)*dx
    a=1/waist**2+1j*math.pi/(lam*f)
    incoming=torch.exp(-a*(x[:,None]**2+x[None,:]**2))
    xo=(torch.arange(n,dtype=torch.float64)-n//2)*op.dx_out
    b=1+1j*a*lam*z/math.pi
    analytic=np.exp(2j*math.pi*z/lam)/b*torch.exp(-a*(xo[:,None]**2+xo[None,:]**2)/b)
    assert torch.linalg.vector_norm(op(incoming)-analytic)/torch.linalg.vector_norm(analytic)<1e-9

@pytest.mark.software
def test_png_phase_and_metrics():
    codes=torch.arange(256,dtype=torch.uint8)
    assert torch.equal(encode_phase(decode_phase(codes)),codes)
    assert float(decode_phase(codes)[0])==pytest.approx(-math.pi)
    assert float(decode_phase(codes)[-1])==pytest.approx(math.pi)
    x=torch.rand((128,128),dtype=torch.float64)
    m=image_metrics(x,x)
    assert m['pearson']==pytest.approx(1) and m['ssim_global']==pytest.approx(1)
    assert image_metrics(torch.ones_like(x),x)['pearson'] is None

@pytest.mark.software
def test_multiplane_reproducibility():
    targets=torch.rand((4,128,128),dtype=torch.float64,generator=torch.Generator().manual_seed(0))
    a,_,ha,_=synthesize(targets,iterations=3,seed=4)
    b,_,hb,_=synthesize(targets,iterations=3,seed=4)
    assert torch.equal(a,b) and ha==hb
    assert ha[-1]['amplitude_nmse']<ha[0]['amplitude_nmse']


@pytest.mark.numerical
@pytest.mark.parametrize('shape',[(1,1),(1,7),(7,1),(7,9)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_polar_batch_against_dense_spatial_kernel(shape,dtype):
    """Independent separable quadratic kernel, including singleton dimensions."""
    h,w=shape;dx=10e-6;dy=13e-6;z=.0277;lam=532e-9
    operator=FresnelSingleFFT(shape,lam,dx,dy,z,dtype=dtype)
    generator=torch.Generator().manual_seed(41)
    magnitude=torch.rand((2,h,w),dtype=dtype,generator=generator)+.1
    phase=torch.linspace(-2.,3.,2*h*w,dtype=dtype).reshape(2,h,w)
    field=magnitude*torch.exp(1j*phase)
    # Preserve the shape while exercising noncontiguous input storage.
    field=field.transpose(-2,-1).contiguous().transpose(-2,-1)
    x=(np.arange(w)-w//2)*dx;y=(np.arange(h)-h//2)*dy
    xo=(np.arange(w)-w//2)*operator.dx_out;yo=(np.arange(h)-h//2)*operator.dy_out
    kernel_x=np.exp(1j*np.pi*(xo[:,None]-x[None,:])**2/(lam*z))
    kernel_y=np.exp(1j*np.pi*(yo[:,None]-y[None,:])**2/(lam*z))
    coefficient=dx*dy*np.exp(2j*np.pi*z/lam)/(1j*lam*z)
    expected=np.stack([coefficient*kernel_y@sample.numpy().astype(np.complex128)@kernel_x.T for sample in field])
    actual=operator(field)
    tolerance=5e-7 if dtype==torch.float32 else 1e-11
    relative=np.linalg.norm(actual.numpy()-expected)/np.linalg.norm(expected)
    assert relative<tolerance
    assert torch.linalg.vector_norm(operator.inverse(actual)-field)/torch.linalg.vector_norm(field)<tolerance
    incoming=field.abs().double().square().sum((-2,-1))*dx*dy
    outgoing=actual.abs().double().square().sum((-2,-1))*operator.dx_out*operator.dy_out
    assert torch.allclose(incoming,outgoing,rtol=tolerance,atol=0.)


@pytest.mark.software
@pytest.mark.parametrize('kwargs',[
    {'distance':1e-308}, {'distance':1e308}, {'dx':1e-308},
    {'wavelength':1e-308}, {'wavelength':1e-200,'distance':1e200},
    {'wavelength':1.,'distance':1.,'dx':1e154,'dy':1e-154},
])
def test_unrepresentable_derived_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError,match='representable|overflows'):
        FresnelSingleFFT((2,2),**kwargs)


@pytest.mark.software
@pytest.mark.parametrize('pitch',[1e-23,1e-20,1e20])
def test_scale_representability_respects_selected_precision(pitch):
    # All these parameters and the resulting coefficients fit float64. The
    # forward coefficient or its reciprocal does not fit complex64.
    with pytest.raises(ValueError,match='not representable'):
        FresnelSingleFFT((1,1),wavelength=1.,distance=1.,dx=pitch,dy=pitch,dtype=torch.float32)
    operator=FresnelSingleFFT((1,1),wavelength=1.,distance=1.,dx=pitch,dy=pitch,dtype=torch.float64)
    field=torch.ones((1,1),dtype=torch.complex128)
    assert torch.allclose(operator.inverse(operator(field)),field,rtol=1e-14,atol=0.)


@pytest.mark.software
def test_overflow_in_complex_reciprocal_is_rejected():
    with pytest.raises(ValueError,match='not representable'):
        FresnelSingleFFT((1,1),wavelength=1.,distance=1.125,dx=1.3e154,dy=1.3e154)


@pytest.mark.software
def test_fresnel_input_contracts():
    operator=FresnelSingleFFT((4,5))
    with pytest.raises(TypeError,match='torch tensor'):
        operator(np.ones((4,5),dtype=np.complex128))
    for field in (torch.ones((4,5)),torch.ones((4,5),dtype=torch.complex64),
                  torch.ones((4,6),dtype=torch.complex128)):
        with pytest.raises(ValueError,match='matching operator'):
            operator(field)
    for shape in (None,4,(4,0),(4,True)):
        with pytest.raises(ValueError,match='shape'):
            FresnelSingleFFT(shape)
    for kwargs in ({'wavelength':float('nan')},{'dx':float('inf')},{'dy':0.},
                   {'distance':-1.},{'wavelength':True}):
        with pytest.raises(ValueError,match='finite positive real scalar'):
            FresnelSingleFFT((4,5),**kwargs)


@pytest.mark.numerical
def test_feasible_four_plane_gs_fixed_point():
    """Known jointly feasible targets must not be altered by GS projections."""
    seed=4
    initial=2*math.pi*torch.rand((128,128),dtype=torch.float64,
                               generator=torch.Generator().manual_seed(seed))-math.pi
    field=torch.exp(1j*initial)
    targets=torch.stack([FresnelSingleFFT((128,128),distance=z)(field).abs()
                         for z in (.0241,.0253,.0265,.0277)])
    result,_,history,_=synthesize(targets,iterations=2,seed=seed)
    assert torch.allclose(torch.exp(1j*result),field,rtol=0.,atol=1e-13)
    assert max(row['amplitude_nmse'] for row in history)<1e-25


@pytest.mark.numerical
def test_numpy_single_precision_parameters_are_promoted_before_arithmetic():
    parameters=dict(wavelength=np.float32(532e-9),dx=np.float32(10e-6),
                    dy=np.float32(13e-6),distance=np.float32(.0241))
    operator=FresnelSingleFFT((128,128),**parameters)
    reference=FresnelSingleFFT((128,128),**{key:float(value) for key,value in parameters.items()})
    field=torch.ones((128,128),dtype=torch.complex128)
    assert torch.equal(operator(field),reference(field))


@pytest.mark.software
def test_explicit_cpu_operator_ignores_global_default_device():
    with torch.device('meta'):
        operator=FresnelSingleFFT((4,5),device='cpu')
    assert all(buffer.device.type=='cpu' for buffer in operator.buffers())
