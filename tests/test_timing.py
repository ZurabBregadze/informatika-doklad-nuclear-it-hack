import torch

from doe_p0.fresnel_baseline import synthesize
from doe_p0.timing import timed_call


def test_gs_history_toggle_preserves_phase_and_global_rng():
    targets = torch.rand(4, 128, 128, dtype=torch.float64)
    rng = torch.get_rng_state().clone()
    phase, _, history, _ = synthesize(targets, iterations=3, seed=11)
    fast, _, no_history, _ = synthesize(targets, iterations=3, seed=11, record_history=False)
    assert torch.equal(phase, fast)
    assert len(history) == 4 and no_history == []
    assert torch.equal(rng, torch.get_rng_state())


def test_cuda_latency_synchronizes_before_and_after_call(monkeypatch):
    import doe_p0.timing as timing
    events = []
    ticks = iter([10., 10.25])
    def clock():
        events.append('clock')
        return next(ticks)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device:events.append('sync'))
    monkeypatch.setattr(timing.time, 'perf_counter', clock)
    def work():
        events.append('work')
        return 42
    result, seconds = timed_call(work, 'cuda')
    assert result == 42 and seconds == .25
    assert events == ['sync', 'clock', 'work', 'sync', 'clock']
