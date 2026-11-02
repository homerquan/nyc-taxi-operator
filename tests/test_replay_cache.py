import numpy as np
import torch
from mobility import inference
from mobility.graph import HOUR


def test_incremental_and_disk_replay_match_fresh(dataset, checkpoint):
    data, _, _ = dataset
    cutoff = 1735743600
    inference._sessions.clear()
    first = inference.forecast(data, checkpoint, cutoff, top_k=9)[0]
    assert (checkpoint / 'replay-cpu.pt').is_file()
    later = inference.forecast(data, checkpoint, cutoff+HOUR, top_k=9)[0]
    inference._sessions.clear()
    from_disk = inference.forecast(data, checkpoint, cutoff+HOUR, top_k=9)[0]
    assert later == from_disk
    inference._sessions.clear()
    (checkpoint / 'replay-cpu.pt').unlink()
    fresh = inference.forecast(data, checkpoint, cutoff+HOUR, top_k=9)[0]
    assert later == fresh
    # Moving backwards must not use future memory.
    earlier = inference.forecast(data, checkpoint, cutoff, top_k=9)[0]
    assert earlier == first
