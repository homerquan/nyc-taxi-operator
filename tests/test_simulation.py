from dataclasses import replace
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app, cached_forecast
from mobility.simulation import FleetSimulation, SimulationConfig, compare_policies, historical_requests

START = 1735743600


def prediction(source=1, target=2, fare=10, duration=10):
    return {'source_zone': source, 'target_zone': target, 'predicted_trips': 5,
            'predicted_fare_usd': fare, 'predicted_duration_min': duration}


def request(i, at=1, fare=10, distance=2, duration=600):
    return {'id': i, 'at': float(at), 'source': 1, 'target': 2,
            'duration': float(duration), 'fare': float(fare), 'distance': float(distance)}


def test_capacity_accounting_and_repeatability(dataset):
    data, zones, _ = dataset
    cfg = SimulationConfig(start=START, hours=1, fleet_size=1, max_wait_minutes=2)
    a = compare_policies(data, cfg, [prediction()])
    b = compare_policies(data, cfg, [prediction()])
    assert a == b
    assert a['request_count'] == 4
    for policy in ('baseline', 'optimized'):
        for s in a[policy]['timeline']:
            assert s['in_service'] <= cfg.fleet_size
            assert s['requested'] == s['accepted'] + s['rejected'] + s['pending']
            assert s['accepted'] == s['completed'] + s['in_service']
            assert s['net_revenue'] == pytest.approx(s['revenue']-s['fuel_cost']-s['driver_cost'], abs=.02)
            assert sum(v['status'] == 'busy' for v in s['vehicles']) == s['in_service']
        assert a[policy]['summary']['pending'] == 0
        assert len({d['id'] for d in a[policy]['decisions']}) == a['request_count']


def test_declines_loss_making_trip_and_books_on_completion(dataset):
    _, zones, _ = dataset
    cfg = SimulationConfig(start=START, hours=1, fleet_size=1, fuel_price=10, mpg=1)
    rows = [request(1, fare=1, distance=10)]
    base = FleetSimulation(cfg, zones, rows, [prediction()], 'baseline', [1]).run()
    ai = FleetSimulation(cfg, zones, rows, [prediction()], 'optimized', [1]).run()
    assert base['summary']['revenue'] == 1
    assert base['summary']['fuel_cost'] == 100
    assert base['summary']['net_revenue'] == -99
    assert ai['summary']['accepted'] == 0
    assert ai['summary']['fuel_cost'] == ai['summary']['revenue'] == 0
    assert ai['summary']['net_revenue'] > base['summary']['net_revenue']


def test_horizon_and_driver_cost(dataset):
    _, zones, _ = dataset
    cfg = SimulationConfig(start=START, hours=1, fleet_size=1, driver_cost_hour=20)
    rows = [request(1, at=3500, duration=1200)]
    base = FleetSimulation(cfg, zones, rows, [prediction()], 'baseline', [1]).run()
    s = base['summary']
    assert s['completed'] == 0 and s['in_service'] == 1
    assert s['revenue'] == 0 and s['fuel_cost'] > 0
    assert s['driver_cost'] == 20


def test_global_batch_matching_selects_higher_margin(dataset):
    _, zones, _ = dataset
    cfg = SimulationConfig(start=START, hours=1, fleet_size=1, max_wait_minutes=1)
    rows = [request(1, fare=5, duration=1200), request(2, at=2, fare=40, duration=1200)]
    ai = FleetSimulation(cfg, zones, rows, [prediction()], 'optimized', [1]).run()
    base = FleetSimulation(cfg, zones, rows, [prediction()], 'baseline', [1]).run()
    assert ai['decisions'][0]['id'] == 2
    assert base['decisions'][0]['id'] == 1
    assert ai['summary']['net_revenue'] > base['summary']['net_revenue']


def test_history_window_scaling_and_validation(dataset):
    data, _, _ = dataset
    cfg = SimulationConfig(start=START, hours=1)
    requests = historical_requests(data, cfg)
    assert len(requests) == 4
    assert all(0 <= r['at'] < 3600 for r in requests)
    doubled = historical_requests(data, replace(cfg, demand_scale=2))
    assert len(doubled) == 8
    with pytest.raises(ValueError, match='cover every hour'):
        historical_requests(data, replace(cfg, hours=2))
    with pytest.raises(ValueError):
        replace(cfg, fuel_price=float('nan'))
    with pytest.raises(ValueError):
        replace(cfg, fleet_size=0)


def test_simulation_api_background_job(dataset, checkpoint, monkeypatch):
    data, _, _ = dataset
    monkeypatch.setenv('NYC_DATA_DIR', str(data))
    monkeypatch.setenv('NYC_ARTIFACTS_DIR', str(checkpoint))
    cached_forecast.cache_clear()
    client = TestClient(app)
    body = {'as_of': '2025-01-01T10:00:00-05:00', 'hours': 1, 'fleet_size': 2}
    response = client.post('/api/simulations', json=body)
    assert response.status_code == 202
    key = response.json()['id']
    for _ in range(200):
        job = client.get(f'/api/jobs/{key}').json()
        if job['status'] in ('complete','failed'):
            break
        time.sleep(.01)
    assert job['status'] == 'complete', job
    assert job['result']['request_count'] == 4
    assert client.get('/api/jobs/missing').status_code == 404
    assert client.post('/api/simulations', json={**body, 'fleet_size': 0}).status_code == 422
    assert client.post('/api/simulations', json={**body, 'as_of': '2025-01-01T10:00:00'}).status_code == 422
    assert client.post('/api/training', json={'epochs': 0}).status_code == 422
    assert client.get('/api/training').status_code == 200


def test_duration_prior_uses_only_completed_history(dataset):
    from mobility.simulation import completed_duration_priors
    data, _, _ = dataset
    assert completed_duration_priors(data, START) == {}
    priors = completed_duration_priors(data, START+3600)
    # The 10:55 departure completes after 11:00 and is unavailable here.
    assert priors[(1, 2)][1] == 2
    assert priors[(1, 2)][0] == pytest.approx(25)
    assert completed_duration_priors(data, START+2*3600)[(1, 2)][1] == 3
