"""Seeded TLC aggregate request replay and forecast-guided SimPy fleet dispatch.

Only the request source reads the replay window's labels. Dispatch receives the
current quote/route and the forecast made at the simulation start, never future
requests or their outcomes. Both policies see the identical request stream.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import duckdb
import numpy as np
import simpy
from scipy.optimize import linear_sum_assignment

from mobility.graph import HOUR, format_time, read_zones
from mobility.preprocess import sql_string


@dataclass(frozen=True)
class SimulationConfig:
    start: int
    hours: int = 2
    fleet_size: int = 80
    fuel_price: float = 3.5
    mpg: float = 25.0
    price_multiplier: float = 1.0
    demand_scale: float = 1.0
    max_wait_minutes: float = 8.0
    driver_cost_hour: float = 0.0
    seed: int = 42

    def __post_init__(self):
        limits = {'hours': (1, 6), 'fleet_size': (1, 300), 'fuel_price': (0, 20),
                  'mpg': (1, 100), 'price_multiplier': (.1, 5), 'demand_scale': (.01, 2),
                  'max_wait_minutes': (1, 30), 'driver_cost_hour': (0, 100), 'seed': (0, 2**31-1)}
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{name} must be between {low} and {high}')
        if self.start % HOUR:
            raise ValueError('Simulation start must be hour-aligned')


def historical_requests(data: Path, cfg: SimulationConfig):
    """Preserve observed OD/quarter-hour counts; sample within each bin.

    Conditional duration, distance and fare use the aggregate's mean, so this
    replays a modeled market of completed TLC trips, not original request logs.
    """
    glob = data / 'processed' / 'targets' / '*.parquet'
    with duckdb.connect() as con:
        frame = con.execute(f'SELECT * FROM read_parquet({sql_string(glob)}) '
                            'WHERE pickup_hour >= ? AND pickup_hour < ? ORDER BY pickup_hour, src, dst',
                            [cfg.start, cfg.start + cfg.hours * HOUR]).df()
    if frame.empty:
        raise ValueError('No historical trips in the selected simulation window')
    observed_hours = set(frame.pickup_hour.astype(int))
    if any(cfg.start + h * HOUR not in observed_hours for h in range(cfg.hours)):
        raise ValueError('Historical data does not cover every hour of this simulation window')
    rng = np.random.default_rng(cfg.seed)
    requests = []
    for row in frame.itertuples(index=False):
        for quarter in range(4):
            count = int(getattr(row, f'bin{quarter}'))
            whole, fraction = divmod(cfg.demand_scale, 1)
            count = int(whole) * count + int(rng.binomial(count, fraction))
            if len(requests) + count > 50000:
                raise ValueError('This window exceeds 50,000 requests; reduce hours or market share')
            offsets = rng.uniform(0, 900, count)
            for offset in offsets:
                requests.append({'at': float(row.pickup_hour - cfg.start + quarter * 900 + offset),
                                 'source': int(row.src), 'target': int(row.dst),
                                 'duration': max(1., float(row.duration)) * 60,
                                 'fare': max(0., float(row.fare)) * cfg.price_multiplier,
                                 'distance': max(0., float(row.distance))})
    requests.sort(key=lambda r: r['at'])
    for i, request in enumerate(requests):
        request['id'] = i + 1
    return requests


def completed_duration_priors(data: Path, cutoff: int):
    """Only recently completed observations known before the scenario starts."""
    glob = data / 'processed' / 'observations' / '*.parquet'
    with duckdb.connect() as con:
        rows = con.execute(f'SELECT src, dst, sum(duration_sum)/sum(trip_count), sum(trip_count) '
                           f'FROM read_parquet({sql_string(glob)}) '
                           'WHERE available_at <= ? AND pickup_hour >= ? GROUP BY src, dst',
                           [cutoff, cutoff - 28 * HOUR * 24]).fetchall()
    return {(int(s), int(d)): (float(duration), int(count)) for s, d, duration, count in rows}


class FleetSimulation:
    def __init__(self, cfg, zones, requests, predictions, policy, initial_zones, duration_priors=None):
        self.cfg, self.requests, self.policy = cfg, requests, policy
        self.env = simpy.Environment()
        self.end = cfg.hours * HOUR
        self.coords = {int(z.zone_id): (float(z.longitude), float(z.latitude)) for z in zones.itertuples()}
        self.vehicles = [{'id': i+1, 'zone': int(z), 'target': int(z), 'status': 'idle',
                          'started_at': 0., 'busy_until': 0.} for i, z in enumerate(initial_zones)]
        self._road_cache = {}
        self.duration_priors = duration_priors or {}
        self.forecasts = {(p['source_zone'], p['target_zone']): p for p in predictions}
        demand, margin, minutes = {}, {}, {}
        for p in predictions:
            z = p['source_zone']; n = max(0., p['predicted_trips'])
            demand[z] = demand.get(z, 0.) + n
            margin[z] = margin.get(z, 0.) + n * p['predicted_fare_usd'] * cfg.price_multiplier
            minutes[z] = minutes.get(z, 0.) + n * max(1., p['predicted_duration_min'])
        total = max(1., sum(demand.values()))
        self.zone_value = {}
        for z in self.coords:
            n = demand.get(z, 0.)
            taxis = max(1., cfg.fleet_size * n / total)
            pressure = n / (n + taxis * 3) if n else 0.
            self.zone_value[z] = min(2., margin.get(z, 0.) / max(1., minutes.get(z, 0.))) * pressure
        self.pending, self.decisions, self.timeline = [], [], []
        self.revenue = self.fuel = self.miles = self.busy_seconds = self.wait_seconds = 0.
        self.requested = self.accepted = self.completed = self.rejected = 0
        self.reasons = {'no_vehicle': 0, 'unprofitable': 0, 'opportunity_cost': 0, 'end_of_window': 0}
        self.env.process(self.arrivals())
        self.env.process(self.dispatch())

    def road_miles(self, a, b):
        if a == b:
            return 0.
        key = (min(a, b), max(a, b))
        if key in self._road_cache:
            return self._road_cache[key]
        x, y = self.coords[a], self.coords[b]
        lon1, lat1, lon2, lat2 = map(math.radians, (*x, *y))
        hav = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
        result = 1.3 * 2 * 3958.7613 * math.asin(min(1., math.sqrt(hav)))
        self._road_cache[key] = result
        return result

    def arrivals(self):
        for request in self.requests:
            yield self.env.timeout(max(0., request['at'] - self.env.now))
            self.pending.append(request)
            self.requested += 1

    def offer(self, vehicle, request):
        miles = self.road_miles(vehicle['zone'], request['source'])
        pickup_seconds = miles / 12 * HOUR
        if self.env.now - request['at'] + pickup_seconds > self.cfg.max_wait_minutes * 60:
            return None
        p = self.forecasts.get((request['source'], request['target']))
        # Forecast duration is used for selection; actual duration is hidden until service.
        duration = max(1., p['predicted_duration_min'] if p else 20.)
        prior = self.duration_priors.get((request['source'], request['target']))
        if prior:
            # Stabilize small/noisy TGN duration estimates with a pre-cutoff
            # completed-trip prior. Future actual duration never enters scoring.
            weight = .6 * prior[1] / (prior[1] + 20)
            duration = (1-weight)*duration + weight*max(1., prior[0])
        fuel = (miles + request['distance']) / self.cfg.mpg * self.cfg.fuel_price
        contribution = request['fare'] - fuel
        # Optimize earning rate rather than favoring long, high-fare rides
        # that tie up the limited fleet. Zone values express forecast demand
        # in comparable dollars/minute, with a modest destination credit.
        opportunity = .2 * self.zone_value.get(vehicle['zone'], 0.)
        destination = .1 * self.zone_value.get(request['target'], 0.)
        score = contribution / max(1., duration + pickup_seconds/60) - opportunity + destination
        return score, contribution, miles, pickup_seconds, fuel

    def log(self, request, accepted, reason, vehicle=None, offer=None):
        self.decisions.append({'id': request['id'], 'at': round(self.env.now, 2),
                               'requested_at': round(request['at'], 2), 'source': request['source'],
                               'target': request['target'], 'accepted': accepted, 'reason': reason,
                               'vehicle': vehicle['id'] if vehicle else None, 'fare': round(request['fare'], 2),
                               'estimated_contribution': round(offer[1], 2) if offer else None,
                               'score': round(offer[0], 2) if offer else None})

    def serve(self, vehicle, request, offer):
        _, _, deadhead, pickup_seconds, fuel = offer
        service_seconds = pickup_seconds + request['duration']
        vehicle.update(status='busy', target=request['target'], started_at=self.env.now,
                       busy_until=self.env.now + service_seconds)
        self.accepted += 1
        self.wait_seconds += self.env.now - request['at'] + pickup_seconds
        self.log(request, True, 'first_come' if self.policy == 'baseline' else 'positive_forecast_value', vehicle, offer)
        # Charge fuel as a commitment when dispatched; book the fare on completion.
        self.fuel += fuel
        self.miles += deadhead + request['distance']
        self.busy_seconds += max(0., min(service_seconds, self.end-self.env.now))
        yield self.env.timeout(service_seconds)
        self.completed += 1
        self.revenue += request['fare']
        vehicle.update(zone=request['target'], status='idle')

    def dispatch(self):
        while self.env.now <= self.end:
            idle = [v for v in self.vehicles if v['status'] == 'idle']
            matched = set()
            if self.policy == 'baseline':
                for r in self.pending:
                    choices = [(v, self.offer(v, r)) for v in idle]
                    choices = [(v, o) for v, o in choices if o is not None]
                    if choices:
                        v, offer = min(choices, key=lambda item: (item[1][3], item[0]['id']))
                        self.env.process(self.serve(v, r, offer))
                        idle.remove(v); matched.add(r['id'])
            elif idle and self.pending:
                offers = [[self.offer(v, r) for r in self.pending] for v in idle]
                # Zero-valued dummy columns let every vehicle stay idle. Positive
                # matching is globally optimal for this batch's estimated value.
                scores = np.zeros((len(idle), len(self.pending) + len(idle)))
                for i, row in enumerate(offers):
                    for j, offer in enumerate(row):
                        scores[i, j] = offer[0] if offer and offer[1] > 0 else -1e9
                vi, ri = linear_sum_assignment(scores, maximize=True)
                for i, j in zip(vi, ri):
                    if j < len(self.pending) and scores[i, j] > 0:
                        r = self.pending[j]
                        self.env.process(self.serve(idle[i], r, offers[i][j]))
                        matched.add(r['id'])
            remaining = []
            for r in self.pending:
                if r['id'] in matched:
                    continue
                # Retain the economic reason while a feasible taxi still exists;
                # at the exact deadline most remote pickups are infeasible.
                feasible_now = [self.offer(v, r) for v in idle] if self.policy == 'optimized' else []
                feasible_now = [o for o in feasible_now if o is not None]
                if feasible_now:
                    r = {**r, '_decline_reason': 'unprofitable' if max(o[1] for o in feasible_now) <= 0 else 'opportunity_cost'}
                if self.env.now - r['at'] >= self.cfg.max_wait_minutes * 60 or self.env.now >= self.end:
                    candidates = [self.offer(v, r) for v in idle]
                    feasible = [o for o in candidates if o is not None]
                    reason = r.get('_decline_reason') or ('unprofitable' if feasible and max(o[1] for o in feasible) <= 0 else
                              'opportunity_cost' if feasible else 'no_vehicle')
                    if self.env.now >= self.end and self.env.now-r['at'] < self.cfg.max_wait_minutes*60:
                        reason = 'end_of_window'
                    self.rejected += 1; self.reasons[reason] += 1
                    self.log(r, False, reason)
                else:
                    remaining.append(r)
            self.pending = remaining
            yield self.env.timeout(30)

    def snapshot(self):
        now = min(self.env.now, self.end)
        driver_cost = self.cfg.fleet_size * self.cfg.driver_cost_hour * now / HOUR
        metrics = {'requested': self.requested, 'accepted': self.accepted, 'completed': self.completed,
                   'rejected': self.rejected, 'pending': len(self.pending),
                   'in_service': self.accepted-self.completed,
                   'revenue': round(self.revenue, 2), 'fuel_cost': round(self.fuel, 2),
                   'driver_cost': round(driver_cost, 2), 'net_revenue': round(self.revenue-self.fuel-driver_cost, 2),
                   'miles': round(self.miles, 2),
                   'utilization': round(sum(v['status'] == 'busy' for v in self.vehicles) / self.cfg.fleet_size, 4),
                   'mean_wait_minutes': round(self.wait_seconds / max(1, self.accepted) / 60, 2)}
        return {'at': round(now, 2), **metrics, 'vehicles': [v.copy() for v in self.vehicles]}

    def run(self):
        self.timeline.append(self.snapshot())
        for second in range(60, self.end+1, 60):
            # Numeric run(until) excludes boundary events; the tiny epsilon
            # includes dispatch/completions at exactly this sample time.
            self.env.run(until=second + 1e-6)
            self.timeline.append(self.snapshot())
        final = self.timeline[-1]
        return {'policy': self.policy, 'summary': {k: v for k, v in final.items() if k not in ('vehicles', 'at')},
                'rejection_reasons': self.reasons, 'timeline': self.timeline, 'decisions': self.decisions}


def compare_policies(data: Path, cfg: SimulationConfig, predictions: list[dict]):
    zones = read_zones(data)
    requests = historical_requests(data, cfg)
    valid = set(zones.zone_id.astype(int))
    if any(r['source'] not in valid or r['target'] not in valid for r in requests):
        raise ValueError('Request zones do not match processed geometry')
    # Identical, forecast-weighted starting fleet for a fair policy comparison.
    weights = {z: 0. for z in valid}
    for p in predictions:
        if p['source_zone'] in weights:
            weights[p['source_zone']] += max(0., p['predicted_trips'])
    ids = sorted(valid)
    probabilities = np.array([weights[z] for z in ids])
    probabilities = probabilities / probabilities.sum() if probabilities.sum() else np.ones(len(ids))/len(ids)
    initial = np.random.default_rng(cfg.seed+1).choice(ids, cfg.fleet_size, p=probabilities)
    priors = completed_duration_priors(data, cfg.start)
    baseline = FleetSimulation(cfg, zones, requests, predictions, 'baseline', initial, priors).run()
    optimized = FleetSimulation(cfg, zones, requests, predictions, 'optimized', initial, priors).run()
    return {'config': asdict(cfg), 'start_time': format_time(cfg.start), 'request_count': len(requests),
            'source': 'TLC historical OD / quarter-hour aggregates with seeded within-bin arrival times',
            'notice': 'Modeled completed-trip demand; not live or original request logs. Forecast frozen at start. '
                      'Fare booked on completion; full-trip fuel committed at dispatch. In-service trips remain at the horizon. '
                      'Net is fare minus fuel and configured driver cost; other operating costs excluded. '
                      'The optimized policy is a heuristic using TGN forecasts and batch matching; gains are not guaranteed.',
            'baseline': baseline, 'optimized': optimized,
            'net_uplift': round(optimized['summary']['net_revenue']-baseline['summary']['net_revenue'], 2)}
