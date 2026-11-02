import { useEffect, useMemo, useState } from 'react';
import { ArrowRight, Car, DollarSign, Fuel, Pause, Play, RotateCcw, Sparkles, Timer, TrendingUp } from 'lucide-react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { get, number, post, when } from './api';
import { SummaryCard } from './SummaryCard';
import type { Job, SimulationResult, Snapshot } from './simulationTypes';
import type { Zones } from './types';

const money = (value: number) => `$${number(value, 2)}`;
const defaults = { hours: 2, fleet_size: 80, fuel_price: 3.5, mpg: 25, price_multiplier: 1, demand_scale: 1, max_wait_minutes: 8, driver_cost_hour: 0, seed: 42 };
export function Simulator({ active = true }: { active?: boolean }) {
  const [visible, setVisible] = useState(!document.hidden);
  useEffect(() => {
    const update = () => setVisible(!document.hidden);
    document.addEventListener('visibilitychange', update);
    return () => document.removeEventListener('visibilitychange', update);
  }, []);
  const [zones, setZones] = useState<Zones | null>(null);
  const [times, setTimes] = useState<string[]>([]);
  const [asOf, setAsOf] = useState('');
  const [config, setConfig] = useState(defaults);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [jobId, setJobId] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(100);
  const [cursor, setCursor] = useState(0);
  const [policy, setPolicy] = useState<'optimized' | 'baseline'>('optimized');
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let active = true;
    Promise.all([get<Zones>('/zones'), get<{times: string[]}>('/available-times')]).then(([z, t]) => {
      if (active) { setZones(z); setTimes(t.times); setAsOf(t.times[0] || ''); if (!t.times.length) setError('No held-out hours are available. Train a model using the Model & training tab.'); }
    }).catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, [refresh]);
  useEffect(() => {
    if (!jobId) return;
    let active = true;
    let timer: number;
    const poll = async () => {
      try {
        const job = await get<Job<SimulationResult>>(`/jobs/${jobId}`);
        if (!active) return;
        if (job.status === 'complete' && job.result) { setResult(job.result); setCursor(0); setPlaying(true); setBusy(false); setJobId(''); }
        else if (job.status === 'failed') { setError(job.error || 'Simulation failed'); setBusy(false); setJobId(''); }
        else timer = window.setTimeout(poll, 750);
      } catch (e) { if (active) { setError((e as Error).message); setBusy(false); setJobId(''); } }
    };
    void poll();
    return () => { active = false; window.clearTimeout(timer); };
  }, [jobId]);
  useEffect(() => {
    if (!playing || !result || !active || !visible) return;
    let previous = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now(); const delta = (now - previous) / 1000 * speed; previous = now;
      setCursor(current => Math.min(current + delta, result.config.hours * 3600));
    }, 100);
    return () => window.clearInterval(timer);
  }, [playing, speed, result, active, visible]);
  useEffect(() => {
    if (result && cursor >= result.config.hours * 3600) setPlaying(false);
  }, [cursor, result]);
  const run = async () => {
    setBusy(true); setError(''); setPlaying(false);
    try { const job = await post<Job<SimulationResult>>('/simulations', { ...config, as_of: asOf }); setJobId(job.id); }
    catch (e) { setError((e as Error).message); setBusy(false); }
  };
  const at = Math.floor(cursor / 60);
  const optimized = result?.optimized.timeline[Math.min(at, result.optimized.timeline.length-1)];
  const baseline = result?.baseline.timeline[Math.min(at, result.baseline.timeline.length-1)];
  const snapshot = policy === 'optimized' ? optimized : baseline;
  const lookup = useMemo(() => new Map(zones?.features.map(z => [z.properties.zone_id, z.properties]) || []), [zones]);
  const name = (id: number) => lookup.get(id)?.zone || `Zone ${id}`;
  const decisionTime = Math.floor(cursor / 30) * 30;
  const decisions = useMemo(() => {
    const items = result?.[policy].decisions || [];
    // Decision times are ordered: locate the current window without scanning
    // tens of thousands of historical requests on every animation frame.
    let low = 0, high = items.length;
    while (low < high) { const mid = (low+high) >>> 1; if (items[mid].at <= decisionTime) low = mid+1; else high = mid; }
    return items.slice(Math.max(0,low-10),low).reverse();
  }, [result, policy, decisionTime]);
  const startEpoch = result ? new Date(result.start_time).getTime() : 0;
  const chart = useMemo(() => result?.optimized.timeline.slice(0, at+1).map((s, i) => ({ time: when(new Date(startEpoch+s.at*1000).toISOString(), { hour: 'numeric', minute: '2-digit' }), optimized: s.net_revenue, baseline: result.baseline.timeline[i]?.net_revenue })) || [], [result, at, startEpoch]);
  const timeOptions = useMemo(() => times.map(t => <option key={t} value={t}>{when(t, {month:'short', day:'numeric', year:'numeric', hour:'numeric', timeZoneName:'short'})}</option>), [times]);
  const field = (key: keyof typeof defaults, label: string, min: number, max: number, step = 1) => <label className="sim-field">{label}<input type="number" aria-label={label} required min={min} max={max} step={step} value={config[key]} onChange={e => setConfig(c => ({ ...c, [key]: e.target.valueAsNumber }))}/></label>;
  return <main className="operations-shell">
    <div className="heading-row"><div><div className="eyebrow"><span className="eyebrow-dash"/> FLEET OPERATIONS / SIMULATION LAB</div><h1>Make every mile count<span>.</span></h1><p>Replay NYC demand. Let the forecast guide the fleet. Measure the return.</p></div><div className="sim-engine"><Sparkles size={17}/><div><strong>TGN + optimal batch matching</strong><small>SimPy discrete-event engine · seeded scenarios</small></div></div></div>
    <section className="scenario-card"><div className="scenario-title"><div className="card-kicker">CONFIGURE A SCENARIO</div><p>Both policies get the same requests and starting fleet.</p><button className="text-button" onClick={() => { setError(''); setRefresh(x => x+1); }} disabled={busy}>Refresh data</button></div>
      <form onSubmit={e => { e.preventDefault(); void run(); }}><div className="scenario-fields"><label className="sim-field time-field">Historical start<select aria-label="Historical start" value={asOf} required onChange={e => setAsOf(e.target.value)}>{timeOptions}</select></label>
        {field('hours', 'Hours to simulate', 1, 6)}{field('fleet_size', 'Fleet size', 1, 300)}{field('fuel_price', 'Fuel / gallon ($)', 0, 20, .1)}{field('mpg', 'Fuel economy (mpg)', 1, 100)}{field('price_multiplier', 'Fare multiplier', .1, 5, .1)}
      </div><details className="advanced-options"><summary>Demand, service & operating assumptions</summary><div className="scenario-fields">{field('demand_scale', 'Market share / demand scale', .01, 2, .01)}{field('max_wait_minutes', 'Max pickup wait (min)', 1, 30)}{field('driver_cost_hour', 'Driver cost / taxi-hour ($)', 0, 100, 1)}{field('seed', 'Random seed', 0, 2147483647)}</div><p>1.0 replays all recorded completed-trip demand. Fuel and fare are editable scenario assumptions. Other costs are excluded.</p></details><div className="scenario-footer"><span><span className="live-dot"/> HISTORICAL REPLAY · FORECAST FROZEN AT START</span><button className="primary-button" disabled={busy || !asOf}>{busy ? <><span className="spinner"/>Running both policies…</> : <><Play size={15}/>Run comparison</>}</button></div></form>
    </section>
    {error && <div role="alert" className="error-banner">{error}</div>}
    <div className="stat-grid sim-stats"><SummaryCard label="NET REVENUE" value={snapshot ? money(snapshot.net_revenue) : '—'} detail="Completed fares − fuel − driver cost" icon={<DollarSign size={17}/>}/><SummaryCard label="VS. FIRST COME" value={optimized && baseline ? `${optimized.net_revenue-baseline.net_revenue >= 0 ? '+' : '−'}${money(Math.abs(optimized.net_revenue-baseline.net_revenue))}` : '—'} detail="Forecast policy minus baseline · same demand" icon={<TrendingUp size={17}/>}/><SummaryCard label="COMPLETED / ACCEPTED" value={snapshot ? `${snapshot.completed} / ${snapshot.accepted}` : '—'} detail={snapshot ? `${snapshot.in_service} in service · ${snapshot.pending} waiting · ${snapshot.rejected} declined` : 'Constrained by fleet size and pickup wait'} icon={<Car size={17}/>}/><SummaryCard label="FUEL COST" value={snapshot ? money(snapshot.fuel_cost) : '—'} detail={snapshot ? `${number(snapshot.miles,1)} miles committed · ${number(snapshot.utilization*100)}% busy` : 'Includes empty pickup miles'} icon={<Fuel size={17}/>}/></div>
    <div className="simulation-grid"><section className="map-card"><div className="section-head"><div><div className="card-kicker">THE FLEET <span>01</span></div><h2>New York, in motion</h2><p>Zone-to-zone vehicle movement · approximate centroid travel</p></div><div className="segmented"><button className={policy === 'optimized' ? 'selected' : ''} onClick={() => setPolicy('optimized')}>Forecast policy</button><button className={policy === 'baseline' ? 'selected' : ''} onClick={() => setPolicy('baseline')}>First come</button></div></div>
      <div className="fleet-map">{zones && <FleetMap zones={zones} snapshot={snapshot} cursor={cursor}/>}<div className="fleet-map-label"><span className="live-dot"/> {result ? 'HISTORICAL SIMULATION' : 'READY TO SIMULATE'}<small>{result ? `${number(result.request_count)} modeled requests · ${result.config.fleet_size} taxis` : 'Configure a scenario and run a comparison'}</small></div><div className="fleet-legend"><span><i/>Available taxi</span><span><i/>Serving a trip</span><span>● TLC taxi zone</span></div>{!result && <div className="simulation-empty"><Car size={30}/><h3>Your fleet starts here.</h3><p>Run a scenario to watch dispatch decisions unfold.</p></div>}</div>
      <div className="simulation-playback"><button className="play-button" aria-label={playing ? 'Pause simulation' : 'Play simulation'} disabled={!result} onClick={() => { if (result && cursor >= result.config.hours*3600) setCursor(0); setPlaying(p => !p); }}>{playing ? <Pause size={16}/> : <Play size={16}/>}</button><button className="icon-button" aria-label="Restart playback" disabled={!result} onClick={() => { setCursor(0); setPlaying(false); }}><RotateCcw size={16}/></button><div className="playback-track"><div><strong>{result ? when(new Date(startEpoch+cursor*1000).toISOString(), { hour:'numeric', minute:'2-digit', second:'2-digit', timeZoneName:'short' }) : 'SIMULATION CLOCK'}</strong><span>{result ? `${number(cursor/60,1)} / ${result.config.hours*60} min` : 'Playback up to 100×'}</span></div><input aria-label="Simulation time" type="range" min={0} max={result ? result.config.hours*3600 : 1} step={1} value={cursor} disabled={!result} onChange={e => { setPlaying(false); setCursor(Number(e.target.value)); }}/></div><label className="speed-select"><Timer size={15}/><select aria-label="Playback speed" value={speed} onChange={e => setSpeed(Number(e.target.value))}>{[1,10,25,50,100].map(n => <option key={n} value={n}>{n}×</option>)}</select></label></div>
    </section><aside className="decision-card"><div className="section-head"><div><div className="card-kicker">DISPATCH INTELLIGENCE <span>02</span></div><h2>Take it or wait?</h2><p>Latest decisions at the simulation clock</p></div><Sparkles size={19}/></div><div className="decision-list">{decisions.length ? decisions.map(d => <div className="decision" key={d.id}><div><span className={`decision-badge ${d.accepted ? 'take' : 'skip'}`}>{d.accepted ? `TAKE · TAXI ${d.vehicle}` : 'DECLINE'}</span><strong>{money(d.fare)}</strong></div><p title={`${name(d.source)} → ${name(d.target)}`}>{name(d.source)} <ArrowRight size={12}/> {name(d.target)}</p><small>{({ first_come: 'Nearest available taxi', positive_forecast_value: 'Positive value after fuel & forecast opportunity cost', no_vehicle: 'No taxi can meet the pickup deadline', unprofitable: 'Fare does not cover fuel', opportunity_cost: 'Keep capacity for higher-value demand', end_of_window: 'Simulation window closed before assignment' } as Record<string, string>)[d.reason]}{d.score != null && policy === 'optimized' ? ` · score ${money(d.score)}/min` : ''}</small></div>) : <div className="decision-empty"><Sparkles size={25}/><p>{result ? 'Decisions will appear as the clock advances.' : 'The dispatcher weighs fares, fuel, pickup time, and predicted demand.'}</p></div>}</div><div className="decision-note">Every idle taxi can wait. The optimizer assigns the best positive-value batch; the baseline accepts requests in arrival order.</div></aside></div>
    <div className="simulation-analysis"><section className="chart-card"><div className="chart-title"><div><div className="card-kicker">POLICY COMPARISON <span>03</span></div><h3>Revenue after operating costs</h3></div><span className="chart-chip">SAME REQUEST STREAM</span></div><div className="chart-body">{result ? <ResponsiveContainer width="100%" height="100%"><LineChart data={chart} margin={{left:5,right:15,top:10,bottom:0}}><CartesianGrid vertical={false} stroke="#edf0ed"/><XAxis dataKey="time" minTickGap={65} fontSize={10}/><YAxis fontSize={10} tickFormatter={n => `$${n}`}/><Tooltip formatter={v => money(Number(v))}/><Legend/><Line dataKey="optimized" name="Forecast policy" stroke="#218273" dot={false} strokeWidth={2.5} isAnimationActive={false}/><Line dataKey="baseline" name="First come" stroke="#c99b5a" dot={false} strokeWidth={2} isAnimationActive={false}/></LineChart></ResponsiveContainer> : <div className="chart-empty">Compare the two policies as the simulation plays.</div>}</div></section><section className="chart-card"><div className="card-kicker">THE SCORECARD <span>04</span></div><h3 className="scorecard-title">Same city. Different decisions.</h3><table className="scorecard"><thead><tr><th>At current clock</th><th>Forecast policy</th><th>First come</th></tr></thead><tbody>{[['Gross fares', 'revenue'], ['Fuel committed', 'fuel_cost'], ['Driver cost', 'driver_cost'], ['Net revenue', 'net_revenue'], ['Completed trips', 'completed'], ['Declined requests', 'rejected'], ['Avg pickup wait (min)', 'mean_wait_minutes']].map(([label,key]) => <tr key={key}><td>{label}</td><td>{optimized ? (['revenue','fuel_cost','driver_cost','net_revenue'].includes(key) ? money(Number(optimized[key as keyof Snapshot])) : number(Number(optimized[key as keyof Snapshot]), key==='mean_wait_minutes'?1:0)) : '—'}</td><td>{baseline ? (['revenue','fuel_cost','driver_cost','net_revenue'].includes(key) ? money(Number(baseline[key as keyof Snapshot])) : number(Number(baseline[key as keyof Snapshot]), key==='mean_wait_minutes'?1:0)) : '—'}</td></tr>)}</tbody></table></section></div>
    <p className="simulation-disclosure">{result?.notice || 'Requests are modeled from TLC completed-trip counts by route and quarter hour. Arrival times are seeded; duration, distance and fare use historical means. This is a policy experiment, and forecast-guided decisions may earn less than the baseline.'}</p>
  </main>;
}

function FleetMap({ zones, snapshot, cursor }: { zones: Zones; snapshot?: Snapshot; cursor: number }) {
  const project = (lon: number, lat: number) => [(lon+74.28)/.62*800, (40.95-lat)/.47*580];
  const paths = useMemo(() => zones.features.map(f => {
    const polygons = f.geometry.type === 'MultiPolygon' ? f.geometry.coordinates as number[][][][] : [f.geometry.coordinates as number[][][]];
    return { id: f.properties.zone_id, name: f.properties.zone, d: polygons.map(poly => poly.map(ring => ring.map(([lon,lat], i) => `${i ? 'L' : 'M'}${project(lon,lat).join(',')}`).join(' ')+'Z').join(' ')).join(' ') };
  }), [zones]);
  const lookup = useMemo(() => new Map(zones.features.map(f => [f.properties.zone_id, f.properties])), [zones]);
  const background = useMemo(() => <><defs><pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="#dbe9e6" strokeWidth=".6"/></pattern></defs><rect width="800" height="580" fill="#e5efed"/><rect width="800" height="580" fill="url(#grid)"/>{paths.map(p => <path key={p.id} d={p.d} fill="#f5f6ed" stroke="#c0d1c3" strokeWidth=".65" fillRule="evenodd"><title>{p.name}</title></path>)}{[[-74.13,40.61,'STATEN ISLAND'],[-73.96,40.64,'BROOKLYN'],[-73.81,40.74,'QUEENS'],[-74.01,40.76,'MANHATTAN'],[-73.88,40.88,'THE BRONX']].map(([lon,lat,name]) => {const [x,y] = project(Number(lon),Number(lat)); return <text key={name} x={x} y={y} textAnchor="middle" className="borough-label">{name}</text>;})}</>, [paths]);
  return <svg viewBox="0 0 800 580" aria-label="NYC taxi fleet map" role="img">{background}{snapshot?.vehicles.map(v => {
    const origin = lookup.get(v.zone), destination = lookup.get(v.target); if (!origin || !destination) return null;
    const progress = v.status === 'busy' ? Math.min(1, Math.max(0, (cursor-v.started_at)/Math.max(1,v.busy_until-v.started_at))) : 0;
    const [x,y] = project(origin.longitude+(destination.longitude-origin.longitude)*progress, origin.latitude+(destination.latitude-origin.latitude)*progress);
    return <g key={v.id}>{v.status === 'busy' && <path d={`M${project(origin.longitude,origin.latitude).join(',')}L${project(destination.longitude,destination.latitude).join(',')}`} stroke="#e4b052" strokeWidth=".7" opacity=".3"/>}<circle cx={x} cy={y} r="3.4" fill={v.status === 'busy' ? '#e4a13f' : '#247d6c'} stroke="white" strokeWidth="1.2"><title>Taxi {v.id} · {v.status} · {origin.zone} → {destination.zone}</title></circle></g>;
  })}<text x="770" y="35" className="map-north">N ↑</text></svg>;
}
