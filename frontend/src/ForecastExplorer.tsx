import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, ArrowDownLeft, ArrowRight, ArrowUpRight, BarChart3, Clock3, Compass, Layers3, Map as MapIcon, Pause, Play, Route, Search, Sparkles, X } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { get, number, when } from './api';
import { MapView } from './MapView';
import { SummaryCard } from './SummaryCard';
import type { ActualRoute, Actuals, Flow, Forecast, Metrics, Prediction, Timeline, ZoneForecast, Zones } from './types';

type Times = { times: string[]; last_observed_pickup_hour: string; model_selection_end: string; quick_model: boolean };
type Mode = 'predicted' | 'actual' | 'error';
type Heat = 'none' | 'pickups' | 'dropoffs';

const formatHour = (t: string) => when(t, { hour: 'numeric', hour12: true });

export default function App() {
  const [zones, setZones] = useState<Zones | null>(null);
  const [times, setTimes] = useState<Times | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [index, setIndex] = useState(0);
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [actuals, setActuals] = useState<Actuals | null>(null);
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [zoneForecast, setZoneForecast] = useState<ZoneForecast | null>(null);
  const [selectedZone, setSelectedZone] = useState<number | null>(null);
  const [selectedFlow, setSelectedFlow] = useState<Flow | null>(null);
  const [mode, setMode] = useState<Mode>('predicted');
  const [heat, setHeat] = useState<Heat>('none');
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [zoneError, setZoneError] = useState('');

  useEffect(() => {
    Promise.all([get<Zones>('/zones'), get<Times>('/available-times'), get<Metrics>('/metrics')])
      .then(([z,t,m]) => { setZones(z); setTimes(t); setMetrics(m); if (!t.times.length) setError('No hours are available after model selection. Train with a longer dataset.'); })
      .catch(e => setError(e.message));
  }, []);
  const asOf = times?.times[index];
  useEffect(() => {
    if (!asOf) return;
    let active = true;
    setLoading(true); setError(''); setForecast(null); setActuals(null); setTimeline(null); setSelectedFlow(null);
    Promise.all([get<Forecast>('/forecast', { as_of: asOf, top_k: 150 }), get<Actuals>('/actuals', { as_of: asOf }), get<Timeline>('/trends', { as_of: asOf, hours: 12 })])
      .then(([f,a,t]) => { if (active) { setForecast(f); setActuals(a); setTimeline(t); } })
      .catch(e => { if (active) setError(e.message); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [asOf]);
  useEffect(() => {
    if (!asOf || selectedZone == null) { setZoneForecast(null); setZoneError(''); return; }
    let active = true;
    setZoneForecast(null); setZoneError('');
    get<ZoneForecast>(`/zone/${selectedZone}/forecast`, { as_of: asOf }).then(z => { if (active) setZoneForecast(z); }).catch(e => { if (active) setZoneError(e.message); });
    return () => { active = false; };
  }, [selectedZone, asOf]);
  useEffect(() => {
    if (!asOf || !selectedFlow || selectedFlow.predicted) return;
    let active = true;
    get<{prediction: Prediction; actual: ActualRoute | null}>('/route', {as_of: asOf, source_zone: selectedFlow.source, target_zone: selectedFlow.target})
      .then(result => { if (active) setSelectedFlow(current => current && current.source === selectedFlow.source && current.target === selectedFlow.target ? {...current, predicted: result.prediction, actual: result.actual || undefined} : current); })
      .catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, [selectedFlow, asOf]);
  useEffect(() => {
    if (!playing || loading || !times?.times.length) return;
    const timer = window.setTimeout(() => setIndex(i => { if (i >= times.times.length - 1) { setPlaying(false); return i; } return i + 1; }), 2200);
    return () => window.clearTimeout(timer);
  }, [playing, loading, index, times]);

  const lookup = useMemo(() => new Map(zones?.features.map(f => [f.properties.zone_id, f.properties]) || []), [zones]);
  const name = (id: number) => lookup.get(id)?.zone || `Zone ${id}`;
  const actualByRoute = useMemo(() => new Map(actuals?.routes.map(r => [`${r.source_zone}-${r.target_zone}`, r]) || []), [actuals]);
  const flows = useMemo<Flow[]>(() => {
    const records = mode === 'actual' ? (actuals?.routes || []).slice(0,150).map(a => ({ source: a.source_zone, target: a.target_zone, value: a.trip_count, actual: a, predicted: forecast?.predictions.find(p => p.source_zone === a.source_zone && p.target_zone === a.target_zone) }))
      : (forecast?.predictions || []).map(p => { const a = actualByRoute.get(`${p.source_zone}-${p.target_zone}`); const error = actuals?.available ? p.predicted_trips - (a?.trip_count || 0) : undefined; return { source: p.source_zone, target: p.target_zone, value: mode === 'error' ? Math.abs(error || 0) : p.predicted_trips, actual: a, predicted: p, error: mode === 'error' ? error : undefined }; });
    return records.flatMap(r => { const from = lookup.get(r.source), to = lookup.get(r.target); return from && to ? [{ ...r, from: [from.longitude, from.latitude] as [number,number], to: [to.longitude, to.latitude] as [number,number] }] : []; });
  }, [mode, actuals, forecast, actualByRoute, lookup]);
  const heatValues = heat === 'none' ? null : heat === 'pickups' ? forecast?.summary.zone_pickups || null : forecast?.summary.zone_dropoffs || null;
  const selectZone = useCallback((id: number) => { setSelectedFlow(null); setSelectedZone(id); }, []);
  const selectFlow = useCallback((flow: Flow) => { setSelectedZone(null); setSelectedFlow(flow); }, []);
  const selectPrediction = (p: Prediction) => {
    const from = lookup.get(p.source_zone), to = lookup.get(p.target_zone);
    if (from && to) selectFlow({source: p.source_zone, target: p.target_zone, from: [from.longitude, from.latitude], to: [to.longitude, to.latitude], value: p.predicted_trips, predicted: p, actual: actualByRoute.get(`${p.source_zone}-${p.target_zone}`)});
  };
  const totalDiff = forecast && actuals?.available ? forecast.summary.predicted_total - actuals.total_trips : null;
  const chartHours = (timeline?.hours || []).map(h => ({ name: formatHour(h.time), actual: h.actual_trips, predicted: h.time === asOf ? forecast?.summary.predicted_total : undefined }));
  const topRoutes = (forecast?.predictions || []).slice(0,10).map(p => ({ name: `${name(p.source_zone)} → ${name(p.target_zone)}`, trips: p.predicted_trips }));
  const metricData = Object.entries(metrics?.models || {}).map(([key, value]) => ({ name: key === 'tgn' ? 'TGN' : key === 'recent' ? 'Recent' : 'Seasonal', mrr: value.mrr, mae: value.volume_mae }));
  const intervals = [[0,10], [10,20], [20,30], [30,45], [45,Infinity]];
  const profileData = intervals.map(([low,high]) => ({ name: high === Infinity ? '45+' : `${low}–${high}`,
    duration: (forecast?.predictions || []).filter(p => p.predicted_duration_min >= low && p.predicted_duration_min < high).length,
    fare: (forecast?.predictions || []).filter(p => p.predicted_fare_usd >= low && p.predicted_fare_usd < high).length }));
  const outgoingTotal = zoneForecast?.outgoing.reduce((sum,p) => sum+p.predicted_trips,0) || 0;
  const zoneDeparture = [0,1,2,3].map(i => outgoingTotal ? zoneForecast!.outgoing.reduce((sum,p) => sum+p.predicted_trips*p.departure_bin_probabilities[i],0)/outgoingTotal : 0);
  const activeZone = selectedZone == null ? null : lookup.get(selectedZone);

  return <div className="app-shell">
    <aside className="rail"><div className="brand-mark"><span>n</span><span>y</span><span>c</span></div><div className="rail-line"/><div className="rail-item active" title="Forecast"><MapIcon size={19}/></div><a className="rail-item" title="Model evaluation" href="#performance"><BarChart3 size={19}/></a><a className="rail-item" title="Trends" href="#trends"><Activity size={19}/></a><div className="rail-bottom"><span className="rail-dot"/>TLC<br/>DATA</div></aside>
    <div className="workspace">
      <header className="topbar"><div className="wordmark">MOTION<span> / </span>NYC</div><div className="topbar-center"><span className="live-dot"/> HISTORICAL FORECAST EXPLORER</div><div className="topbar-right"><Clock3 size={14}/> EASTERN TIME <span className="divider"/> <span className="top-pill">RESEARCH DEMO</span></div></header>
      <main className="main-content">
        <div className="heading-row"><div><div className="eyebrow"><span className="eyebrow-dash"/> MOBILITY INTELLIGENCE <span className="eyebrow-separator">/</span> NYC TAXI</div><h1>Where the city moves<span>.</span></h1><p>Explore predicted yellow taxi flows, one hour at a time.</p></div><div className="heading-aside"><span className="heading-aside-label">FORECAST WINDOW</span><strong>Next 60 minutes</strong><small>Powered by a temporal graph network</small></div></div>
        <div className="stat-grid">
          <SummaryCard label="PREDICTED TRIPS" value={number(forecast?.summary.predicted_total)} detail="Across all zone pairs · next hour" icon={<Route size={16}/>}/>
          <SummaryCard label="OBSERVED TRIPS" value={actuals?.available ? number(actuals.total_trips) : '—'} detail={actuals?.available ? 'TLC records · same hour' : 'Ground truth unavailable'} icon={<Activity size={16}/>}/>
          <SummaryCard label="FORECAST DIFFERENCE" value={totalDiff == null ? '—' : `${totalDiff > 0 ? '+' : ''}${number(totalDiff)}`} detail="Predicted minus observed trips" icon={<BarChart3 size={16}/>}/>
          <SummaryCard label="MAPPED CONNECTIONS" value={number(flows.length)} detail={mode === 'actual' ? 'Highest observed flows' : 'Highest predicted flows'} icon={<Compass size={16}/>}/>
        </div>
        <section className="explorer-grid"><div className="map-card"><div className="section-head"><div><div className="card-kicker">THE NETWORK <span>01</span></div><h2>Citywide flow map</h2><p>Taxi-zone connections for the selected hour</p></div><div className="map-badge"><span/> {loading ? 'LOADING' : 'INTERACTIVE MAP'}</div></div>
          <div className="map-controls"><div className="segmented" aria-label="Flow visualization">{(['predicted','actual','error'] as Mode[]).map(m => <button key={m} className={mode===m ? 'selected' : ''} onClick={() => setMode(m)}>{m === 'error' ? 'Difference' : m[0].toUpperCase()+m.slice(1)}</button>)}</div><label className="heat-control"><Layers3 size={15}/><select value={heat} onChange={e => setHeat(e.target.value as Heat)} aria-label="Zone heatmap"><option value="none">No heatmap</option><option value="pickups">Pickup demand</option><option value="dropoffs">Dropoff demand</option></select></label></div>
          <div className="map-area">{zones && <MapView zones={zones} flows={flows} selectedZone={selectedZone} heat={heatValues} onZone={selectZone} onFlow={selectFlow}/>}<div className="map-label"><span className="map-label-dot"/>{mode === 'predicted' ? 'PREDICTED TRAFFIC' : mode === 'actual' ? 'OBSERVED TRAFFIC' : 'FORECAST DIFFERENCE'}<small>TOP {flows.length} CONNECTIONS</small></div><div className="map-legend"><span>{mode === 'error' ? 'UNDER' : 'LOWER'}</span><i/><span>{mode === 'error' ? 'OVER' : 'HIGHER'}</span><small>FLOW VOLUME</small></div>{loading && <div className="map-loading"><span className="spinner"/> Reconstructing graph history…</div>}{!loading && mode === 'error' && !actuals?.available && <div className="map-loading">No observed trips are available for comparison.</div>}</div>
          <div className="timeline"><button className="play-button" onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause replay' : 'Play replay'} disabled={!times?.times.length}>{playing ? <Pause size={17} fill="currentColor"/> : <Play size={17} fill="currentColor"/>}</button><div className="timeline-main"><div className="timeline-labels"><span>HOURLY REPLAY</span><strong>{asOf ? when(asOf, { weekday:'short', month:'short', day:'numeric', year:'numeric', hour:'numeric', timeZoneName:'short' }) : '—'}</strong></div><input type="range" min="0" max={Math.max(0,(times?.times.length || 1)-1)} value={index} onChange={e => { setPlaying(false); setIndex(Number(e.target.value)); }} aria-label="Select forecast hour"/><div className="range-ends"><span>{times?.times[0] ? when(times.times[0], {month:'short',day:'numeric'}) : '—'}</span><span>{times?.times.at(-1) ? when(times.times.at(-1)!, {month:'short',day:'numeric'}) : '—'}</span></div></div><select className="time-select" value={index} onChange={e => { setPlaying(false); setIndex(Number(e.target.value)); }} aria-label="Choose historical date and hour">{times?.times.map((t,i) => <option key={t} value={i}>{when(t, { month:'short', day:'numeric', hour:'numeric', timeZoneName:'short' })}</option>)}</select></div>
        </div>
        <aside className="inspector"><div className="inspector-head"><div><div className="card-kicker">IN FOCUS <span>02</span></div><h2>{selectedFlow ? 'Route details' : activeZone ? 'Zone details' : 'Explore a connection'}</h2></div>{(selectedFlow || activeZone) && <button className="icon-button" aria-label="Clear selection" onClick={() => { setSelectedFlow(null); setSelectedZone(null); }}><X size={17}/></button>}</div>
          {selectedFlow ? <RouteDetails flow={selectedFlow} name={name} actualAvailable={!!actuals?.available}/> : activeZone ? <><div className="zone-title"><div className="zone-glyph"><MapIcon size={20}/></div><div><small>{activeZone.borough.toUpperCase()} · ZONE {activeZone.zone_id}</small><h3>{activeZone.zone}</h3></div></div>{zoneError && <div className="inline-error">{zoneError}</div>}{zoneForecast ? <><div className="zone-numbers"><div><span><ArrowUpRight size={14}/> PREDICTED OUT</span><strong>{number(zoneForecast.predicted_pickups)}</strong></div><div><span><ArrowDownLeft size={14}/> PREDICTED IN</span><strong>{number(zoneForecast.predicted_dropoffs)}</strong></div></div><div className="panel-label">TOP DESTINATIONS <span>NEXT HOUR</span></div><div className="destination-list">{zoneForecast.outgoing.slice(0,5).map((p,i) => <button key={p.target_zone} onClick={() => selectPrediction(p)}><span className="list-index">0{i+1}</span><span className="destination-name">{name(p.target_zone)}</span><strong>{number(p.predicted_trips,1)}</strong><ArrowRight size={13}/></button>)}</div><div className="panel-label">DEPARTURES <span>TOP 10 ROUTES</span></div><div className="zone-departures">{zoneDeparture.map((v,i) => <div key={i}><span>:{String(i*15).padStart(2,'0')}</span><strong>{number(v*100)}%</strong></div>)}</div><div className="panel-label">RECENT PICKUPS <span>OBSERVED</span></div><div className="mini-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={zoneForecast.recent_actuals}><Line type="monotone" dataKey="actual_trips" stroke="#248878" strokeWidth={2} dot={false}/><Tooltip labelFormatter={x => when(String(x), {month:'short',day:'numeric',hour:'numeric'})}/></LineChart></ResponsiveContainer></div><p className="panel-foot">The zone trend shows actual pickups in the preceding 24 hours.</p></> : !zoneError && <p className="panel-foot">Loading zone details…</p>}</> : <div className="empty-inspector"><div className="empty-icon"><Search size={24}/><span/></div><h3>Follow the flow</h3><p>Select a zone or a connection on the map to see its forecast, trip details, and recent history.</p><div className="hint"><span>01</span> Click a shaded taxi zone</div><div className="hint"><span>02</span> Select a flow line</div><div className="inspector-note"><Sparkles size={16}/><p>Each line is a directed route. Brighter, thicker lines represent more trips.</p></div></div>}
          <div className="inspector-bottom"><span className="live-dot"/> TLC YELLOW TAXI RECORDS <span>↗</span></div></aside></section>
        {error && <div className="error-banner"><strong>Unable to load the forecast.</strong> {error} <span>Follow the setup steps in the project README.</span></div>}
        <div className="analytics-heading" id="trends"><div><div className="eyebrow"><span className="eyebrow-dash"/> THE BIGGER PICTURE</div><h2>Patterns behind the movement<span>.</span></h2></div><p>Forecasts and observations from the selected hour</p></div>
        <div className="analytics-grid"><section className="chart-card wide"><div className="chart-title"><div><div className="card-kicker">VOLUME OVER TIME <span>03</span></div><h3>Citywide trip activity</h3></div><span className="chart-chip">LAST 12 HOURS</span></div><p>Observed pickups and the selected next-hour forecast</p><div className="chart-body"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartHours} margin={{top:10,right:10,left:-20,bottom:0}}><CartesianGrid vertical={false} stroke="#edf0ed"/><XAxis dataKey="name" tickLine={false} axisLine={false} fontSize={10} tick={{fill:'#89938e'}}/><YAxis tickLine={false} axisLine={false} fontSize={10} tick={{fill:'#89938e'}}/><Tooltip formatter={(value) => number(Number(value),0)}/><Legend iconType="circle"/><Bar isAnimationActive={false} dataKey="actual" name="Observed trips" fill="#b7d1c9" radius={[3,3,0,0]}/><Bar isAnimationActive={false} dataKey="predicted" name="Predicted trips" fill="#1e8274" radius={[3,3,0,0]}/></BarChart></ResponsiveContainer></div></section>
          <section className="chart-card"><div className="chart-title"><div><div className="card-kicker">ROUTE LEADERBOARD <span>04</span></div><h3>Top predicted routes</h3></div><ArrowUpRight size={18} color="#83938b"/></div><p>Expected trips by connection</p><div className="route-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={topRoutes} layout="vertical" margin={{top:0,right:20,left:12,bottom:0}}><XAxis type="number" hide/><YAxis type="category" dataKey="name" width={138} tickLine={false} axisLine={false} fontSize={9} tickFormatter={(v: string) => v.length > 26 ? `${v.slice(0,25)}…` : v} tick={{fill:'#697772'}}/><Tooltip formatter={(value) => number(Number(value),1)}/><Bar isAnimationActive={false} dataKey="trips" fill="#2c8c7d" radius={[0,3,3,0]}>{topRoutes.map((_,i) => <Cell key={i} fill={i===0?'#176c60':'#9acabe'}/>)}</Bar></BarChart></ResponsiveContainer></div></section>
          <section className="chart-card" id="performance"><div className="chart-title"><div><div className="card-kicker">MODEL BENCHMARK <span>05</span></div><h3>Link ranking</h3></div><span className="chart-chip">HELD OUT</span></div><p>Mean reciprocal rank · higher is better</p><div className="chart-body small">{metricData.length ? <ResponsiveContainer width="100%" height="100%"><BarChart data={metricData} margin={{top:10,right:10,left:-25,bottom:0}}><CartesianGrid vertical={false} stroke="#edf0ed"/><XAxis dataKey="name" tickLine={false} axisLine={false} fontSize={10}/><YAxis domain={[0,1]} tickLine={false} axisLine={false} fontSize={10}/><Tooltip/><Bar isAnimationActive={false} dataKey="mrr" name="MRR" fill="#318d7d" radius={[3,3,0,0]}/></BarChart></ResponsiveContainer> : <div className="chart-empty">Run evaluation to see model comparisons.</div>}</div><small className="chart-note">{metrics?.population ? `${metrics.population.hours} test hours · ${metrics.population.negatives_per_positive} sampled negatives per positive` : 'No evaluation results yet'}</small>{metricData.length > 0 && <div className="metric-row"><span>TRIP VOLUME MAE</span>{metricData.map(m => <span key={m.name}>{m.name}: <b>{number(m.mae,2)}</b></span>)}</div>}</section>
          <section className="chart-card"><div className="chart-title"><div><div className="card-kicker">TRIP CHARACTERISTICS <span>06</span></div><h3>Route estimate distribution</h3></div><span className="chart-chip">TOP 150</span></div><p>Number of routes by duration (minutes) and metered fare ($)</p><div className="chart-body small"><ResponsiveContainer width="100%" height="100%"><BarChart data={profileData} margin={{top:10,right:10,left:-25,bottom:0}}><CartesianGrid vertical={false} stroke="#edf0ed"/><XAxis dataKey="name" tickLine={false} axisLine={false} fontSize={10}/><YAxis allowDecimals={false} tickLine={false} axisLine={false} fontSize={10}/><Tooltip/><Legend iconType="circle"/><Bar isAnimationActive={false} dataKey="duration" name="Duration (min)" fill="#248878"/><Bar isAnimationActive={false} dataKey="fare" name="Fare ($)" fill="#e4a55d"/></BarChart></ResponsiveContainer></div></section>
        </div><footer><span>MOTION / NYC</span><p>Built with NYC TLC trip records. Monthly data is delayed; this is not live traffic.</p><span>NEW YORK, NY ↗</span></footer>
      </main></div>
  </div>;
}

function RouteDetails({ flow, name, actualAvailable }: { flow: Flow; name: (id:number) => string; actualAvailable: boolean }) {
  const p = flow.predicted, a = flow.actual;
  return <><div className="route-heading"><span>ORIGIN <i/></span><h3>{name(flow.source)}</h3><div className="route-line"><span/> <ArrowRight size={14}/></div><span>DESTINATION <i/></span><h3>{name(flow.target)}</h3></div><div className="route-stats"><div><span>PREDICTED TRIPS</span><strong>{number(p?.predicted_trips,1)}</strong></div><div><span>LINK PROBABILITY</span><strong>{p ? `${number(p.link_probability*100,0)}%` : '—'}</strong></div><div><span>AVG. DURATION</span><strong>{p ? `${number(p.predicted_duration_min,1)} min` : '—'}</strong></div><div><span>METERED FARE</span><strong>{p ? `$${number(p.predicted_fare_usd,2)}` : '—'}</strong></div></div><div className="panel-label">DEPARTURE DISTRIBUTION <span>NEXT HOUR</span></div><div className="departure-bars">{(p?.departure_bin_probabilities || []).map((v,i) => <div key={i}><span>{number(v*100)}%</span><i style={{height:`${Math.max(4,v*120)}px`}}/><small>:{String(i*15).padStart(2,'0')}</small></div>)}</div><div className="comparison"><div>OBSERVED TRIPS <strong>{actualAvailable ? number(a?.trip_count || 0) : '—'}</strong></div><div>DIFFERENCE <strong>{actualAvailable && p ? `${p.predicted_trips-(a?.trip_count || 0) > 0 ? '+' : ''}${number(p.predicted_trips-(a?.trip_count || 0),1)}` : '—'}</strong></div><div>OBSERVED DURATION <strong>{a ? `${number(a.duration,1)} min` : '—'}</strong></div><div>OBSERVED FARE <strong>{a ? `$${number(a.fare,2)}` : '—'}</strong></div></div><p className="panel-foot">Duration and metered fare are conditional estimates for trips on this route.</p></>;
}
