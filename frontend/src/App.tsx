import { lazy, Suspense, useState } from 'react';
import { Activity, BrainCircuit, Car, Map } from 'lucide-react';
import { Simulator } from './Simulator';
import { TrainingPanel } from './TrainingPanel';

const ForecastExplorer = lazy(() => import('./ForecastExplorer'));

export default function App() {
  const [tab, setTab] = useState<'simulation' | 'forecast' | 'training'>('simulation');
  return <><nav className="app-nav" aria-label="Workspace"><a className="app-brand" href="#">MOTION <span>/ NYC</span><Activity size={17}/></a><div className="app-tabs">
    <button className={tab === 'simulation' ? 'active' : ''} onClick={() => setTab('simulation')}><Car size={16}/>Fleet simulation</button>
    <button className={tab === 'forecast' ? 'active' : ''} onClick={() => setTab('forecast')}><Map size={16}/>Demand forecast</button>
    <button className={tab === 'training' ? 'active' : ''} onClick={() => setTab('training')}><BrainCircuit size={16}/>Model & training</button>
  </div><span className="app-local"><i/> HISTORICAL TLC DATA</span></nav>
    <div hidden={tab !== 'simulation'}><Simulator active={tab === 'simulation'}/></div>
    {tab === 'forecast' && <Suspense fallback={<div className="operations-shell">Loading forecast explorer…</div>}><ForecastExplorer/></Suspense>}
    {tab === 'training' && <TrainingPanel/>}
  </>;
}
