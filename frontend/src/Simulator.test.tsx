import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Simulator } from './Simulator';
import { get, post } from './api';
import type { SimulationResult } from './simulationTypes';

vi.mock('./api', async importOriginal => ({ ...await importOriginal<typeof import('./api')>(), get: vi.fn(), post: vi.fn() }));
vi.mock('recharts', () => ({
  ResponsiveContainer: ({children}: {children: React.ReactNode}) => <div>{children}</div>,
  LineChart: ({children}: {children: React.ReactNode}) => <div>{children}</div>,
  CartesianGrid: () => null, Legend: () => null, Line: () => null, Tooltip: () => null, XAxis: () => null, YAxis: () => null,
}));

const start = '2026-04-01T00:00:00-04:00';
const snapshot = { at: 0, requested: 0, accepted: 0, completed: 0, rejected: 0, pending: 0,
  in_service: 0, revenue: 0, fuel_cost: 0, driver_cost: 0, net_revenue: 0, miles: 0,
  utilization: 0, mean_wait_minutes: 0, vehicles: [] };
const result = { start_time: start, request_count: 0, source: 'test', notice: 'test', net_uplift: 0,
  config: { hours: 2, fleet_size: 80 },
  optimized: {policy:'optimized', summary:snapshot, timeline:[snapshot,{...snapshot, at:7200}], decisions:[], rejection_reasons:{}},
  baseline: {policy:'baseline', summary:snapshot, timeline:[snapshot,{...snapshot, at:7200}], decisions:[], rejection_reasons:{}},
} as SimulationResult;

async function startPlayback() {
  const times = Array.from({length:4392}, (_,i) => new Date(Date.parse(start)+i*3600000).toISOString());
  vi.mocked(get).mockImplementation(async path => {
    if (path === '/zones') return {type:'FeatureCollection', features:[]};
    if (path === '/available-times') return {times};
    return {id:'test-job',kind:'simulation',status:'complete',result};
  });
  vi.mocked(post).mockResolvedValue({id:'test-job',kind:'simulation',status:'queued'});
  const view = render(<Simulator/>);
  await act(async () => { await Promise.resolve(); });
  fireEvent.click(screen.getByRole('button', {name:'Run comparison'}));
  await act(async () => { await Promise.resolve(); });
  expect(screen.getByRole('button', {name:'Pause simulation'})).toBeTruthy();
  return view;
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.useRealTimers(); vi.mocked(get).mockReset(); vi.mocked(post).mockReset(); });

describe('simulation playback resource regression', () => {
  it('keeps historical options unchanged while the 100× clock advances', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'performance'] });
    await startPlayback();
    const selector = screen.getByLabelText('Historical start');
    const changes: MutationRecord[] = [];
    const observer = new MutationObserver(records => changes.push(...records));
    observer.observe(selector, {childList:true,subtree:true,characterData:true});
    const formatter = vi.spyOn(Intl, 'DateTimeFormat');
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(Number((screen.getByLabelText('Simulation time') as HTMLInputElement).value)).toBeCloseTo(200, 0);
    expect(selector.children.length).toBe(4392);
    expect([...changes, ...observer.takeRecords()]).toHaveLength(0);
    // No thousands of fresh native Intl objects on each playback frame.
    expect(formatter.mock.calls.length).toBeLessThan(5);
    observer.disconnect();
  });

  it('suspends playback when the workspace is inactive, preserving the scenario', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'performance'] });
    const view = await startPlayback();
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    const before = (screen.getByLabelText('Simulation time') as HTMLInputElement).value;
    view.rerender(<Simulator active={false}/>);
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect((screen.getByLabelText('Simulation time') as HTMLInputElement).value).toBe(before);
    view.rerender(<Simulator active/>);
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(Number((screen.getByLabelText('Simulation time') as HTMLInputElement).value)).toBeCloseTo(Number(before)+100,0);
  });
});
