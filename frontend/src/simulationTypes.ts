export type Vehicle = { id: number; zone: number; target: number; status: 'idle' | 'busy'; started_at: number; busy_until: number };
export type Snapshot = { at: number; requested: number; accepted: number; completed: number; rejected: number; pending: number; in_service: number; revenue: number; fuel_cost: number; driver_cost: number; net_revenue: number; miles: number; utilization: number; mean_wait_minutes: number; vehicles: Vehicle[] };
export type Decision = { id: number; at: number; source: number; target: number; accepted: boolean; reason: string; vehicle: number | null; fare: number; score: number | null; estimated_contribution: number | null };
export type PolicyResult = { policy: string; summary: Omit<Snapshot, 'at' | 'vehicles'>; timeline: Snapshot[]; decisions: Decision[]; rejection_reasons: Record<string, number> };
export type SimulationResult = { start_time: string; request_count: number; source: string; notice: string; baseline: PolicyResult; optimized: PolicyResult; net_uplift: number; config: { hours: number; fleet_size: number } };
export type Job<T> = { id: string; kind: string; status: 'queued' | 'running' | 'complete' | 'failed'; error?: string; result?: T };
