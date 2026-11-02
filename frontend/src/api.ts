export async function get<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const query = params ? `?${new URLSearchParams(Object.entries(params).map(([k,v]) => [k, String(v)]))}` : '';
  const response = await fetch(`/api${path}${query}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

// Intl formatters hold native resources; constructing one per option on every
// playback frame can exhaust a browser renderer even with a small JS heap.
const numberFormatters = new Map<number, Intl.NumberFormat>();
const dateFormatters = new Map<string, Intl.DateTimeFormat>();
export function number(n: number | undefined, digits = 0) {
  if (n == null || !Number.isFinite(n)) return '—';
  let formatter = numberFormatters.get(digits);
  if (!formatter) {
    formatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits });
    numberFormatters.set(digits, formatter);
  }
  return formatter.format(n);
}
export function when(value: string, options: Intl.DateTimeFormatOptions = {}) {
  const resolved = { timeZone: 'America/New_York', ...options };
  const key = JSON.stringify(Object.entries(resolved).sort(([a], [b]) => a.localeCompare(b)));
  let formatter = dateFormatters.get(key);
  if (!formatter) {
    formatter = new Intl.DateTimeFormat('en-US', resolved);
    if (dateFormatters.size >= 32) dateFormatters.delete(dateFormatters.keys().next().value!);
    dateFormatters.set(key, formatter);
  }
  return formatter.format(new Date(value));
}

export async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail || result));
  return result as T;
}
