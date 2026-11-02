import type { ReactNode } from 'react';

export function SummaryCard({ label, value, detail, icon }: { label: string; value: string; detail: string; icon: ReactNode }) {
  return <div className="stat-card"><div className="stat-top"><span>{label}</span><span className="stat-icon">{icon}</span></div><div className="stat-value">{value}</div><div className="stat-detail">{detail}</div></div>;
}
