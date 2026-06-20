import React, { useEffect, useState } from 'react';
import apiClient from '../api/client';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function Overview() {
  const [kpis, setKpis] = useState(null);
  const [causes, setCauses] = useState([]);
  const [durations, setDurations] = useState([]);
  const [zones, setZones] = useState([]);

  useEffect(() => {
    apiClient.get('/kpis').then(res => setKpis(res.data)).catch(console.error);
    apiClient.get('/charts/causes').then(res => setCauses(res.data)).catch(console.error);
    apiClient.get('/charts/durations').then(res => setDurations(res.data)).catch(console.error);
    apiClient.get('/charts/zones').then(res => setZones(res.data)).catch(console.error);
  }, []);

  const KpiCard = ({ title, value, icon, subtitle, color }) => (
    <div className="card-gradient p-5 text-center transform transition-transform hover:scale-105 hover:shadow-indigo-500/10">
      <div className="text-3xl mb-2 drop-shadow-md">{icon}</div>
      <div className="text-[11px] text-slate-400 uppercase tracking-widest font-bold mb-1">{title}</div>
      <div className={`text-4xl font-extrabold leading-none ${color} drop-shadow-lg`}>{value}</div>
      <div className="text-[11px] text-slate-500 mt-2">{subtitle}</div>
    </div>
  );

  return (
    <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        <KpiCard title="Total Events" value={kpis?.total_events?.toLocaleString() || '-'} icon="📋" subtitle="All records" color="text-indigo-400" />
        <KpiCard title="Avg Severity" value={kpis?.avg_severity || '-'} icon="⚠️" subtitle="0–3 scale" color="text-orange-500" />
        <KpiCard title="Median Dur." value={kpis?.median_duration ? `${kpis.median_duration}m` : '-'} icon="⏱" subtitle="Per event" color="text-purple-400" />
        <KpiCard title="Severe Events" value={kpis?.severe_events || '-'} icon="🔴" subtitle="Class 3" color="text-red-500" />
        <KpiCard title="Road Closures" value={kpis?.road_closures || '-'} icon="🚧" subtitle="Requires closure" color="text-yellow-500" />
        <KpiCard title="Closed Events" value={kpis?.closed_pct ? `${kpis.closed_pct}%` : '-'} icon="✅" subtitle="Resolution rate" color="text-emerald-500" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card-gradient p-4 md:p-6 h-80 md:h-96 flex flex-col">
          <h3 className="text-sm font-bold text-slate-300 mb-6 uppercase tracking-widest border-b border-slate-700 pb-2">Event Cause Distribution</h3>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={causes} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
                <XAxis type="number" stroke="#475569" />
                <YAxis dataKey="cause" type="category" stroke="#94a3b8" width={100} tick={{fontSize: 11}} />
                <Tooltip cursor={{fill: '#1e293b'}} contentStyle={{backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px'}} />
                <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]} animationDuration={1500} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card-gradient p-4 md:p-6 h-80 md:h-96 flex flex-col">
          <h3 className="text-sm font-bold text-slate-300 mb-6 uppercase tracking-widest border-b border-slate-700 pb-2">Zone-wise Mean Severity</h3>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={zones} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
                <XAxis type="number" stroke="#475569" />
                <YAxis dataKey="zone" type="category" stroke="#94a3b8" width={100} tick={{fontSize: 11}} />
                <Tooltip cursor={{fill: '#1e293b'}} contentStyle={{backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px'}} />
                <Bar dataKey="mean_sev" fill="#f97316" radius={[0, 4, 4, 0]} animationDuration={1500} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card-gradient p-4 md:p-6 h-80 md:h-96 md:col-span-2 flex flex-col">
          <h3 className="text-sm font-bold text-slate-300 mb-6 uppercase tracking-widest border-b border-slate-700 pb-2">Event Duration Distribution (min)</h3>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={durations} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="duration" stroke="#475569" tick={{fontSize: 11}} />
                <YAxis stroke="#475569" tick={{fontSize: 11}} />
                <Tooltip cursor={{fill: '#1e293b'}} contentStyle={{backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px'}} />
                <Bar dataKey="count" fill="#a855f7" radius={[4, 4, 0, 0]} animationDuration={1500} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
