import React, { useEffect, useState } from 'react';
import apiClient from '../api/client';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function Features() {
  const [importance, setImportance] = useState([]);

  useEffect(() => {
    apiClient.get('/features/importance').then(res => setImportance(res.data)).catch(console.error);
  }, []);

  return (
    <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
      
      <div className="card-gradient p-4 md:p-6 flex flex-col h-[400px] md:h-[500px]">
        <h3 className="text-sm font-bold text-slate-300 mb-6 uppercase tracking-widest border-b border-slate-700 pb-2">
          Top 20 Feature Importances (Severity Model)
        </h3>
        {importance.length > 0 ? (
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={importance} layout="vertical" margin={{ top: 0, right: 30, left: 60, bottom: 0 }}>
                <XAxis type="number" stroke="#475569" />
                <YAxis dataKey="feature" type="category" stroke="#94a3b8" width={140} tick={{fontSize: 11}} />
                <Tooltip cursor={{fill: '#1e293b'}} contentStyle={{backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px'}} />
                <Bar dataKey="importance" fill="#22c55e" radius={[0, 4, 4, 0]} animationDuration={1500} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="text-slate-500">Feature importance N/A. Ensure models are trained.</div>
        )}
      </div>

      <div className="card-gradient p-4 md:p-6">
        <h3 className="text-sm font-bold text-slate-300 mb-4 uppercase tracking-widest border-b border-slate-700 pb-2">
          Correlation Matrix
        </h3>
        <div className="flex justify-center bg-slate-900/50 p-4 rounded-xl border border-slate-800">
          <img 
            src={`${apiClient.defaults.baseURL}/reports/correlation_matrix.png`} 
            alt="Correlation Matrix" 
            className="max-w-full h-auto rounded-lg opacity-90 hover:opacity-100 transition-opacity"
            onError={(e) => { e.target.parentElement.innerHTML = '<div class="text-slate-500 p-6 text-center">Run the pipeline first to generate correlation matrix.</div>'; }}
          />
        </div>
      </div>

    </div>
  );
}
