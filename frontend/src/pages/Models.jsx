import React, { useEffect, useState } from 'react';
import apiClient from '../api/client';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function Models() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    apiClient.get('/models/metrics').then(res => setMetrics(res.data)).catch(console.error);
  }, []);

  if (!metrics) {
    return <div className="text-slate-400 p-6">Loading model results...</div>;
  }

  const { classification, regression } = metrics;
  
  // Format for BarChart (Comparing balanced accuracy across models)
  const chartDataMap = {};
  const targetsSet = new Set();
  
  if (classification) {
    Object.entries(classification).forEach(([target, info]) => {
      const targetName = target.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      targetsSet.add(targetName);
      Object.entries(info.all_models || {}).forEach(([modelName, modelMetrics]) => {
        if (!chartDataMap[modelName]) chartDataMap[modelName] = { name: modelName };
        chartDataMap[modelName][targetName] = modelMetrics.balanced_accuracy || 0;
      });
    });
  }
  const chartData = Object.values(chartDataMap);
  const targets = Array.from(targetsSet);
  const colors = ['#6366f1', '#a855f7', '#ec4899', '#f97316'];

  // Table rows
  const tableRows = [];
  if (classification) {
    Object.entries(classification).forEach(([target, info]) => {
      const targetName = target.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      const tm = info.test_metrics || {};
      tableRows.push({
        target: targetName,
        bestModel: info.best_model || 'N/A',
        accuracy: `${(tm.accuracy * 100 || 0).toFixed(1)}%`,
        balancedAcc: `${(tm.balanced_accuracy * 100 || 0).toFixed(1)}%`,
        f1: (tm.f1 || 0).toFixed(3),
        auc: tm.roc_auc ? tm.roc_auc.toFixed(3) : '—',
        kappa: (tm.cohen_kappa || 0).toFixed(3),
      });
    });
  }

  const regTm = regression?.test_metrics || {};

  return (
    <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
      
      <div className="card-gradient p-6 flex flex-col h-96">
        <h3 className="text-sm font-bold text-slate-300 mb-6 uppercase tracking-widest border-b border-slate-700 pb-2">
          Model Comparison — Balanced Accuracy
        </h3>
        {chartData.length > 0 ? (
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                <XAxis dataKey="name" stroke="#475569" tick={{fontSize: 11}} angle={-30} textAnchor="end" />
                <YAxis stroke="#475569" tick={{fontSize: 11}} />
                <Tooltip cursor={{fill: '#1e293b'}} contentStyle={{backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px'}} />
                <Legend wrapperStyle={{fontSize: 12, paddingTop: '10px'}} />
                {targets.map((tgt, i) => (
                  <Bar key={tgt} dataKey={tgt} fill={colors[i % colors.length]} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="text-slate-500">No model comparison data available.</div>
        )}
      </div>

      <div className="card-gradient p-6">
        <h3 className="text-sm font-bold text-slate-300 mb-4 uppercase tracking-widest border-b border-slate-700 pb-2">
          Classification Model Test Results
        </h3>
        {tableRows.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-800 text-[11px] uppercase tracking-wider text-slate-400 border-b border-slate-700">
                  <th className="p-3 font-semibold">Target</th>
                  <th className="p-3 font-semibold">Best Model</th>
                  <th className="p-3 font-semibold">Accuracy</th>
                  <th className="p-3 font-semibold">Balanced Acc</th>
                  <th className="p-3 font-semibold">F1</th>
                  <th className="p-3 font-semibold">AUC</th>
                  <th className="p-3 font-semibold">Kappa</th>
                </tr>
              </thead>
              <tbody>
                {tableRows.map((r, idx) => (
                  <tr key={idx} className={`border-b border-slate-800/50 text-sm ${idx % 2 === 0 ? 'bg-slate-900/30' : 'bg-transparent'}`}>
                    <td className="p-3 text-slate-200">{r.target}</td>
                    <td className="p-3 text-indigo-400 font-medium">{r.bestModel}</td>
                    <td className="p-3 text-slate-300">{r.accuracy}</td>
                    <td className="p-3 text-slate-300">{r.balancedAcc}</td>
                    <td className="p-3 text-slate-300">{r.f1}</td>
                    <td className="p-3 text-slate-300">{r.auc}</td>
                    <td className="p-3 text-slate-300">{r.kappa}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-slate-500">No classification results yet.</div>
        )}
      </div>

      <div className="card-gradient p-6">
        <h3 className="text-sm font-bold text-slate-300 mb-4 uppercase tracking-widest border-b border-slate-700 pb-2">
          Regression — Duration Prediction
        </h3>
        {regression ? (
          <div>
            <p className="text-slate-400 mb-4 text-sm">
              <span className="font-semibold text-indigo-400">Best:</span> {regression.best_model || 'N/A'} |{' '}
              <span className="font-semibold text-indigo-400">MAE:</span> {(regTm.MAE || 0).toFixed(1)} min |{' '}
              <span className="font-semibold text-indigo-400">R²:</span> {(regTm.R2 || 0).toFixed(3)}
            </p>
            <div className="mt-4 flex justify-center bg-slate-900/50 p-4 rounded-xl border border-slate-800">
              <img 
                src="http://localhost:8000/api/reports/duration_residuals.png" 
                alt="Duration Residuals" 
                className="max-w-full h-auto rounded-lg opacity-90 hover:opacity-100 transition-opacity"
                onError={(e) => { e.target.parentElement.innerHTML = '<div class="text-slate-500 p-6 text-center">No duration residual plot available.</div>'; }}
              />
            </div>
          </div>
        ) : (
          <div className="text-slate-500">No regression results yet.</div>
        )}
      </div>

    </div>
  );
}
