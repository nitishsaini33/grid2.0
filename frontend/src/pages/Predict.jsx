import React, { useState } from 'react';
import apiClient from '../api/client';

export default function Predict() {
  const [formData, setFormData] = useState({
    event_cause: 'accident',
    event_type: 'unplanned',
    priority: 'High',
    road_closure: false,
    lat: 12.9716,
    lon: 77.5946,
    hour: 8
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const causes = ["vehicle_breakdown", "accident", "construction", "water_logging", "pot_holes", "tree_fall", "congestion", "public_event", "vip_movement", "procession", "protest", "road_conditions", "others"];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await apiClient.post('/predict', formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  const getSeverityColor = (cls) => {
    const colors = ['text-emerald-400', 'text-yellow-400', 'text-orange-400', 'text-red-500'];
    return colors[cls] || 'text-slate-400';
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="card-gradient p-6 col-span-1">
        <h2 className="text-xl font-bold text-slate-200 border-b border-slate-700 pb-4 mb-6">⚙️ Event Parameters</h2>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div>
            <label className="block text-xs font-bold uppercase tracking-widest text-slate-400 mb-2">Event Cause</label>
            <select className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-slate-200" value={formData.event_cause} onChange={e => setFormData({...formData, event_cause: e.target.value})}>
              {causes.map(c => <option key={c} value={c}>{c.replace('_', ' ')}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs font-bold uppercase tracking-widest text-slate-400 mb-2">Event Type</label>
            <select className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-slate-200" value={formData.event_type} onChange={e => setFormData({...formData, event_type: e.target.value})}>
              <option value="unplanned">Unplanned</option>
              <option value="planned">Planned</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-bold uppercase tracking-widest text-slate-400 mb-2">Priority</label>
            <select className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-slate-200" value={formData.priority} onChange={e => setFormData({...formData, priority: e.target.value})}>
              <option value="High">High</option>
              <option value="Low">Low</option>
            </select>
          </div>
          <div className="flex items-center gap-3 mt-2">
            <input type="checkbox" id="road_closure" className="w-4 h-4 accent-indigo-500" checked={formData.road_closure} onChange={e => setFormData({...formData, road_closure: e.target.checked})} />
            <label htmlFor="road_closure" className="text-sm font-bold text-slate-300">Requires Road Closure?</label>
          </div>
          <div className="grid grid-cols-2 gap-4 mt-2">
            <div>
              <label className="block text-xs font-bold uppercase tracking-widest text-slate-400 mb-2">Latitude</label>
              <input type="number" step="0.001" className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-slate-200" value={formData.lat} onChange={e => setFormData({...formData, lat: parseFloat(e.target.value)})} />
            </div>
            <div>
              <label className="block text-xs font-bold uppercase tracking-widest text-slate-400 mb-2">Longitude</label>
              <input type="number" step="0.001" className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-slate-200" value={formData.lon} onChange={e => setFormData({...formData, lon: parseFloat(e.target.value)})} />
            </div>
          </div>
          <div>
            <label className="block text-xs font-bold uppercase tracking-widest text-slate-400 mb-2 mt-2">Hour of Day ({formData.hour}:00)</label>
            <input type="range" min="0" max="23" className="w-full accent-indigo-500" value={formData.hour} onChange={e => setFormData({...formData, hour: parseInt(e.target.value)})} />
          </div>
          
          <button type="submit" disabled={loading} className="mt-4 w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-3 rounded-xl shadow-lg shadow-indigo-500/30 transition-all disabled:opacity-50">
            {loading ? 'Predicting...' : '🔮 Predict Impact'}
          </button>
        </form>
      </div>

      <div className="card-gradient p-6 col-span-2 flex flex-col">
        <h2 className="text-xl font-bold text-slate-200 border-b border-slate-700 pb-4 mb-6">📋 Prediction Results</h2>
        
        {result ? (
          <div className="flex-1 flex flex-col gap-6">
            <div className="bg-slate-900/50 border border-slate-700 rounded-xl p-6 text-center shadow-inner">
              <div className="text-sm uppercase tracking-widest text-slate-400 font-bold mb-2">Predicted Congestion Severity</div>
              <div className={`text-5xl font-extrabold ${getSeverityColor(result.congestion_severity.class)} drop-shadow-lg`}>
                {result.congestion_severity.label}
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-slate-800/80 border border-slate-700 rounded-xl p-4 text-center">
                <div className="text-2xl mb-2">⏱</div>
                <div className="text-xs uppercase tracking-widest text-slate-400 font-bold mb-1">Duration</div>
                <div className="text-xl font-bold text-slate-200">{result.predicted_duration_min} min</div>
              </div>
              <div className="bg-slate-800/80 border border-slate-700 rounded-xl p-4 text-center">
                <div className="text-2xl mb-2">👮</div>
                <div className="text-xs uppercase tracking-widest text-slate-400 font-bold mb-1">Manpower</div>
                <div className="text-xl font-bold text-slate-200">{result.manpower.officers_required} officers</div>
              </div>
              <div className="bg-slate-800/80 border border-slate-700 rounded-xl p-4 text-center">
                <div className="text-2xl mb-2">🚧</div>
                <div className="text-xs uppercase tracking-widest text-slate-400 font-bold mb-1">Barricades</div>
                <div className="text-xl font-bold text-slate-200">Class {result.barricades.class}</div>
              </div>
              <div className="bg-slate-800/80 border border-slate-700 rounded-xl p-4 text-center">
                <div className="text-2xl mb-2">↩️</div>
                <div className="text-xs uppercase tracking-widest text-slate-400 font-bold mb-1">Diversion</div>
                <div className="text-xl font-bold text-slate-200">{result.diversion.class ? 'Required' : 'None'}</div>
              </div>
            </div>
            
            <div className="mt-auto text-center text-sm text-emerald-500 font-bold">
              ✅ Event saved to database successfully with ID #{result.event_id}
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-500 opacity-50">
            <div className="text-6xl mb-4">🔮</div>
            <p>Enter event parameters and click Predict Impact</p>
          </div>
        )}
      </div>
    </div>
  );
}
