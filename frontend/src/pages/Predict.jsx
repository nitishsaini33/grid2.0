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

      <div className="col-span-2 flex flex-col gap-6">
        <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-purple-400 flex items-center gap-3">
          <span className="text-3xl">📋</span> Prediction Results
        </h2>
        
        {loading ? (
          <div className="flex-1 flex flex-col items-center justify-center bg-[#1e2330] border border-slate-700 rounded-xl p-12 min-h-[400px]">
            <div className="relative mb-6">
              <div className="w-16 h-16 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
              <div className="absolute inset-0 flex items-center justify-center text-2xl">🔮</div>
            </div>
            <h3 className="text-xl font-bold text-indigo-400 mb-2 animate-pulse">Analyzing Event Intelligence...</h3>
            <p className="text-slate-500">Running spatial models and historical simulations</p>
          </div>
        ) : result ? (
          <div className="flex flex-col gap-6 animate-in fade-in zoom-in-95 duration-500">
            
            {/* Top Card: Severity & Duration */}
            <div className="bg-[#1e2330] border border-slate-700 rounded-xl p-6 shadow-lg flex flex-col md:flex-row">
              <div className="flex-1 pr-6 md:border-r border-slate-700 flex flex-col justify-center">
                <div className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-4">Congestion Severity</div>
                <div className={`text-5xl font-extrabold mb-4 ${getSeverityColor(result.congestion_severity.class)}`}>
                  {result.congestion_severity.label}
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2.5 mb-3">
                  <div className={`h-2.5 rounded-full ${getSeverityColor(result.congestion_severity.class).replace('text-', 'bg-')}`} style={{ width: `${(result.congestion_severity.class + 1) * 25}%` }}></div>
                </div>
                <div className="text-sm text-slate-500">Confidence: 100.0%</div>
              </div>
              <div className="flex-1 pl-0 md:pl-6 pt-6 md:pt-0 flex flex-col items-center justify-center text-center">
                <div className="text-6xl font-extrabold text-purple-400 mb-2">{result.predicted_duration_min}</div>
                <div className="text-sm text-slate-400 mb-1 font-medium">minutes</div>
                <div className="text-sm text-slate-500 mb-5">≈ {(result.predicted_duration_min / 60).toFixed(1)} hours</div>
                <div className="text-xs font-bold uppercase tracking-widest text-slate-500">Expected Duration</div>
              </div>
            </div>

            {/* 3 Info Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Officers */}
              <div className="bg-[#1e2330] border border-slate-700 rounded-xl p-6 text-center shadow-lg">
                <div className="text-4xl mb-4">👮</div>
                <div className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-3">Police Officers</div>
                <div className="text-5xl font-extrabold text-blue-500 mb-3">{result.manpower.officers_required}</div>
                <div className="text-sm text-slate-500">Class {result.manpower.officers_required <= 4 ? 1 : result.manpower.officers_required <= 8 ? 2 : 3} deployment</div>
              </div>
              {/* Barricades */}
              <div className="bg-[#1e2330] border border-slate-700 rounded-xl p-6 text-center shadow-lg">
                <div className="text-4xl mb-4">🚧</div>
                <div className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-3">Barricades</div>
                <div className="text-5xl font-extrabold text-orange-500 mb-3">{[0, 2, 5, 10][result.barricades.class] || 0}</div>
                <div className="text-sm text-slate-500">{["None", "Low barricading", "Medium barricading", "High barricading"][result.barricades.class] || "None"}</div>
              </div>
              {/* Diversion */}
              <div className="bg-[#1e2330] border border-slate-700 rounded-xl p-6 text-center shadow-lg">
                <div className="text-4xl mb-4">🔀</div>
                <div className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-3">Diversion</div>
                <div className={`text-5xl font-extrabold mb-3 ${result.diversion.class ? 'text-red-500' : 'text-emerald-500'}`}>
                  {result.diversion.class ? 'YES' : 'NO'}
                </div>
                <div className="text-sm text-slate-500">{result.diversion.class ? 'Diversion Required' : 'No Diversion'}</div>
              </div>
            </div>

            {/* Action Plan */}
            <div className="bg-[#1e2330] border border-slate-700 rounded-xl p-6 shadow-lg">
              <div className="flex items-center gap-2 mb-5">
                <span className="text-xl">📋</span>
                <h3 className="text-lg font-bold text-indigo-300">Action Plan</h3>
              </div>
              <ul className="space-y-3 pl-2">
                <li className="flex items-start gap-3">
                  <span className="text-slate-500 mt-1">•</span>
                  <span className="text-slate-300">Deploy {result.manpower.officers_required} officers to the site</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-slate-500 mt-1">•</span>
                  <span className="text-slate-300">Set up {[0, 2, 5, 10][result.barricades.class] || 0} {["", "low", "medium", "high"][result.barricades.class] || ""} barricades</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-slate-500 mt-1">•</span>
                  <span className={result.diversion.class ? "text-red-400 font-medium" : "text-emerald-400 font-medium"}>
                    {result.diversion.class ? "Implement immediate traffic diversion" : "No diversion needed at this time"}
                  </span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-slate-500 mt-1">•</span>
                  <span className="text-slate-300">Expect incident to last approximately {result.predicted_duration_min} minutes ({(result.predicted_duration_min / 60).toFixed(1)} hours)</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-slate-500 mt-1">•</span>
                  <span className="text-slate-300">Severity Level: {result.congestion_severity.label} — coordinate with {result.manpower.officers_required} officers</span>
                </li>
              </ul>
            </div>
            
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-500 opacity-50 bg-[#1e2330] border border-slate-700 rounded-xl p-12">
            <div className="text-6xl mb-4">🔮</div>
            <p>Enter event parameters and click Predict Impact</p>
          </div>
        )}
      </div>
    </div>
  );
}
