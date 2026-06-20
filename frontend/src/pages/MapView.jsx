import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import apiClient from '../api/client';

export default function MapView() {
  const [markers, setMarkers] = useState([]);

  useEffect(() => {
    apiClient.get('/map').then(res => setMarkers(res.data)).catch(console.error);
  }, []);

  const getSeverityColor = (severity) => {
    switch(severity) {
      case 0: return '#22c55e'; // Low - Green
      case 1: return '#eab308'; // Moderate - Yellow
      case 2: return '#f97316'; // High - Orange
      case 3: return '#ef4444'; // Severe - Red
      default: return '#94a3b8'; // Unknown
    }
  };

  return (
    <div className="card-gradient h-[80vh] flex flex-col p-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="mb-6 flex justify-between items-center border-b border-slate-700 pb-4">
        <h2 className="text-xl font-bold text-slate-200">📍 Traffic Event Hotspot Map</h2>
        <div className="flex gap-4 text-xs font-bold uppercase tracking-wider">
          <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]"></span>Low</div>
          <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-yellow-500 shadow-[0_0_8px_#eab308]"></span>Moderate</div>
          <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-orange-500 shadow-[0_0_8px_#f97316]"></span>High</div>
          <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_#ef4444]"></span>Severe</div>
        </div>
      </div>
      <div className="flex-1 rounded-xl overflow-hidden border border-slate-700 shadow-inner">
        <MapContainer center={[12.9716, 77.5946]} zoom={11} style={{ height: '100%', width: '100%' }} preferCanvas={true}>
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          />
          {markers.map((m, i) => (
            <CircleMarker
              key={i}
              center={[m.lat, m.lon]}
              radius={6}
              pathOptions={{
                color: getSeverityColor(m.severity),
                fillColor: getSeverityColor(m.severity),
                fillOpacity: 0.6,
                weight: 1
              }}
            >
              <Popup>
                <div className="text-slate-900 font-sans p-1">
                  <strong>Cause:</strong> {m.cause.replace('_', ' ')}<br/>
                  <strong>Zone:</strong> {m.zone}<br/>
                  <strong>Severity Class:</strong> {m.severity}
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
    </div>
  );
}
