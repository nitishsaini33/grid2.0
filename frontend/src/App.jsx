import React, { useState } from 'react';
import Overview from './pages/Overview';
import MapView from './pages/MapView';
import Predict from './pages/Predict';
import Models from './pages/Models';
import Features from './pages/Features';
import History from './pages/History';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderTab = () => {
    switch(activeTab) {
      case 'overview': return <Overview />;
      case 'map': return <MapView />;
      case 'predict': return <Predict />;
      case 'models': return <Models />;
      case 'features': return <Features />;
      case 'history': return <History />;
      default: return <Overview />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans flex flex-col">
      <header className="bg-slate-900 border-b border-slate-800 p-4 md:p-6 flex justify-between items-center shadow-lg">
        <div className="flex items-center gap-4">
          <div className="text-4xl">🚦</div>
          <div>
            <h1 className="text-xl md:text-2xl font-bold text-gradient m-0">ClearWay — Traffic Impact Predictor</h1>
            <p className="hidden md:block text-sm text-slate-400 m-0 tracking-wide">Event-Driven Traffic Impact Prediction & Resource Optimization</p>
          </div>
        </div>

      </header>

      <nav className="bg-slate-900 border-b border-slate-800 px-6 py-3 flex gap-2 overflow-x-auto">
        {[
          { id: 'overview', label: '📊 Overview' },
          { id: 'map', label: '🗺 Hotspot Map' },
          { id: 'predict', label: '🤖 Predict Event' },
          { id: 'models', label: '📈 Model Results' },
          { id: 'features', label: '🔍 Feature Analysis' },
          { id: 'history', label: '📋 Historical Data' },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-5 py-2.5 rounded-xl font-medium transition-all duration-300 whitespace-nowrap ${
              activeTab === tab.id 
                ? 'bg-indigo-500 text-white shadow-lg shadow-indigo-500/25 border border-indigo-400' 
                : 'text-slate-400 hover:text-white hover:bg-slate-800 border border-transparent'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="flex-1 p-4 md:p-6 overflow-auto">
        {renderTab()}
      </main>
    </div>
  );
}
