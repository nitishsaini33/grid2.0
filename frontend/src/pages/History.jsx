import React, { useEffect, useState, useMemo } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
} from '@tanstack/react-table';
import apiClient from '../api/client';

export default function History() {
  const [data, setData] = useState([]);

  useEffect(() => {
    apiClient.get('/history').then(res => setData(res.data)).catch(console.error);
  }, []);

  const columns = useMemo(() => [
    { header: 'Start Time', accessorKey: 'start_datetime', cell: info => new Date(info.getValue()).toLocaleString() },
    { header: 'Cause', accessorKey: 'event_cause', cell: info => info.getValue()?.replace('_', ' ') },
    { header: 'Type', accessorKey: 'event_type' },
    { header: 'Priority', accessorKey: 'priority', cell: info => {
      const v = info.getValue();
      const color = v === 'High' ? 'text-red-400 bg-red-400/10 border-red-400/20' : 'text-blue-400 bg-blue-400/10 border-blue-400/20';
      return <span className={`px-2 py-1 rounded-full text-xs font-bold border ${color}`}>{v}</span>;
    }},
    { header: 'Zone', accessorKey: 'zone' },
    { header: 'Police Station', accessorKey: 'police_station' },
    { header: 'Severity', accessorKey: 'congestion_severity', cell: info => {
      const colors = ['text-emerald-400', 'text-yellow-400', 'text-orange-400', 'text-red-500'];
      const labels = ['Low', 'Moderate', 'High', 'Severe'];
      const v = info.getValue();
      return <span className={`font-bold ${colors[v] || ''}`}>{labels[v] || v}</span>;
    }},
    { header: 'Duration (m)', accessorKey: 'duration_min', cell: info => Math.round(info.getValue() || 0) },
    { header: 'Status', accessorKey: 'status', cell: info => (
      <span className={`capitalize ${info.getValue() === 'closed' ? 'text-emerald-400' : 'text-yellow-400'}`}>
        {info.getValue()}
      </span>
    )},
  ], []);

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <div className="card-gradient p-4 md:p-6 flex flex-col h-[75vh] md:h-[80vh] animate-in fade-in slide-in-from-bottom-4 duration-700">
      <h2 className="text-xl font-bold text-slate-200 border-b border-slate-700 pb-4 mb-4">📋 Historical Events (Last 500)</h2>
      <div className="flex-1 overflow-auto rounded-xl border border-slate-700 shadow-inner">
        <table className="w-full text-left border-collapse">
          <thead className="bg-slate-900 sticky top-0 z-10 shadow-md">
            {table.getHeaderGroups().map(headerGroup => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map(header => (
                  <th key={header.id} className="p-4 text-xs uppercase tracking-widest text-slate-400 font-bold border-b border-slate-700 whitespace-nowrap">
                    {flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody className="divide-y divide-slate-800">
            {table.getRowModel().rows.map(row => (
              <tr key={row.id} className="hover:bg-slate-800/50 transition-colors">
                {row.getVisibleCells().map(cell => (
                  <td key={cell.id} className="p-4 text-sm text-slate-300 whitespace-nowrap">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
            {data.length === 0 && (
              <tr><td colSpan={columns.length} className="p-8 text-center text-slate-500">Loading data...</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
