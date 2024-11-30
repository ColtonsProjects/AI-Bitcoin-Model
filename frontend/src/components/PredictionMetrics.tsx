import React from 'react';
import { Target, ArrowUpRight, AlertTriangle, BarChart2 } from 'lucide-react';

export default function PredictionMetrics() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="bg-white rounded-xl p-6 shadow-lg">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 bg-blue-50 rounded-lg">
            <Target className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-800">Accuracy Score</h3>
            <p className="text-sm text-gray-500">Last 7 days prediction accuracy</p>
          </div>
        </div>
        <p className="text-3xl font-bold text-blue-600">94.8%</p>
        <div className="mt-2 flex items-center gap-1">
          <ArrowUpRight className="w-4 h-4 text-green-500" />
          <span className="text-sm text-green-500">+2.4% from last week</span>
        </div>
      </div>

      <div className="bg-white rounded-xl p-6 shadow-lg">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 bg-purple-50 rounded-lg">
            <AlertTriangle className="w-6 h-6 text-purple-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-800">Volatility Index</h3>
            <p className="text-sm text-gray-500">Market volatility indicator</p>
          </div>
        </div>
        <p className="text-3xl font-bold text-purple-600">Medium</p>
        <div className="mt-2 flex items-center gap-1">
          <BarChart2 className="w-4 h-4 text-yellow-500" />
          <span className="text-sm text-yellow-500">Stable conditions</span>
        </div>
      </div>
    </div>
  );
}