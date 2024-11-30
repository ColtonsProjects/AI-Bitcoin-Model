import React, { useState, useEffect } from 'react';
import { Bitcoin } from 'lucide-react';
import PriceChart from './components/PriceChart';
import PredictionMetrics from './components/PredictionMetrics';

function App() {
  const [currentPrice, setCurrentPrice] = useState(45000);
  const [selectedInterval, setSelectedInterval] = useState('1h');

  // Simulate price updates
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPrice(prev => {
        const change = (Math.random() - 0.5) * 1000;
        return Math.max(0, prev + change);
      });
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <Bitcoin className="w-8 h-8 text-indigo-600" />
              <span className="text-xl font-bold text-gray-900">CryptoForecast AI</span>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Bitcoin Price Forecast</h1>
          <p className="mt-2 text-gray-600">
            Real-time price tracking and AI-powered predictions
          </p>
        </div>

        <div className="grid gap-8">
          <PriceChart
            currentPrice={currentPrice}
            selectedInterval={selectedInterval}
            onIntervalChange={setSelectedInterval}
          />
          <PredictionMetrics />
        </div>

        <div className="mt-8 bg-white rounded-xl p-6 shadow-lg">
          <h2 className="text-xl font-bold text-gray-800 mb-4">About Our Model</h2>
          <p className="text-gray-600 leading-relaxed">
            Our advanced machine learning model combines historical price data, market sentiment analysis, 
            and technical indicators to provide accurate Bitcoin price predictions. The model is continuously 
            trained on real-time data and adapts to changing market conditions.
          </p>
        </div>
      </main>

      <footer className="bg-white mt-12 border-t border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-gray-500 text-sm">
            © {new Date().getFullYear()} CryptoForecast AI. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;