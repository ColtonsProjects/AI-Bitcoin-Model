import React, { useEffect, useState, useRef } from 'react';
import { LineChart, ArrowUp, ArrowDown, Clock } from 'lucide-react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import { Line } from 'react-chartjs-2';

const intervals = ['1h', '4h', '24h', '7d', '30d'];
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

interface PriceEntry {
  open_time: string;
  close: number;
}

interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    borderColor: string;
    fill: boolean;
  }[];
}

export default function PriceChart({ 
  currentPrice, 
  selectedInterval, 
  onIntervalChange 
}: { 
  currentPrice: number;
  selectedInterval: string;
  onIntervalChange: (interval: string) => void;
}) {
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [livePrice, setLivePrice] = useState<number>(currentPrice);
  const [priceDifference, setPriceDifference] = useState<number | null>(null);
  const [predictedPrice, setPredictedPrice] = useState<number | null>(null);
  const hasFetched = useRef(false);

  useEffect(() => {
    const fetchCurrentPrice = () => {
      fetch('http://127.0.0.1:5001/price')
        .then(response => response.json())
        .then(data => setLivePrice(data))
        .catch(error => console.error('Error fetching current price:', error));
    };

    fetchCurrentPrice();
    const intervalId = setInterval(fetchCurrentPrice, 10000);

    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    let url = '';
    if (selectedInterval === '4h') {
      url = 'http://127.0.0.1:5001/four-hour-price';
    } else if (selectedInterval === '1h') {
      url = 'http://127.0.0.1:5001/hour-price';
    } else if (selectedInterval === '24h') {
      url = 'http://127.0.0.1:5001/day-price';
    } else if (selectedInterval === '7d') {
      url = 'http://127.0.0.1:5001/week-price';
    } else if (selectedInterval === '30d') {
      url = 'http://127.0.0.1:5001/month-price';
    }

    if (url) {
      fetch(url)
        .then(response => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.json();
        })
        .then(data => {
          const formattedData = {
            labels: data.map((entry: PriceEntry) => {
              const date = new Date(entry.open_time);
              
              // Manually add 5 hours to the date
              date.setHours(date.getHours());

              const options: Intl.DateTimeFormatOptions = {
                timeZone: 'UTC', // Use UTC since we're manually adjusting the time
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                hour12: true // Use 12-hour format to include AM/PM
              };

              return date.toLocaleString('en-US', options);
            }),
            datasets: [
              {
                label: 'BTC Price',
                data: data.map((entry: PriceEntry) => entry.close),
                borderColor: 'rgba(75,192,192,1)',
                fill: false,
              },
            ],
          };
          setChartData(formattedData);

          // Calculate price difference
          if (data.length > 0) {
            const oldestPrice = data[0].close;
            const difference = livePrice - oldestPrice;
            setPriceDifference(difference);
          }
        })
        .catch(error => console.error('Error fetching data:', error));
    }
  }, [selectedInterval, livePrice]);

  useEffect(() => {
    if (!hasFetched.current) {
      const fetchPredictedPrice = () => {
        fetch('http://127.0.0.1:5001/predict-next-price')
          .then(response => response.json())
          .then(data => setPredictedPrice(data.predicted_next_price))
          .catch(error => console.error('Error fetching predicted price:', error));
      };

      fetchPredictedPrice();
      hasFetched.current = true;
    }
  }, [selectedInterval]);

  const formatPriceDifference = (difference: number) => {
    const formattedDifference = Math.abs(difference).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
    return difference >= 0 ? `+$${formattedDifference}` : `-$${formattedDifference}`;
  };

  return (
    <div className="bg-white rounded-xl p-6 shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <LineChart className="w-6 h-6 text-indigo-600" />
          <h2 className="text-xl font-bold text-gray-800">Price Analysis</h2>
        </div>
        <div className="flex gap-2">
          {intervals.map((interval) => (
            <button
              key={interval}
              onClick={() => onIntervalChange(interval)}
              className={`px-3 py-1 rounded-full text-sm font-medium transition-all ${
                selectedInterval === interval
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {interval}
            </button>
          ))}
        </div>
      </div>
      
      <div className="relative h-64 mb-6 w-full">
        <div className="absolute inset-0 bg-gradient-to-b from-indigo-50/50 to-transparent rounded-lg"></div>
        {chartData ? (
          <Line 
            data={chartData} 
            options={{
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                x: {
                  grid: {
                    display: false
                  }
                },
                y: {
                  grid: {
                    color: 'rgba(0, 0, 0, 0.05)'
                  }
                }
              }
            }}
          />
        ) : (
          <p>Loading chart...</p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-5 h-5 text-indigo-600" />
            <span className="text-sm text-gray-600">Current Price</span>
          </div>
          <div className="flex items-center gap-2">
            <p className="text-2xl font-bold text-gray-900">${livePrice.toLocaleString()}</p>
            {priceDifference !== null && (
              <span className={`text-sm font-medium ${priceDifference >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {priceDifference >= 0 ? <ArrowUp className="w-4 h-4 inline" /> : <ArrowDown className="w-4 h-4 inline" />}
                {formatPriceDifference(priceDifference)}
              </span>
            )}
          </div>
        </div>
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <ArrowUp className="w-5 h-5 text-green-600" />
            <span className="text-sm text-gray-600">Tomorrow's Predicted Price</span>
          </div>
          <p className="text-2xl font-bold text-green-600">
            {predictedPrice !== null ? `$${predictedPrice.toLocaleString()}` : 'Loading...'}
          </p>
        </div>
      </div>
    </div>
  );
}

