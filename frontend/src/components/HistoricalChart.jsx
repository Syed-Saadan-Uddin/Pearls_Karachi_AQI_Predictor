import React from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceArea,
} from 'recharts'
import { format, parseISO } from 'date-fns'

function HistoricalChart({ data }) {
  const getCategoryColor = (aqi) => {
    if (aqi <= 50) return '#00e400'
    if (aqi <= 100) return '#ffff00'
    if (aqi <= 150) return '#ff7e00'
    if (aqi <= 200) return '#ff0000'
    if (aqi <= 300) return '#8f3f97'
    return '#7e0023'
  }

  const chartData = data.map((point) => ({
    ...point,
    date: format(parseISO(point.timestamp), 'MMM dd'),
    fullDate: point.timestamp,
  }))

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis
            dataKey="date"
            stroke="#666"
            tick={{ fill: '#666' }}
            interval="preserveStartEnd"
          />
          <YAxis
            stroke="#666"
            tick={{ fill: '#666' }}
            label={{ value: 'AQI', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #ccc',
              borderRadius: '8px',
            }}
            formatter={(value, name) => [
              `${value} (${chartData.find((d) => d.aqi === value)?.category})`,
              'AQI',
            ]}
            labelFormatter={(label) => {
              const point = chartData.find((d) => d.date === label)
              return point ? format(parseISO(point.fullDate), 'PPpp') : label
            }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="aqi"
            stroke="#667eea"
            strokeWidth={3}
            dot={{ fill: '#667eea', r: 4 }}
            activeDot={{ r: 6 }}
            name="AQI Value"
          />
          {/* Add reference areas for AQI categories */}
          <ReferenceArea y1={0} y2={50} fill="#00e400" fillOpacity={0.1} label="Good" />
          <ReferenceArea y1={50} y2={100} fill="#ffff00" fillOpacity={0.1} label="Moderate" />
          <ReferenceArea y1={100} y2={150} fill="#ff7e00" fillOpacity={0.1} label="USG" />
          <ReferenceArea y1={150} y2={200} fill="#ff0000" fillOpacity={0.1} label="Unhealthy" />
          <ReferenceArea y1={200} y2={300} fill="#8f3f97" fillOpacity={0.1} label="Very Unhealthy" />
        </LineChart>
      </ResponsiveContainer>
      
      {/* Legend */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
        {[
          { name: 'Good', range: '0-50', color: '#00e400' },
          { name: 'Moderate', range: '51-100', color: '#ffff00' },
          { name: 'USG', range: '101-150', color: '#ff7e00' },
          { name: 'Unhealthy', range: '151-200', color: '#ff0000' },
          { name: 'Very Unhealthy', range: '201-300', color: '#8f3f97' },
          { name: 'Hazardous', range: '301+', color: '#7e0023' },
        ].map((cat) => (
          <div key={cat.name} className="flex items-center space-x-2">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: cat.color }}
            />
            <div className="text-sm">
              <div className="font-semibold text-gray-700">{cat.name}</div>
              <div className="text-gray-500 text-xs">{cat.range}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default HistoricalChart

