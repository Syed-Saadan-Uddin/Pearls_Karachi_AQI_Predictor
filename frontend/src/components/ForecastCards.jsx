import React from 'react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { format, parseISO } from 'date-fns'

function ForecastCards({ predictions }) {
  const getTrendIcon = (current, previous) => {
    if (!previous) return null
    const diff = ((current - previous) / previous) * 100
    if (diff > 5) return <TrendingUp className="w-5 h-5 text-red-500" />
    if (diff < -5) return <TrendingDown className="w-5 h-5 text-green-500" />
    return <Minus className="w-5 h-5 text-gray-400" />
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {predictions.map((pred, index) => {
        const previous = index > 0 ? predictions[index - 1].predicted_aqi : null
        const date = parseISO(pred.date)
        
        return (
          <div
            key={pred.date}
            className="bg-white rounded-xl shadow-lg overflow-hidden transform transition-all hover:scale-105 hover:shadow-2xl"
            style={{ borderTop: `5px solid ${pred.color}` }}
          >
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-800">
                  {format(date, 'EEEE')}
                </h3>
                {getTrendIcon(pred.predicted_aqi, previous)}
              </div>
              
              <p className="text-sm text-gray-500 mb-2">
                {format(date, 'MMMM d, yyyy')}
              </p>
              
              <div className="mb-4">
                <div className="flex items-baseline space-x-2">
                  <span
                    className="text-4xl font-bold"
                    style={{ color: pred.color }}
                  >
                    {pred.predicted_aqi}
                  </span>
                  <span className="text-gray-500 text-sm">AQI</span>
                </div>
                <p
                  className="text-lg font-semibold mt-2"
                  style={{ color: pred.color }}
                >
                  {pred.category}
                </p>
              </div>
              
              <p className="text-sm text-gray-600 leading-relaxed">
                {pred.health_implication}
              </p>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default ForecastCards

