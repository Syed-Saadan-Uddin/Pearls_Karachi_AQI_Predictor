import React from 'react'
import { format, parseISO } from 'date-fns'
import { MapPin, Clock, AlertCircle } from 'lucide-react'

function CurrentAQI({ currentAQI, loading, error }) {
  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-indigo-600"></div>
          <span className="ml-3 text-gray-600">Loading current AQI...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl shadow-lg p-6 mb-6">
        <div className="flex items-center space-x-3">
          <AlertCircle className="w-6 h-6 text-red-600" />
          <div>
            <h3 className="text-red-800 font-semibold">Error loading current AQI</h3>
            <p className="text-red-600 text-sm">{error}</p>
          </div>
        </div>
      </div>
    )
  }

  if (!currentAQI) {
    return null
  }

  const borderColor = currentAQI.color || '#667eea'
  const timestamp = currentAQI.timestamp
    ? format(parseISO(currentAQI.timestamp), 'PPpp')
    : 'Unknown'

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-6 border-t-4" style={{ borderTopColor: borderColor }}>
      <div className="flex items-start justify-between mb-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2 flex items-center">
            <span className="mr-2">️</span>
            Current Air Quality
          </h2>
          <div className="flex items-center space-x-4 text-sm text-gray-600">
            <div className="flex items-center space-x-1">
              <MapPin className="w-4 h-4" />
              <span>{currentAQI.city || 'Unknown Location'}</span>
            </div>
            <div className="flex items-center space-x-1">
              <Clock className="w-4 h-4" />
              <span>{timestamp}</span>
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="text-5xl font-bold mb-1" style={{ color: borderColor }}>
            {currentAQI.aqi}
          </div>
          <div className="text-lg font-semibold" style={{ color: borderColor }}>
            {currentAQI.category}
          </div>
        </div>
      </div>

      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
        <p className="text-gray-700 text-sm">{currentAQI.health_implication}</p>
      </div>

      {currentAQI.pollutants && (
        <div className="mt-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {Object.entries(currentAQI.pollutants).map(([key, value]) => {
            if (value === null || value === undefined) return null
            return (
              <div key={key} className="bg-gray-50 rounded-lg p-3">
                <div className="text-xs text-gray-500 uppercase mb-1">{key.replace('_', '.')}</div>
                <div className="text-lg font-semibold text-gray-800">
                  {typeof value === 'number' ? value.toFixed(1) : value}
                </div>
                <div className="text-xs text-gray-500">μg/m³</div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default CurrentAQI

