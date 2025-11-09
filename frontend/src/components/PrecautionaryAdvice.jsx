import React, { useState } from 'react'
import { AlertTriangle, CheckCircle, XCircle, Info, Heart } from 'lucide-react'

function PrecautionaryAdvice({ predictions }) {
  const [selectedDay, setSelectedDay] = useState(0)

  const getIcon = (advice) => {
    if (advice.includes('')) return <CheckCircle className="w-5 h-5 text-green-500" />
    if (advice.includes('️')) return <AlertTriangle className="w-5 h-5 text-yellow-500" />
    if (advice.includes('') || advice.includes('')) return <XCircle className="w-5 h-5 text-red-500" />
    return <Info className="w-5 h-5 text-blue-500" />
  }

  const cleanAdvice = (advice) => {
    return advice.replace(/[️‍️]/g, '').trim()
  }

  const selectedPrediction = predictions[selectedDay]

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="mb-6">
        <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
          <Heart className="w-6 h-6 text-red-500 mr-2" />
          Health Recommendations for {new Date(selectedPrediction.date).toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}
        </h3>
        
        {/* Day selector */}
        <div className="flex space-x-2 mb-6">
          {predictions.map((pred, index) => (
            <button
              key={index}
              onClick={() => setSelectedDay(index)}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                selectedDay === index
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Day {index + 1} ({pred.predicted_aqi} AQI)
            </button>
          ))}
        </div>

        {/* Current AQI Badge */}
        <div
          className="inline-flex items-center px-4 py-2 rounded-lg mb-6"
          style={{
            backgroundColor: `${selectedPrediction.color}20`,
            color: selectedPrediction.color,
            border: `2px solid ${selectedPrediction.color}`,
          }}
        >
          <span className="font-bold text-lg mr-2">{selectedPrediction.predicted_aqi}</span>
          <span className="font-semibold">{selectedPrediction.category}</span>
        </div>
      </div>

      {/* Advice List */}
      <div className="space-y-3">
        {selectedPrediction.precautionary_advice.map((advice, index) => (
          <div
            key={index}
            className="flex items-start space-x-3 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
          >
            {getIcon(advice)}
            <p className="text-gray-700 flex-1">{cleanAdvice(advice)}</p>
          </div>
        ))}
      </div>

      {/* Additional Tips */}
      <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <h4 className="font-semibold text-blue-800 mb-2 flex items-center">
          <Info className="w-5 h-5 mr-2" />
          General Tips
        </h4>
        <ul className="text-sm text-blue-700 space-y-1 list-disc list-inside">
          <li>Monitor air quality regularly, especially if you have respiratory conditions</li>
          <li>Keep indoor air clean with air purifiers and proper ventilation</li>
          <li>Stay hydrated and maintain a healthy diet to support your immune system</li>
          <li>Consider using N95 masks when air quality is poor</li>
        </ul>
      </div>
    </div>
  )
}

export default PrecautionaryAdvice

