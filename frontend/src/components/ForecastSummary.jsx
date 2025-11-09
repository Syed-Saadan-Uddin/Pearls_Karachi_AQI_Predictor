import React from 'react'
import { Info } from 'lucide-react'

function ForecastSummary({ summary }) {
  return (
    <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl shadow-lg p-6 text-white">
      <div className="flex items-start space-x-4">
        <Info className="w-6 h-6 mt-1 flex-shrink-0" />
        <div>
          <h3 className="text-xl font-semibold mb-2">Forecast Summary</h3>
          <p className="text-indigo-50 leading-relaxed">{summary}</p>
        </div>
      </div>
    </div>
  )
}

export default ForecastSummary

