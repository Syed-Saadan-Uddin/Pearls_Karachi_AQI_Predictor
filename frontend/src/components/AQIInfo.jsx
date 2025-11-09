import React from 'react'
import { Info, Wind, Droplets, Sun } from 'lucide-react'

function AQIInfo() {
  const categories = [
    {
      name: 'Good',
      range: '0-50',
      color: '#00e400',
      description: 'Air quality is satisfactory, and air pollution poses little or no risk.',
    },
    {
      name: 'Moderate',
      range: '51-100',
      color: '#ffff00',
      description:
        'Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.',
    },
    {
      name: 'Unhealthy for Sensitive Groups',
      range: '101-150',
      color: '#ff7e00',
      description:
        'Members of sensitive groups may experience health effects. The general public is less likely to be affected.',
    },
    {
      name: 'Unhealthy',
      range: '151-200',
      color: '#ff0000',
      description:
        'Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.',
    },
    {
      name: 'Very Unhealthy',
      range: '201-300',
      color: '#8f3f97',
      description: 'Health alert: The risk of health effects is increased for everyone.',
    },
    {
      name: 'Hazardous',
      range: '301+',
      color: '#7e0023',
      description:
        'Health warning of emergency conditions: everyone is more likely to be affected.',
    },
  ]

  const pollutants = [
    { name: 'PM2.5', icon: <Wind className="w-5 h-5" />, description: 'Fine particulate matter (2.5 micrometers or smaller)' },
    { name: 'PM10', icon: <Wind className="w-5 h-5" />, description: 'Coarse particulate matter (10 micrometers or smaller)' },
    { name: 'O3', icon: <Sun className="w-5 h-5" />, description: 'Ozone' },
    { name: 'CO', icon: <Droplets className="w-5 h-5" />, description: 'Carbon monoxide' },
    { name: 'SO2', icon: <Droplets className="w-5 h-5" />, description: 'Sulfur dioxide' },
    { name: 'NO2', icon: <Droplets className="w-5 h-5" />, description: 'Nitrogen dioxide' },
  ]

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* What is AQI */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
          <Info className="w-6 h-6 mr-2 text-indigo-600" />
          What is AQI?
        </h3>
        <p className="text-gray-600 leading-relaxed mb-4">
          The Air Quality Index (AQI) is a standardized indicator for reporting air quality.
          It tells you how clean or polluted the air is and what associated health effects might
          be of concern. The AQI focuses on health effects you may experience within a few hours
          or days after breathing polluted air.
        </p>

        <h4 className="font-semibold text-gray-800 mt-6 mb-3">Pollutants Measured</h4>
        <div className="grid grid-cols-2 gap-3">
          {pollutants.map((pollutant) => (
            <div
              key={pollutant.name}
              className="flex items-start space-x-2 p-3 bg-gray-50 rounded-lg"
            >
              <div className="text-indigo-600 mt-0.5">{pollutant.icon}</div>
              <div>
                <div className="font-semibold text-gray-800">{pollutant.name}</div>
                <div className="text-xs text-gray-600">{pollutant.description}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* AQI Categories */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-gray-800 mb-4">
          AQI Categories and Health Implications
        </h3>
        <div className="space-y-3">
          {categories.map((category) => (
            <div
              key={category.name}
              className="p-4 rounded-lg border-l-4"
              style={{
                borderLeftColor: category.color,
                backgroundColor: `${category.color}15`,
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <span
                  className="font-semibold text-gray-800"
                  style={{ color: category.color }}
                >
                  {category.name}
                </span>
                <span className="text-sm font-medium text-gray-600">
                  {category.range}
                </span>
              </div>
              <p className="text-sm text-gray-600">{category.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default AQIInfo

