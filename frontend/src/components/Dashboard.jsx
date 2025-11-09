import React, { useState, useEffect } from 'react'
import axios from 'axios'
import ForecastCards from './ForecastCards'
import HistoricalChart from './HistoricalChart'
import PrecautionaryAdvice from './PrecautionaryAdvice'
import ForecastSummary from './ForecastSummary'
import AQIInfo from './AQIInfo'
import CurrentAQI from './CurrentAQI'
import { AlertCircle, RefreshCw } from 'lucide-react'

function Dashboard() {
  const [forecast, setForecast] = useState(null)
  const [historical, setHistorical] = useState(null)
  const [currentAQI, setCurrentAQI] = useState(null)
  const [loading, setLoading] = useState(true)
  const [currentLoading, setCurrentLoading] = useState(true)
  const [error, setError] = useState(null)
  const [currentError, setCurrentError] = useState(null)

  const fetchCurrentAQI = async () => {
    try {
      setCurrentLoading(true)
      setCurrentError(null)
      const response = await axios.get('/api/current?city=Karachi')
      setCurrentAQI(response.data)
    } catch (err) {
      setCurrentError(err.response?.data?.detail || 'Failed to load current AQI')
      console.error('Error fetching current AQI:', err)
    } finally {
      setCurrentLoading(false)
    }
  }

  const fetchData = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const [forecastRes, historicalRes] = await Promise.all([
        axios.get('/api/forecast?days=3'),
        axios.get('/api/historical?days=30')
      ])
      
      setForecast(forecastRes.data)
      setHistorical(historicalRes.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load data')
      console.error('Error fetching data:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
    fetchCurrentAQI()
    // Refresh data every 5 minutes
    const interval = setInterval(() => {
      fetchData()
      fetchCurrentAQI()
    }, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="flex justify-center items-center py-20">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-indigo-600"></div>
          <p className="mt-4 text-gray-600">Loading dashboard data...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
        <div className="flex items-center space-x-3">
          <AlertCircle className="w-6 h-6 text-red-600" />
          <div>
            <h3 className="text-red-800 font-semibold">Error loading data</h3>
            <p className="text-red-600">{error}</p>
            <button
              onClick={fetchData}
              className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center space-x-2"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Retry</span>
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Current AQI - Display above predictions */}
      <CurrentAQI currentAQI={currentAQI} loading={currentLoading} error={currentError} />

      {/* Forecast Summary */}
      {forecast && <ForecastSummary summary={forecast.summary} />}

      {/* Forecast Cards */}
      {forecast && (
        <section>
          <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
            <span className="mr-2"></span>
            3-Day AQI Forecast
          </h2>
          <ForecastCards predictions={forecast.predictions} />
        </section>
      )}

      {/* Precautionary Advice */}
      {forecast && (
        <section>
          <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
            <span className="mr-2"></span>
            Precautionary Advice
          </h2>
          <PrecautionaryAdvice predictions={forecast.predictions} />
        </section>
      )}

      {/* AQI Information */}
      <section>
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
          <span className="mr-2">ℹ️</span>
          Understanding Air Quality Index
        </h2>
        <AQIInfo />
      </section>
    </div>
  )
}

export default Dashboard

