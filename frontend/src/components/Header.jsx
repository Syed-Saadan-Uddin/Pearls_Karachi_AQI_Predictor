import React from 'react'
import { Wind } from 'lucide-react'

function Header() {
  return (
    <header className="bg-white shadow-lg mb-8">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Wind className="w-10 h-10 text-indigo-600" />
            <div>
              <h1 className="text-3xl font-bold text-gray-800">
                Air Quality Index Dashboard
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                Real-time AQI predictions and health recommendations
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-500">
              Last updated: {new Date().toLocaleString()}
            </p>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header

