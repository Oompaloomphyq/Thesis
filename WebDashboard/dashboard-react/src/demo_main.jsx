import React from 'react'
import ReactDOM from 'react-dom/client'
import { ParkingProvider } from './context/ParkingContext'
import ParkingScene from './components/ParkingScene'
import './App.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ParkingProvider>
      <div style={{ width: '100vw', height: '100vh', background: '#1a1a2e' }}>
        <ParkingScene />
      </div>
    </ParkingProvider>
  </React.StrictMode>
)
