import { ParkingProvider } from './context/ParkingContext'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import ParkingScene from './components/ParkingScene'
import './App.css'

export default function App() {
  return (
    <ParkingProvider>
      <div className="dashboard">
        <Header />
        <div className="main-content">
          <Sidebar />
          <div id="parking-container" style={{ position: 'relative', flex: 1, height: '100%' }}>
            <ParkingScene />
          </div>
        </div>
      </div>
    </ParkingProvider>
  )
} 
