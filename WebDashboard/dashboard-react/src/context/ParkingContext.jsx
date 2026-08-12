import { createContext, useContext, useState, useCallback } from 'react'

const ParkingContext = createContext()

const SLOT_POSITIONS = [
  { id: 'slot_1',  x: -8.0,  z: -2.1, width: 1.6, depth: 4.2, rotation: -Math.PI/4 },
  { id: 'slot_2',  x: -5.9,  z: -2.0, width: 1.4, depth: 4.1, rotation: -Math.PI/4 },
  { id: 'slot_3',  x: -3.9,  z: -1.9, width: 1.4, depth: 4.0, rotation: -Math.PI/4 },
  { id: 'slot_4',  x: -1.8,  z: -1.9, width: 1.4, depth: 3.9, rotation: -Math.PI/4 },
  { id: 'slot_5',  x:  0.3,  z: -1.9, width: 1.4, depth: 3.8, rotation: -Math.PI/4 },
  { id: 'slot_6',  x:  2.5,  z: -1.9, width: 1.4, depth: 3.7, rotation: -Math.PI/4 },
  { id: 'slot_7',  x:  4.6,  z: -1.9, width: 1.4, depth: 3.7, rotation: -Math.PI/4 },
  { id: 'slot_8',  x:  6.7,  z: -1.9, width: 1.4, depth: 3.7, rotation: -Math.PI/4 },
  { id: 'slot_9',  x:  8.8,  z: -1.9, width: 1.4, depth: 3.7, rotation: -Math.PI/4 },
  { id: 'slot_10', x: 11.4,  z: -2.1, width: 1.6, depth: 4.2, rotation: -Math.PI/4 },
  { id: 'slot_11', x:  5.9,  z:  2.8, width: 2.6, depth: 1.2, rotation: 0 },
  { id: 'slot_12', x:  3.1,  z:  2.8, width: 2.7, depth: 1.2, rotation: 0 },
  { id: 'slot_13', x:  0.2,  z:  2.8, width: 2.7, depth: 1.2, rotation: 0 },
  { id: 'slot_14', x: -2.7,  z:  2.8, width: 2.7, depth: 1.2, rotation: 0 },
  { id: 'slot_15', x: -5.6,  z:  2.8, width: 2.7, depth: 1.2, rotation: 0 },
  { id: 'slot_16', x: -8.4,  z:  2.8, width: 2.6, depth: 1.2, rotation: 0 },
]

const initialSlots = SLOT_POSITIONS.map(s => ({ ...s, occupied: false, status: 'available' }))

export function ParkingProvider({ children }) {
  const [slots, setSlots]                   = useState(initialSlots)
  const [movingCars, setMovingCars]         = useState([])  // [{id, scene_x, scene_z}]
  const [connectionStatus, setConnectionStatus] = useState('Connecting...')
  const [activityLog, setActivityLog]       = useState([])

  // Update a single slot occupancy
  const updateSlot = useCallback((slotId, occupied) => {
    setSlots(prev => prev.map(s =>
      s.id === slotId ? { ...s, occupied, status: occupied ? 'occupied' : 'available' } : s
    ))
    setActivityLog(prev => [{
      id:      Date.now(),
      message: `Slot ${slotId.replace('slot_', '')} is now ${occupied ? 'OCCUPIED' : 'FREE'}`,
      time:    new Date().toLocaleTimeString()
    }, ...prev.slice(0, 49)])
  }, [])

  // Replace moving cars list each frame
  const updateMovingCars = useCallback((cars) => {
    setMovingCars(cars)
  }, [])

  const totalCapacity  = slots.length
  const occupied       = slots.filter(s => s.occupied).length
  const available      = totalCapacity - occupied
  const occupancyRate  = totalCapacity > 0 ? Math.round((occupied / totalCapacity) * 100) : 0

  return (
    <ParkingContext.Provider value={{
      slots, updateSlot,
      movingCars, updateMovingCars,
      connectionStatus, setConnectionStatus,
      activityLog,
      totalCapacity, occupied, available, occupancyRate,
      SLOT_POSITIONS
    }}>
      {children}
    </ParkingContext.Provider>
  )
}

export const useParking = () => useContext(ParkingContext)