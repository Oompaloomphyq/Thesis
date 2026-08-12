import { useEffect } from 'react'
import { useParking } from '../context/ParkingContext'

export function useWebSocket() {
  const { updateSlot, updateMovingCars, setConnectionStatus } = useParking()

  useEffect(() => {
    let ws
    let reconnectTimer

    function getWebSocketURL() {
      const host = window.location.hostname
      const isLocal = host === 'localhost' || host === '127.0.0.1'
      if (isLocal) {
        return 'ws://localhost:5000/ws/lidar'
      } else {
        return `wss://${host.replace('3000', '5000')}/ws/lidar`
      }
    }

    function connect() {
      ws = new WebSocket(getWebSocketURL())

      ws.onopen = () => {
        setConnectionStatus('Connected')
        clearTimeout(reconnectTimer)
      }

      ws.onclose = () => {
        setConnectionStatus('Disconnected')
        reconnectTimer = setTimeout(connect, 3000)
      }

      ws.onerror = () => {
        setConnectionStatus('Error')
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          if (data.type === 'slot_update' && data.slots) {
            data.slots.forEach(slot => {
              updateSlot(slot.id, slot.occupied)
            })
          }

          if (data.moving_cars !== undefined) {
            updateMovingCars(data.moving_cars)
          }

        } catch (e) {
          console.error('WebSocket parse error:', e)
        }
      }
    }

    connect()

    return () => {
      clearTimeout(reconnectTimer)
      if (ws) ws.close()
    }
  }, [])
}