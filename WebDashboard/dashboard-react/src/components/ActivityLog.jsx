import { useParking } from '../context/ParkingContext'
export default function ActivityLog() {
  const { activityLog } = useParking()

  return (
    <div className="activity-log">
      <h3>Activity Log</h3>
      {activityLog.length === 0 && <p>No activity yet...</p>}
      {activityLog.map(entry => (
        <div key={entry.id} className="log-entry">
          <span className="log-time">{entry.time}</span>
          <span className="log-message">{entry.message}</span>
        </div>
      ))}
    </div>
  )
} 
