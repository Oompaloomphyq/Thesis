import { useParking } from '../context/ParkingContext'

export default function Header() {
  const { connectionStatus, totalCapacity, available, occupied, occupancyRate } = useParking()
  const isConnected = connectionStatus === 'Connected'

  const rateColor = occupancyRate > 75 ? '#ef4444' : occupancyRate > 50 ? '#f59e0b' : '#22c55e'

  return (
    <div className="header" style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 24px',
      gap: '20px',
      minHeight: '64px',
    }}>

      {/* ── TITLE ── */}
      <h1 style={{ margin: 0, whiteSpace: 'nowrap', fontSize: 22 }}>
        Smart Parking Dashboard
      </h1>

      {/* ── STATS ── */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '20px',
        flex: 1,
        justifyContent: 'center',
      }}>

        {/* TOTAL */}
        <div style={pill('#06b6d4')}>
          <span style={{ fontSize: 10, color: '#94a3b8', letterSpacing: 1 }}>TOTAL</span>
          <span style={{ fontSize: 26, fontWeight: 900, color: '#06b6d4', lineHeight: 1 }}>{totalCapacity}</span>
          <span style={{ fontSize: 10, color: '#475569' }}>Slots</span>
        </div>

        <div style={divider} />

        {/* VACANT */}
        <div style={pill('#22c55e')}>
          <span style={{ fontSize: 10, color: '#94a3b8', letterSpacing: 1 }}>VACANT</span>
          <span style={{ fontSize: 26, fontWeight: 900, color: '#22c55e', lineHeight: 1 }}>{available}</span>
          <span style={{ fontSize: 10, color: '#475569' }}>Free</span>
        </div>

        <div style={divider} />

        {/* OCCUPIED */}
        <div style={pill('#ef4444')}>
          <span style={{ fontSize: 10, color: '#94a3b8', letterSpacing: 1 }}>OCCUPIED</span>
          <span style={{ fontSize: 26, fontWeight: 900, color: '#ef4444', lineHeight: 1 }}>{occupied}</span>
          <span style={{ fontSize: 10, color: '#475569' }}>Cars</span>
        </div>

        <div style={divider} />

        {/* OCCUPANCY */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, padding: '0 10px' }}>
          <span style={{ fontSize: 10, color: '#94a3b8', letterSpacing: 1 }}>OCCUPANCY</span>
          <span style={{ fontSize: 26, fontWeight: 900, color: rateColor, lineHeight: 1 }}>{occupancyRate}%</span>
          <div style={{ width: 90, height: 6, background: '#1e293b', borderRadius: 999, overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${occupancyRate}%`,
              background: rateColor,
              borderRadius: 999,
              transition: 'width 0.6s ease',
              boxShadow: `0 0 8px ${rateColor}`,
            }} />
          </div>
        </div>

      </div>

      {/* ── CONNECTION STATUS ── */}
      <div className={`status-badge ${isConnected ? 'connected' : ''}`} style={{ whiteSpace: 'nowrap' }}>
        <div className="status-dot" />
        <span id="connection-status">{connectionStatus}</span>
      </div>

    </div>
  )
}

const pill = (glowColor) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  padding: '6px 14px',
  background: '#0f172a',
  borderRadius: 10,
  border: `1px solid ${glowColor}33`,
  boxShadow: `0 0 10px ${glowColor}22`,
  gap: 1,
  minWidth: 70,
})

const divider = {
  width: 1,
  height: 40,
  background: '#1e3a5f',
  margin: '0 10px',
}
