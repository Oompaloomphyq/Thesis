import { useParking } from '../context/ParkingContext'

export default function Sidebar() {
  const { slots } = useParking()

  return (
    <div style={{
      width: '300px',
      minWidth: '300px',
      height: '100vh',
      background: 'linear-gradient(180deg, #060d1f 0%, #0a1628 100%)',
      display: 'flex',
      flexDirection: 'column',
      padding: '14px 12px',
      gap: '10px',
      overflowY: 'hidden',
      boxSizing: 'border-box',
      borderRight: '2px solid #0f2a4a',
    }}>

      {/* ── LABEL ── */}
      <div style={{
        fontSize: 12, fontWeight: 800, color: '#38bdf8',
        letterSpacing: 3, textTransform: 'uppercase',
        textAlign: 'center',
        borderBottom: '1px solid #1e3a5f',
        paddingBottom: 10,
        flexShrink: 0,
      }}>
        Parking Slots
      </div>

      {/* ── 4×4 GRID ── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gridTemplateRows: 'repeat(4,  125px)',
        gap: '20px',
        flex: 1,
      }}>
        {slots.map(slot => {
          const isOccupied = slot.occupied || slot.status === 'occupied'
          const num = slot.id.replace('slot_', '')
          return (
            <div key={slot.id} style={{
              background: isOccupied
                ? 'linear-gradient(160deg, #450a0a, #1c0505)'
                : 'linear-gradient(160deg, #052e16, #021408)',
              border: `2px solid ${isOccupied ? '#dc2626' : '#16a34a'}`,
              borderRadius: 10,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 4,
              transition: 'all 0.4s ease',
              boxShadow: isOccupied
                ? '0 0 12px rgba(220,38,38,0.5), inset 0 0 8px rgba(220,38,38,0.1)'
                : '0 0 12px rgba(22,163,74,0.4), inset 0 0 8px rgba(22,163,74,0.08)',
            }}>

              {/* Slot number */}
              <div style={{
                fontSize: 16,
                fontWeight: 900,
                color: '#ffffff',
                lineHeight: 1,
              }}>
                {num}
              </div>

              {/* Glowing dot */}
              <div style={{
                width: 6, height: 6,
                borderRadius: '50%',
                background: isOccupied ? '#ef4444' : '#22c55e',
                boxShadow: isOccupied
                  ? '0 0 8px #ef4444'
                  : '0 0 8px #22c55e',
              }} />

              {/* Status */}
              <div style={{
                fontSize: 8,
                fontWeight: 800,
                color: '#ffffff',
                letterSpacing: 0.5,
                textTransform: 'uppercase',
              }}>
                {isOccupied ? 'Occupied' : 'Vacant'}
              </div>

            </div>
          )
        })}
      </div>

      {/* ── LEGEND ── */}
      <div style={{
        display: 'flex', justifyContent: 'center', gap: 20,
        borderTop: '1px solid #1e293b', paddingTop: 8,
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 12, height: 12, borderRadius: 4, background: '#22c55e', boxShadow: '0 0 8px #22c55e' }} />
          <span style={{ fontSize: 11, color: '#86efac', fontWeight: 700 }}>Vacant</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 12, height: 12, borderRadius: 4, background: '#ef4444', boxShadow: '0 0 8px #ef4444' }} />
          <span style={{ fontSize: 11, color: '#fca5a5', fontWeight: 700 }}>Occupied</span>
        </div>
      </div>

    </div>
  )
}
