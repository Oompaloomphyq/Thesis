"""
db_api.py - Smart Parking System Database API
Connects React Dashboard ↔ MySQL (phpMyAdmin)
Place this inside: dashboard-react/backend/db_api.py
"""

import mysql.connector
from mysql.connector import Error
from datetime import datetime

# ============================================
# DATABASE CONFIG - match your phpMyAdmin setup
# ============================================
DB_CONFIG = {
    'host':     'localhost',
    'port':     3306,
    'user':     'root',        # your phpMyAdmin username
    'password': '',            # your phpMyAdmin password
    'database': 'smart_parking'
}

# ============================================
# CONNECTION
# ============================================
def get_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"[DB ERROR] Connection failed: {e}")
        return None


# ============================================
# SLOT FUNCTIONS
# ============================================

def get_all_slots():
    """Get current status of all 12 slots."""
    conn = get_connection()
    if not conn: return []
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM current_slot_status")
    result = cursor.fetchall()
    conn.close()
    return result


def get_slot(slot_id):
    """Get one slot by id."""
    conn = get_connection()
    if not conn: return None
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM slots WHERE slot_id = %s", (slot_id,))
    result = cursor.fetchone()
    conn.close()
    return result


def update_slot_status(slot_id, status):
    """Update slot to occupied or vacant."""
    conn = get_connection()
    if not conn: return False
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE slots SET status = %s WHERE slot_id = %s",
        (status, slot_id)
    )
    conn.commit()
    conn.close()
    return True


# ============================================
# VEHICLE FUNCTIONS
# ============================================

def create_vehicle():
    """Register a new vehicle, returns vehicle_id."""
    conn = get_connection()
    if not conn: return None
    cursor = conn.cursor()
    cursor.execute("INSERT INTO vehicles (first_seen) VALUES (NOW())")
    conn.commit()
    vehicle_id = cursor.lastrowid
    conn.close()
    return vehicle_id


# ============================================
# PARKING EVENT FUNCTIONS
# ============================================

def car_arrived(slot_id):
    """
    Called when LiDAR detects a car in a slot.
    1. Creates new vehicle
    2. Logs parking event (time_in)
    3. Updates slot to occupied
    Returns event_id
    """
    conn = get_connection()
    if not conn: return None

    cursor = conn.cursor()

    # 1. Create new vehicle
    cursor.execute("INSERT INTO vehicles (first_seen) VALUES (NOW())")
    vehicle_id = cursor.lastrowid

    # 2. Log parking event
    cursor.execute("""
        INSERT INTO parking_events (vehicle_id, slot_id, status, time_in)
        VALUES (%s, %s, 'occupied', NOW())
    """, (vehicle_id, slot_id))
    event_id = cursor.lastrowid

    # 3. Update slot status
    cursor.execute(
        "UPDATE slots SET status = 'occupied' WHERE slot_id = %s",
        (slot_id,)
    )

    conn.commit()
    conn.close()
    print(f"[DB] Car arrived → slot: {slot_id}, vehicle_id: {vehicle_id}")
    return event_id


def car_left(slot_id):
    """
    Called when LiDAR detects car has left a slot.
    1. Finds the open parking event
    2. Sets time_out and calculates duration
    3. Updates slot to vacant
    """
    conn = get_connection()
    if not conn: return False

    cursor = conn.cursor(dictionary=True)

    # 1. Find open event for this slot
    cursor.execute("""
        SELECT event_id, time_in FROM parking_events
        WHERE slot_id = %s AND status = 'occupied' AND time_out IS NULL
        ORDER BY time_in DESC LIMIT 1
    """, (slot_id,))
    event = cursor.fetchone()

    if not event:
        print(f"[DB] No open event found for slot: {slot_id}")
        conn.close()
        return False

    # 2. Calculate duration
    time_out = datetime.now()
    time_in  = event['time_in']
    duration = int((time_out - time_in).total_seconds() / 60)

    cursor2 = conn.cursor()
    cursor2.execute("""
        UPDATE parking_events
        SET status = 'vacant', time_out = %s, duration_mins = %s
        WHERE event_id = %s
    """, (time_out, duration, event['event_id']))

    # 3. Update slot status
    cursor2.execute(
        "UPDATE slots SET status = 'vacant' WHERE slot_id = %s",
        (slot_id,)
    )

    conn.commit()
    conn.close()
    print(f"[DB] Car left → slot: {slot_id}, duration: {duration} mins")
    return True


# ============================================
# LIDAR DETECTION FUNCTIONS
# ============================================

def log_lidar_detection(x, y, z, slot_id, length=None, width=None, height=None, confidence=None):
    """Log raw LiDAR bounding box detection."""
    conn = get_connection()
    if not conn: return False
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO lidar_detections 
        (slot_id, x, y, z, length, width, height, confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (slot_id, x, y, z, length, width, height, confidence))
    conn.commit()
    conn.close()
    return True


def map_xyz_to_slot(x, z):
    """
    Given a LiDAR x,z coordinate, find which slot it belongs to.
    Returns slot_id or None if no match.
    """
    conn = get_connection()
    if not conn: return None
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT slot_id FROM slots
        WHERE %s BETWEEN x_min AND x_max
        AND   %s BETWEEN z_min AND z_max
        LIMIT 1
    """, (x, z))
    result = cursor.fetchone()
    conn.close()
    return result['slot_id'] if result else None


# ============================================
# SUMMARY / DASHBOARD FUNCTIONS
# ============================================

def get_parking_summary():
    """Returns total, occupied, vacant, occupancy rate."""
    conn = get_connection()
    if not conn: return {}
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM parking_summary")
    result = cursor.fetchone()
    conn.close()
    return result


def get_recent_events(limit=20):
    """Get recent parking activity for Activity Log."""
    conn = get_connection()
    if not conn: return []
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT 
            pe.event_id,
            pe.vehicle_id,
            pe.slot_id,
            pe.status,
            pe.time_in,
            pe.time_out,
            pe.duration_mins
        FROM parking_events pe
        ORDER BY pe.created_at DESC
        LIMIT %s
    """, (limit,))
    result = cursor.fetchall()
    conn.close()
    return result


# ============================================
# TEST - run this file directly to test DB
# ============================================
if __name__ == '__main__':
    print("Testing DB connection...")
    summary = get_parking_summary()
    print("Summary:", summary)

    print("\nAll slots:")
    slots = get_all_slots()
    for s in slots:
        print(s)

    print("\nTesting car_arrived on slot_1...")
    event_id = car_arrived('slot_1')
    print(f"Event ID: {event_id}")

    print("\nTesting car_left on slot_1...")
    car_left('slot_1')

    print("\nDone! Check phpMyAdmin to verify.")
