CREATE TABLE IF NOT EXISTS logs_table (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    slot_id VARCHAR(50),
    vehicle_id VARCHAR(100),
    event_type ENUM('arrival', 'departure', 'moving', 'parked') NOT NULL,
    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    coordinates JSON
);