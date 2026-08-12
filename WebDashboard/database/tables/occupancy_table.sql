CREATE TABLE IF NOT EXISTS occupancy_table (
    slot_id VARCHAR(50) PRIMARY KEY,
    slot_number INT NOT NULL,
    occupied BOOLEAN DEFAULT FALSE,
    vehicle_id VARCHAR(100),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    position_x FLOAT,
    position_z FLOAT
);