-- Smart Parking System Database Schema

CREATE DATABASE IF NOT EXISTS parking_system;
USE parking_system;

-- Parking slots table
CREATE TABLE IF NOT EXISTS occupancy_table (
    slot_id VARCHAR(50) PRIMARY KEY,
    slot_number INT NOT NULL,
    occupied BOOLEAN DEFAULT FALSE,
    vehicle_id VARCHAR(100),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    position_x FLOAT,
    position_z FLOAT,
    INDEX idx_slot_number (slot_number),
    INDEX idx_occupied (occupied)
);

-- Parking event logs table
CREATE TABLE IF NOT EXISTS logs_table (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    slot_id VARCHAR(50),
    vehicle_id VARCHAR(100),
    event_type ENUM('arrival', 'departure', 'moving', 'parked') NOT NULL,
    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    coordinates JSON,
    INDEX idx_slot (slot_id),
    INDEX idx_vehicle (vehicle_id),
    INDEX idx_time (event_time),
    FOREIGN KEY (slot_id) REFERENCES occupancy_table(slot_id)
);

-- Insert initial parking slots (12 slots example)
INSERT INTO occupancy_table (slot_id, slot_number, occupied, position_x, position_z) VALUES
('slot_1', 1, FALSE, -4.5, -7.5),
('slot_2', 2, FALSE, -1.5, -7.5),
('slot_3', 3, FALSE, 1.5, -7.5),
('slot_4', 4, FALSE, 4.5, -7.5),
('slot_5', 5, FALSE, -4.5, 0),
('slot_6', 6, FALSE, -1.5, 0),
('slot_7', 7, FALSE, 1.5, 0),
('slot_8', 8, FALSE, 4.5, 0),
('slot_9', 9, FALSE, -4.5, 7.5),
('slot_10', 10, FALSE, -1.5, 7.5),
('slot_11', 11, FALSE, 1.5, 7.5),
('slot_12', 12, FALSE, 4.5, 7.5);