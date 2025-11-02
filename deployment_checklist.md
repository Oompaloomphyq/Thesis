# Deployment Checklist

## Pre-Deployment Setup

### Raspberry Pi Setup
- [x] Raspberry Pi 5 with Ubuntu 24 installed
- [x] Livox MID-360 LiDAR connected and tested
- [x] Python 3.x installed (`python3 --version`)
- [x] NumPy installed (`sudo apt-get install python3-numpy`)
- [x] scikit-learn installed (`sudo apt-get install python3-sklearn`)
- [x] Network configured (static IP recommended)
- [ ] Files copied to Pi:
  - [ ] `lidar_processor.py`
  - [ ] `run_processor.sh`
  - [ ] `config.ini`
- [ ] Script made executable (`chmod +x run_processor.sh`)

### Laptop Setup
- [x] Python 3.x installed
- [x] Required packages installed:
  ```bash
  pip install numpy matplotlib scikit-learn
  ```
- [ ] Files copied to laptop:
  - [ ] `laptop_receiver.py`
- [ ] Firewall port 5000 UDP opened (if needed)
- [ ] Network configured (same subnet as Pi)

### Network Setup
- [x] Raspberry Pi IP address noted: `192.168.1.24/24`
- [x] Laptop IP address noted: `192.168.137.1`
- [x] Both devices on same network
- [ ] Can ping Pi from laptop: `___.___.___.___`
- [x] Can ping laptop from Pi: `192.168.137.1`
- [ ] Wired Ethernet preferred over Wi-Fi

---

## Deployment Steps

### Step 1: Test Raspberry Pi Processing Standalone
```bash
# On Raspberry Pi
cd /path/to/project
python3 lidar_processor.py
```

**Expected Output:**
```
Initializing Livox MID-360...
Starting LiDAR data acquisition...
Streaming processed data to port 5000
Press Ctrl+C to stop

RANSAC: 285000 -> 1250 points
Height filter: 1250 -> 890 points
DBSCAN: Found 8 clusters
```

**Checklist:**
- [ ] LiDAR initializes successfully
- [ ] Point clouds captured
- [ ] RANSAC removes ground (~99% reduction)
- [ ] Height filter keeps reasonable points
- [ ] DBSCAN finds clusters (0-15 typically)
- [ ] Processing time < 50ms
- [ ] No errors in console

**If Issues:**
- Check LiDAR connection (USB/Ethernet)
- Verify LiDAR SDK installed
- Check permissions (`sudo` may be needed)
- Review `/mnt/project/Manuscript.pdf` Section 3.2.2

---

### Step 2: Test Network Streaming

#### Terminal 1 (Laptop):
```bash
cd /path/to/project
python3 laptop_receiver.py
```

**Expected Output:**
```
Listening for processed data on port 5000...
Press Ctrl+C to stop
```

#### Terminal 2 (Raspberry Pi):
```bash
cd /path/to/project
./run_processor.sh <LAPTOP_IP> 5000
```

**Expected Output:**
```
========================================
LiDAR Processor - Raspberry Pi
========================================

Configuration:
  Laptop IP: 192.168.1.100
  Stream Port: 5000

Starting LiDAR processor...
```

**Checklist:**
- [ ] Laptop shows "Listening..." message
- [ ] Pi shows configuration
- [ ] Laptop starts receiving frames within 5 seconds
- [ ] 3D visualization appears on laptop
- [ ] Point clouds update in real-time
- [ ] Bounding boxes drawn around clusters
- [ ] Console shows frame statistics
- [ ] Receive FPS matches processing FPS (±10%)
- [ ] No "connection refused" errors
- [ ] No packet loss warnings

**If Issues:**
- Verify laptop IP address is correct
- Check firewall (disable temporarily for testing)
- Use `sudo tcpdump -i any port 5000` to see packets
- Try ping test: `ping <laptop-ip>`
- Switch to wired Ethernet if using Wi-Fi

---

### Step 3: Verify Data Quality

**On Laptop Visualization:**
- [ ] Point cloud looks reasonable (not just noise)
- [ ] Clusters correspond to visible vehicles
- [ ] Bounding boxes are vehicle-sized
- [ ] One cluster per vehicle (mostly)
- [ ] Few/no false detections
- [ ] Visualization updates smoothly

**On Console Statistics:**
- [ ] Raw points: 200K-400K (typical for MID-360)
- [ ] Filtered points: 500-2000 (after RANSAC+filter)
- [ ] Clusters: 0-15 (depending on occupancy)
- [ ] Processing time: 30-50ms (target <50ms)
- [ ] Receive FPS: >15 (target >20)

**If Poor Quality:**
- Adjust RANSAC threshold in `config.ini`
- Tune DBSCAN eps and min_samples
- Check LiDAR mounting (should be elevated, ~4m)
- Ensure clear line of sight to parking area
- Review parameters in Section 3.2.2 of manuscript

---

### Step 4: Parameter Tuning

#### Test Different Scenarios:

1. **Empty Parking Lot**
   - [ ] System detects 0-2 clusters (minimal false positives)
   
2. **Single Vehicle**
   - [ ] System detects exactly 1 cluster
   - [ ] Bounding box matches vehicle size
   - [ ] Cluster has 200-1000 points
   
3. **Multiple Vehicles**
   - [ ] Each vehicle gets one cluster
   - [ ] Adjacent vehicles not merged
   - [ ] No splitting of single vehicles
   
4. **Vehicle Entering/Exiting**
   - [ ] Cluster appears/disappears smoothly
   - [ ] Tracking stable during movement
   - [ ] No flickering or jittering

#### Adjust Parameters If Needed:

**Too many false clusters:**
```ini
# In config.ini
[DBSCAN]
eps = 0.7              # Increase from 0.5
min_samples = 15       # Increase from 10

[CLUSTER_FILTERING]
min_cluster_points = 100  # Increase from 50
```

**Vehicles split into multiple clusters:**
```ini
# In config.ini
[DBSCAN]
eps = 0.7              # Increase from 0.5
min_samples = 8        # Decrease from 10
```

**Ground not fully removed:**
```ini
# In config.ini
[RANSAC]
threshold = 0.15       # Increase from 0.1
iterations = 150       # Increase from 100
```

---

### Step 5: Performance Optimization

**Target Metrics:**
- Processing FPS: >20
- End-to-end latency: <200ms
- Network bandwidth: <200 KB/s
- CPU usage on Pi: <70%
- RAM usage on Pi: <2 GB

**If Performance Issues:**

1. **High Processing Time (>70ms):**
   ```ini
   [RANSAC]
   iterations = 50  # Reduce from 100
   
   [DBSCAN]
   eps = 0.6  # Increase for faster clustering
   
   [PERFORMANCE]
   max_points_per_frame = 30000  # Downsample
   ```

2. **Network Lag:**
   - Use wired Ethernet (not Wi-Fi)
   - Check network congestion
   - Reduce visualization update rate
   
3. **High CPU Usage:**
   - Increase DBSCAN eps (faster clustering)
   - Enable downsampling in config
   - Reduce RANSAC iterations

---

### Step 6: Long-term Stability Test

**Run for 1 hour:**
```bash
# On Pi
./run_processor.sh <LAPTOP_IP> > test_log.txt 2>&1
```

**Monitor for:**
- [ ] No crashes or exceptions
- [ ] Consistent processing times
- [ ] Stable memory usage (no leaks)
- [ ] Reliable network streaming
- [ ] No degradation over time

**Check Logs:**
```bash
# On Pi
tail -f test_log.txt

# Look for:
# - Memory errors
# - Network timeouts
# - Processing delays
# - Exception traces
```

---

## Post-Deployment Validation

### Performance Metrics

Record baseline performance:

| Metric | Value | Date | Notes |
|--------|-------|------|-------|
| Processing Time (avg) | ___ ms | _____ | Target: <50ms |
| Processing FPS | ___ | _____ | Target: >20 |
| Network Bandwidth | ___ KB/s | _____ | Target: <200 KB/s |
| Receive FPS | ___ | _____ | Target: >20 |
| Detection Accuracy | ___% | _____ | To be measured |
| CPU Usage (Pi) | ___% | _____ | Target: <70% |
| RAM Usage (Pi) | ___ MB | _____ | Target: <2 GB |

### Accuracy Validation

Test detection accuracy:

1. **Ground Truth Collection:**
   - [ ] Manually count vehicles in parking area
   - [ ] Note their positions (slot numbers)
   - [ ] Record timestamp
   
2. **System Output:**
   - [ ] Count clusters detected
   - [ ] Note cluster positions
   - [ ] Compare with ground truth
   
3. **Calculate Metrics:**
   - [ ] True Positives: Correctly detected vehicles
   - [ ] False Positives: Clusters with no vehicle
   - [ ] False Negatives: Vehicles not detected
   - [ ] Accuracy = TP / (TP + FP + FN)

**Target Accuracy:** >90% (as per manuscript)

---

## Integration with Thesis System

### Next Components to Add:

1. **Vehicle Detection (PV-RCNN)**
   - [ ] Install PV-RCNN on laptop
   - [ ] Load pre-trained model
   - [ ] Process received clusters
   - [ ] Classify vehicle types
   
2. **Parking Slot Assignment**
   - [ ] Load CAD layout of parking area
   - [ ] Define slot boundaries
   - [ ] Map vehicle centroids to slots
   - [ ] Determine occupancy status
   
3. **Digital Twin Visualization**
   - [ ] Set up Unity or Three.js dashboard
   - [ ] Import parking lot 3D model
   - [ ] Connect to processed data stream
   - [ ] Render vehicles in real-time
   - [ ] Color-code occupied/vacant slots
   
4. **Database Integration**
   - [ ] Set up MySQL database
   - [ ] Create schema (see Figure 3.5 in manuscript)
   - [ ] Log occupancy events
   - [ ] Store vehicle tracking data
   - [ ] Generate timestamps
   
5. **User Interface**
   - [ ] Implement dashboard design (Figure 3.4)
   - [ ] Add real-time status display
   - [ ] Show available slot count
   - [ ] Provide navigation guidance

---

## Troubleshooting Reference

### Common Issues and Solutions

#### Issue: "ModuleNotFoundError: No module named 'sklearn'"
```bash
sudo apt-get install python3-sklearn
# or
pip install scikit-learn
```

#### Issue: "Permission denied" when running script
```bash
chmod +x run_processor.sh
# or run with sudo if needed
sudo python3 lidar_processor.py
```

#### Issue: "Address already in use" (port 5000)
```bash
# Find and kill process using port
sudo lsof -ti:5000 | xargs kill -9
```

#### Issue: "No LiDAR data received"
- Check USB/Ethernet connection
- Verify LiDAR power supply
- Test with Livox Viewer software
- Check device permissions
- Review SDK installation

#### Issue: "Network timeout" or "No packets received"
- Verify IP addresses
- Check firewall settings
- Test with ping
- Try different port
- Use wired connection

#### Issue: Processing time too high (>100ms)
- Reduce RANSAC iterations
- Increase DBSCAN eps
- Enable downsampling
- Check CPU usage
- Verify no other processes consuming resources

#### Issue: Too many false detections
- Increase DBSCAN min_samples
- Add cluster size filtering
- Adjust height filter range
- Increase RANSAC threshold
- Enable statistical outlier removal

---

## Pilot Testing Checklist (Chapter 3.4)

### Setup Phase:
- [ ] System deployed at CCS parking area
- [ ] LiDAR mounted at ~4m height
- [ ] Power supply secured
- [ ] Network connection stable
- [ ] Laptop positioned for monitoring

### Testing Phase:
- [ ] Baseline measurements recorded
- [ ] Multiple time periods tested (morning/afternoon)
- [ ] Various occupancy levels tested (0%, 50%, 100%)
- [ ] Entry/exit events captured
- [ ] Performance metrics logged

### Data Collection:
- [ ] Detection logs saved
- [ ] Ground truth data recorded
- [ ] Timing measurements captured
- [ ] Network statistics logged
- [ ] User feedback gathered (if applicable)

### Evaluation Metrics (Section 3.4.2):
- [ ] Precision calculated
- [ ] Recall calculated
- [ ] F1-score calculated
- [ ] System responsiveness measured
- [ ] Latency analyzed

---

## Sign-off

**Deployment Date:** ___________

**Deployed By:** ___________

**Performance Validated:** [ ] Yes [ ] No

**Ready for Pilot Testing:** [ ] Yes [ ] No

**Notes:**
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________

---

## Contact Information

**For technical support:**
- Check documentation in `/documentation/`
- Review manuscript Section 3.2 (Methodology)
- Consult QUICK_START.md for common issues

**For thesis-related questions:**
- Reference Chapter 3 of manuscript
- Review system architecture (Figure 3.2)

- Check evaluation metrics (Section 3.4)
