"""
Data fusion utilities for combining multiple sensor data
"""

from collections import defaultdict
import time

class DataFusion:
    """Fuse data from multiple sources"""
    
    def __init__(self, time_window=0.5):
        self.time_window = time_window  # seconds
        self.data_buffer = defaultdict(list)
    
    def add_data(self, source, data):
        """Add data from a source"""
        timestamp = time.time()
        self.data_buffer[source].append({
            'timestamp': timestamp,
            'data': data
        })
        
        # Clean old data
        self._clean_buffer()
    
    def _clean_buffer(self):
        """Remove data older than time window"""
        current_time = time.time()
        for source in self.data_buffer:
            self.data_buffer[source] = [
                item for item in self.data_buffer[source]
                if current_time - item['timestamp'] < self.time_window
            ]
    
    def get_fused_data(self):
        """Get fused data from all sources"""
        self._clean_buffer()
        
        # Simple fusion: combine all recent data
        fused = []
        for source, items in self.data_buffer.items():
            for item in items:
                fused.append({
                    'source': source,
                    'timestamp': item['timestamp'],
                    'data': item['data']
                })
        
        return sorted(fused, key=lambda x: x['timestamp'], reverse=True)