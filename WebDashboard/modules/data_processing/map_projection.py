"""
Map projection utilities for coordinate transformation
"""

import numpy as np

class CoordinateTransformer:
    """Transform between LiDAR and parking area coordinates"""
    
    def __init__(self, offset_x=0, offset_y=0, offset_z=0, rotation=0, scale=1.0):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z
        self.rotation = rotation  # degrees
        self.scale = scale
        
    def transform_point(self, x, y, z):
        """Transform a single point"""
        # Apply scale
        x_scaled = x * self.scale
        y_scaled = y * self.scale
        z_scaled = z * self.scale
        
        # Apply rotation (around Y-axis)
        if self.rotation != 0:
            angle_rad = np.radians(self.rotation)
            x_rot = x_scaled * np.cos(angle_rad) - z_scaled * np.sin(angle_rad)
            z_rot = x_scaled * np.sin(angle_rad) + z_scaled * np.cos(angle_rad)
            x_scaled, z_scaled = x_rot, z_rot
        
        # Apply offset
        x_final = x_scaled - self.offset_x
        y_final = y_scaled - self.offset_y
        z_final = z_scaled - self.offset_z
        
        return x_final, y_final, z_final
    
    def transform_points(self, points):
        """Transform multiple points"""
        return [self.transform_point(x, y, z) for x, y, z in points]