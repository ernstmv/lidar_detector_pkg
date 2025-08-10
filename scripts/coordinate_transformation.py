import copy
import numpy as np

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

class TransformationKitti:
    def __init__(self, calib_data):
        # Initialize defaults (in case some lines are missing)
        self.P2 = np.eye(3, 4)
        self.R0_rect = np.eye(3)
        self.Tr_lidar_to_cam = np.eye(4)
        self.Tr_imu_to_lidar = np.eye(4)
        
        lines = calib_data.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Corregir parsing de P2 (antes buscaba P0)
            if line.startswith('P2:'):
                nums = line.replace('P2:', '').split()
                nums = [float(x) for x in nums]
                self.P2 = np.array(nums).reshape(3, 4)
                print("P2 matrix:")
                print(self.P2)
                
            # Corregir parsing de R_rect (sin los dos puntos)
            elif line.startswith('R_rect '):  # Nota el espacio en lugar de ':'
                nums = line.replace('R_rect ', '').split()
                nums = [float(x) for x in nums]
                self.R0_rect = np.array(nums).reshape(3, 3)
                print("R_rect matrix:")
                print(self.R0_rect)
                
            # Corregir parsing de Tr_velo_cam (sin los dos puntos)
            elif line.startswith('Tr_velo_cam '):  # Nota el espacio en lugar de ':'
                nums = line.replace('Tr_velo_cam ', '').split()
                nums = [float(x) for x in nums]
                # Crear matriz 4x4 homogénea
                M = np.eye(4)
                M[0:3, :] = np.array(nums).reshape(3, 4)
                self.Tr_lidar_to_cam = M
                print("Tr_velo_cam matrix:")
                print(self.Tr_lidar_to_cam)
                
            elif line.startswith('Tr_imu_velo '):
                nums = line.replace('Tr_imu_velo ', '').split()
                nums = [float(x) for x in nums]
                M = np.eye(4)
                M[0:3, :] = np.array(nums).reshape(3, 4)
                self.Tr_imu_to_lidar = np.linalg.inv(M)

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Output: nx4 points in Homogeneous by appending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def project_3d_to_image(self, pts_3d_rect):
        """
        Project 3D points in rectified camera coordinates to image coordinates.
        Input: pts_3d_rect (Nx3) - points in rectified camera coordinates
        Output: pts_2d (Nx2) - points in image coordinates
        """
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P2))
        
        # Avoid division by zero
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def compute_box_3dto2d(self, bbox3d_input):
        """
        Takes a 3D bounding box [h, w, l, x, y, z, theta] in LiDAR coordinates
        and returns corners_2d: (8,2) array in image coords (or None if behind camera).
        """
        bbox3d = copy.copy(bbox3d_input)
        
        # Create 3D bounding box corners in object coordinate system
        R = roty(bbox3d[6])  # rotation around y-axis
        l = bbox3d[2]  # length
        w = bbox3d[1]  # width  
        h = bbox3d[0]  # height
        
        # Define the 8 corners of the 3D bounding box in object coordinates
        # Bottom face (y=0), then top face (y=-h)
        x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
        y_corners = [0,    0,    0,    0,   -h,   -h,   -h,   -h]
        z_corners = [w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
        
        # Apply rotation
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        
        # Translate to object position in LiDAR coordinates
        corners_3d[0, :] += bbox3d[3]  # x
        corners_3d[1, :] += bbox3d[4]  # y
        corners_3d[2, :] += bbox3d[5]  # z
        
        # Transform from LiDAR to camera coordinates
        corners_3d_hom = np.vstack([corners_3d, np.ones((1, 8))])
        corners_cam = np.dot(self.Tr_lidar_to_cam, corners_3d_hom)
        
        # Apply rectification transformation
        corners_rect = np.dot(self.R0_rect, corners_cam[:3, :])
        
        # Check if any corner is behind the camera
        if np.any(corners_rect[2, :] < 0.1):
            return None
        
        # Transpose to get (8,3) shape for projection
        corners_rect = corners_rect.T
        
        # Project to image coordinates
        corners_2d = self.project_3d_to_image(corners_rect)
        
        return corners_2d

    def lidar_to_camera(self, pts_lidar):
        """
        Transform points from LiDAR to camera coordinates
        Input: pts_lidar (Nx3) - points in LiDAR coordinates
        Output: pts_cam (Nx3) - points in camera coordinates
        """
        pts_lidar_hom = self.cart2hom(pts_lidar)
        pts_cam_hom = np.dot(pts_lidar_hom, self.Tr_lidar_to_cam.T)
        return pts_cam_hom[:, :3]
    
    def camera_to_rect(self, pts_cam):
        """
        Apply rectification to camera coordinates
        Input: pts_cam (Nx3) - points in camera coordinates
        Output: pts_rect (Nx3) - points in rectified camera coordinates
        """
        return np.dot(pts_cam, self.R0_rect.T)
