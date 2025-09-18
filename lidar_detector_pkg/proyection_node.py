import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from message_filters import Subscriber, TimeSynchronizer
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String

from cv_bridge import CvBridge

import numpy as np
from math import cos, sin

from custom_msgs.msg import MarkerArrayStamped

from lidar_detector_scripts.proyector import Proyector

class ProyectionNode(Node):

    def __init__(self):
        super().__init__('proyection_node')

        self.proyector = None
        self.bridge = CvBridge()
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)

        self.declare_parameter('calib_topic', '/camera/calibration')
        calib_topic = self.get_parameter('calib_topic').value
        self.create_subscription(String, calib_topic, self._calib_callback, 1)

        self.declare_parameter('image_input_topic', '/camera/image_raw')
        image_input_topic = self.get_parameter('image_input_topic').value

        self.declare_parameter('detections_topic', '/lidar_detector/detections')
        detections_topic = self.get_parameter('detections_topic').value

        self.ts = TimeSynchronizer([
            Subscriber(self, Image, image_input_topic, qos_profile=qos_profile),
            Subscriber(self, MarkerArrayStamped, detections_topic, qos_profile=qos_profile)
            ],
            10,
            )

        self.ts.registerCallback(self._main_pipeline)

        self.declare_parameter('image_output_topic', '/lidar_detector/image')
        image_output_topic = self.get_parameter('image_output_topic').value
        self.image_publisher = self.create_publisher(Image, image_output_topic, qos_profile)
        self.get_logger().info("Projection node ready and running...")

    def _main_pipeline(self, image_msg, detections):

        if not self.proyector:
            return

        image_cv = self._imgmsg2np(image_msg)
        points = self._markerarr2np(detections)

        output_img = self.proyector.proyect(points, image_cv)
        image_msg = self._np2imgmsg(output_img)
        
        self.image_publisher.publish(image_msg)

    def _imgmsg2np(self, img_msg: Image) -> np.ndarray:
        return self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

    def _np2imgmsg(self, arr: np.ndarray) -> Image:
        return self.bridge.cv2_to_imgmsg(arr, encoding='bgr8')

    def _calib_callback(self, msg):

        if not self.proyector:
            for line in str(msg).split(r'\n'):
                if line.startswith('std_msgs'):
                    values = line.split("'")[1].split(':')[1].split()
                    P = np.array([float(val) for val in values])
                elif line.startswith('R_rect'):
                    values = [val for val in line.split()]
                    R0 = np.array([float(val) for val in values[1:]])
                elif line.startswith('Tr_velo_cam'):
                    values = [val for val in line.split()]
                    V2C = np.array([float(val) for val in values[1:]])

            P, R0, V2C = P.reshape(3, 4), R0.reshape(3, 3), V2C.reshape(3, 4)

            self.proyector = Proyector(P, R0, V2C)

    def _markerarr2np(self, marker_array):
        """
        Convert MarkerArray with CUBE markers to corner coordinates array.
        
        Extracts 8 corners of each bounding box from MarkerArray and returns them
        as a numpy array. Each bounding box is represented by its 8 corner points
        in 3D space, taking into account position, orientation, and scale.
        
        Args:
            marker_array (visualization_msgs.msg.MarkerArray): Input marker array
            
        Returns:
            numpy.ndarray: Array of shape [n, 8, 3] where:
                          - n = number of objects (markers)
                          - 8 = 8 corners per bounding box
                          - 3 = (x, y, z) coordinates
                          
        Corner ordering (following standard convention):
            Bottom face (z_min):     Top face (z_max):
            3 -------- 2            7 -------- 6
            |          |            |          |
            |          |            |          |
            0 -------- 1            4 -------- 5
            
        Coordinate system: ROS standard (x=forward, y=left, z=up)
        """

        marker_array = marker_array.markers
        if not marker_array.markers:
            return np.array([]).reshape(0, 8, 3)
        
        corners_list = []
        
        for i, marker in enumerate(marker_array.markers):
            # Validate marker type (should be CUBE for bounding boxes)
            if marker.type != Marker.CUBE:
                continue
                
            # Extract pose information
            pos_x = marker.pose.position.x
            pos_y = marker.pose.position.y
            pos_z = marker.pose.position.z
            
            # Extract quaternion orientation
            qx = marker.pose.orientation.x
            qy = marker.pose.orientation.y
            qz = marker.pose.orientation.z
            qw = marker.pose.orientation.w
            
            # Extract scale (dimensions)
            length = marker.scale.x  # x-direction (forward)
            width = marker.scale.y   # y-direction (left)
            height = marker.scale.z  # z-direction (up)
            
            # Validate scale values
            if length <= 0 or width <= 0 or height <= 0:
                continue
            
            # Calculate 8 corners of the bounding box
            corners = self._calculate_bbox_corners(
                pos_x, pos_y, pos_z,
                qx, qy, qz, qw,
                length, width, height
            )
            
            corners_list.append(corners)
        
        if not corners_list:
            return np.array([]).reshape(0, 8, 3)
        
        # Convert to numpy array [n_objects, 8_corners, 3_coordinates]
        corners_array = np.array(corners_list)
        
        
        return corners_array


    def _calculate_bbox_corners(self, center_x, center_y, center_z, qx, qy, qz, qw, length, width, height):
        """
        Calculate 8 corners of a 3D bounding box given center, orientation, and dimensions.
        
        Args:
            center_x, center_y, center_z (float): Center position of bounding box
            qx, qy, qz, qw (float): Quaternion orientation
            length, width, height (float): Dimensions of bounding box
            
        Returns:
            numpy.ndarray: Array of shape [8, 3] containing corner coordinates
        """
        # Half dimensions
        l_2 = length / 2.0
        w_2 = width / 2.0
        h_2 = height / 2.0
        
        # Define 8 corners in local coordinate system (before rotation)
        # Origin at center, following ROS convention (x=forward, y=left, z=up)
        local_corners = np.array([
            [-l_2, -w_2, -h_2],  # 0: back-right-bottom
            [+l_2, -w_2, -h_2],  # 1: front-right-bottom
            [+l_2, +w_2, -h_2],  # 2: front-left-bottom
            [-l_2, +w_2, -h_2],  # 3: back-left-bottom
            [-l_2, -w_2, +h_2],  # 4: back-right-top
            [+l_2, -w_2, +h_2],  # 5: front-right-top
            [+l_2, +w_2, +h_2],  # 6: front-left-top
            [-l_2, +w_2, +h_2],  # 7: back-left-top
        ])
        
        # Convert quaternion to rotation matrix
        rotation_matrix = self._quaternion_to_rotation_matrix(qx, qy, qz, qw)
        
        # Apply rotation to all corners
        rotated_corners = np.dot(local_corners, rotation_matrix.T)
        
        # Translate to world coordinates
        center = np.array([center_x, center_y, center_z])
        world_corners = rotated_corners + center
        
        return world_corners


    def _quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        """
        Convert quaternion to 3x3 rotation matrix.
        
        Args:
            qx, qy, qz, qw (float): Quaternion components
            
        Returns:
            numpy.ndarray: 3x3 rotation matrix
        """
        # Normalize quaternion
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm == 0:
            return np.eye(3)
        
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # Convert to rotation matrix using standard formula
        r11 = 1 - 2*(qy*qy + qz*qz)
        r12 = 2*(qx*qy - qz*qw)
        r13 = 2*(qx*qz + qy*qw)
        
        r21 = 2*(qx*qy + qz*qw)
        r22 = 1 - 2*(qx*qx + qz*qz)
        r23 = 2*(qy*qz - qx*qw)
        
        r31 = 2*(qx*qz - qy*qw)
        r32 = 2*(qy*qz + qx*qw)
        r33 = 1 - 2*(qx*qx + qy*qy)
        
        rotation_matrix = np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])
        
        return rotation_matrix

def main(args=None):
    rclpy.init(args=args)
    node = ProyectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
