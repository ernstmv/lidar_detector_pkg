from time import time
from math import sin, cos

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray
from mmdet3d.apis import LidarDet3DInferencer
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import QoSProfile, ReliabilityPolicy
from message_filters import Subscriber, TimeSynchronizer
from cv_bridge import CvBridge

import numpy as np
import cv2

from scripts.coordinate_transformation import TransformationKitti


def draw_3d_box(image, corners_2d, color=(0, 255, 0), thickness=2):
    """
    Dibuja una caja 3D proyectada en 2D sobre una imagen.
    corners_2d: np.array de shape (8,2)
    """
    corners_2d = corners_2d.astype(int)

    # Define las líneas que conectan los vértices de la caja 3D (según convención KITTI)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Base
        (4, 5), (5, 6), (6, 7), (7, 4),  # Techo
        (0, 4), (1, 5), (2, 6), (3, 7)   # Verticales
    ]

    for start, end in connections:
        pt1 = tuple(corners_2d[start])
        pt2 = tuple(corners_2d[end])
        cv2.line(image, pt1, pt2, color, thickness)

    return image


class LidarDetectorNode(Node):
    def __init__(self):
        super().__init__('lidar_detector_node')

        self.calib_done = False

        self.bridge = CvBridge()
        self.transformer = None

        # calib subscription
        self.declare_parameter('calib_topic', '/camera/calibration')
        calib_topic = self.get_parameter('calib_topic').value

        self.create_subscription(String, calib_topic, self._calib_callback, 1)

        # Define QoS profile
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)

        # lidar subscription
        self.declare_parameter('pointcloud_topic', '/lidar/points')
        pointcloud_topic = self.get_parameter('pointcloud_topic').value

        # camera subscription
        self.declare_parameter('image_input_topic', 'image_raw')
        image_input_topic = self.get_parameter('image_input_topic').value

        self.ts = TimeSynchronizer([
                Subscriber(self, Image, image_input_topic, qos_profile=qos_profile),
                Subscriber(self, PointCloud2, pointcloud_topic, qos_profile=qos_profile),
                ],
                10
                )
        self.ts.registerCallback(self._main_pipeline)

        # Declare publisher
        self.declare_parameter('bounding_box_topic', '/detected_bounding_boxes')
        bounding_box_topic = self.get_parameter('bounding_box_topic').value
        self.bbox_publisher = self.create_publisher(MarkerArray, bounding_box_topic, qos_profile)

        # Declare image publisher
        self.declare_parameter('image_publisher_topic', '/detected_image')
        image_publisher_topic = self.get_parameter('image_publisher_topic').value
        self.image_publisher = self.create_publisher(Image, image_publisher_topic, qos_profile)

        # Declare model parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('weights_path', '')
        self.declare_parameter('model_threshold', 0.3)
        self.declare_parameter('device', 'cpu')

        model_path = self.get_parameter('model_path').value
        weights_path = self.get_parameter('weights_path').value
        self.model_threshold = self.get_parameter('model_threshold').value
        device = self.get_parameter('device').value

        self.inferencer = LidarDet3DInferencer(
            model=model_path,
            weights=weights_path,
            device=device
        )

        self.get_logger().info("Lidar detector node running...")

    def _main_pipeline(self, image_mssg, lidar):

        if not self.calib_done:
            return None

        points = self.convert_pc2_to_np(lidar)
        image = self._imgmsg2np(image_mssg)

        if points is None:
            return None

        start = time()
        results = self.inferencer({'points': points}, batch_size=1, show=False)
        end = time()

        detections = self.create_marker_array_from_predictions(results)
        self.create_output_image(image, results)
        img_mssg = self._np2imgmsg(image)

        self.bbox_publisher.publish(detections)
        self.image_publisher.publish(img_mssg)

        self.get_logger().info(f"{len(detections.markers)} detections in {end-start: .4f} s")

    def create_output_image(self, image, results):
        bbox_data = results['predictions'][0]['bboxes_3d']

        for bbox in bbox_data:
            corners_2d = self.transformer.compute_box_3dto2d(bbox)

            if corners_2d is not None:
                image = draw_3d_box(image, corners_2d)

    def create_marker_array_from_predictions(self, results):
        bbox_data = results['predictions'][0]['bboxes_3d']
        bbox_labels = results['predictions'][0]['labels_3d']
        bbox_scores = results['predictions'][0]['scores_3d']
        marker_array = MarkerArray()

        header = Header()
        header.frame_id = "lidar_frame"
        header.stamp = self.get_clock().now().to_msg()

        label_colors = {
            0: (1.0, 0.0, 0.0),
            1: (0.0, 1.0, 0.0),
            2: (0.0, 0.0, 1.0),
        }

        for i, (bbox, label, score) in enumerate(zip(bbox_data, bbox_labels, bbox_scores)):
            if score < self.model_threshold:
                continue

            r, g, b = label_colors.get(label, (1.0, 1.0, 1.0))

            marker = Marker()
            marker.header = header
            marker.ns = "bounding_boxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = bbox[0]
            marker.pose.position.y = bbox[1]
            marker.pose.position.z = bbox[2]

            yaw = bbox[6]
            qx, qy, qz, qw = self.yaw_to_quaternion(yaw)

            marker.pose.orientation.x = qx
            marker.pose.orientation.y = qy
            marker.pose.orientation.z = qz
            marker.pose.orientation.w = qw

            marker.scale.x = bbox[3]
            marker.scale.y = bbox[4]
            marker.scale.z = bbox[5]

            marker.color.a = 0.5
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b

            marker_array.markers.append(marker)

        return marker_array

    def _np2imgmsg(self, arr: np.ndarray) -> Image:
        return self.bridge.cv2_to_imgmsg(arr, encoding='bgr8')

    def _imgmsg2np(self, img_msg: Image) -> np.ndarray:
        return self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

    def _calib_callback(self, mssg):
        if not self.calib_done:
            self.get_logger().info(mssg.data)
            self.transformer = TransformationKitti(mssg.data)
            self.calib_done = True

    def yaw_to_quaternion(self, yaw):
        return (0.0, 0.0, sin(yaw / 2.0), cos(yaw / 2.0))

    def convert_pc2_to_np(self, lidar_msg):
        return pc2.read_points_numpy(
                lidar_msg,
                field_names=("x", "y", "z", "intensity"))


def main(args=None):
    rclpy.init(args=args)
    node = LidarDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
