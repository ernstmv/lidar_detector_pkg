from time import time
from math import sin, cos

from mmdet3d.apis import LidarDet3DInferencer

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2

from custom_msgs.msg import MarkerArrayStamped


class LidarDetectorNode(Node):
    def __init__(self):
        super().__init__('lidar_detector_node')

        # Define QoS profile
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)

        # lidar subscription
        self.declare_parameter('pointcloud_topic', '/lidar/points')
        pointcloud_topic = self.get_parameter('pointcloud_topic').value

        self.create_subscription(PointCloud2, pointcloud_topic, self._main_pipeline, qos_profile=qos_profile)

        # Declare publisher
        self.declare_parameter('bounding_box_topic', '/detected_bounding_boxes')
        bounding_box_topic = self.get_parameter('bounding_box_topic').value
        self.bbox_publisher = self.create_publisher(MarkerArrayStamped, bounding_box_topic, qos_profile)

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

    def _main_pipeline(self, lidar):

        points = self.convert_pc2_to_np(lidar)

        if points is None:
            return None

        start = time()
        results = self.inferencer(
                {'points': points},
                batch_size=1,
                show=False)
        end = time()

        detections = self.create_marker_array_from_predictions(
                results,
                lidar.header.frame_id,
                lidar.header.stamp)

        self.bbox_publisher.publish(detections)

        self.get_logger().info(f"{len(detections.markers.markers)} detections in {end-start: .4f} s")

    def create_marker_array_from_predictions(self, results, frame_id, timestamp):
        bbox_data = results['predictions'][0]['bboxes_3d']
        bbox_labels = results['predictions'][0]['labels_3d']
        bbox_scores = results['predictions'][0]['scores_3d']

        # Creamos el MarkerArray
        marker_array = MarkerArray()

        # Header con frame_id y timestamp que recibiste
        header = Header()
        header.frame_id = frame_id
        header.stamp = timestamp  # Esto es un builtin_interfaces/Time

        if len(bbox_data) == 0:
            marker_array_stamped = MarkerArrayStamped()
            marker_array_stamped.header = header
            marker_array_stamped.markers = marker_array

            return marker_array_stamped

        # Colores para etiquetas
        label_colors = {
            0: (1.0, 0.0, 0.0),  # pedestrian
            1: (1.0, 0.0, 0.0),  # cyclist
        }

        for i, (bbox, label, score) in enumerate(zip(bbox_data, bbox_labels, bbox_scores)):
            if score < self.model_threshold:
                continue

            if label not in [0, 1]:
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

        # Ahora creamos el mensaje MarkerArrayStamped con header y markers
        marker_array_stamped = MarkerArrayStamped()
        marker_array_stamped.header = header
        marker_array_stamped.markers = marker_array

        return marker_array_stamped


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
