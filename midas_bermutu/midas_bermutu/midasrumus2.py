import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from darknet_ros_msgs.msg import BoundingBoxes
import torch
import numpy as np

class DepthEstimator(Node):
    def __init__(self):
        super().__init__('depth_estimator')

        # Declare and get parameters
        self.declare_parameter('fx', 1050.00)
        self.declare_parameter('fy', 1050.00)
        self.declare_parameter('cx', 960)
        self.declare_parameter('cy', 540)
        self.declare_parameter('fx1', 1050.00)
        self.declare_parameter('fy1', 1050.00)
        self.declare_parameter('cx1', 960)
        self.declare_parameter('cy1', 540)
        self.declare_parameter('fx2', 1050.00)
        self.declare_parameter('fy2', 1050.00)
        self.declare_parameter('cx2', 960)
        self.declare_parameter('cy2', 540)
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.fx1 = self.get_parameter('fx1').value
        self.fy1 = self.get_parameter('fy1').value
        self.cx1 = self.get_parameter('cx1').value
        self.cy1 = self.get_parameter('cy1').value
        self.fx2 = self.get_parameter('fx2').value
        self.fy2 = self.get_parameter('fy2').value
        self.cx2 = self.get_parameter('cx2').value
        self.cy2 = self.get_parameter('cy2').value
        # self.fx = self.get_parameter('fx').value
        # self.fy = self.get_parameter('fy').value
        # self.cx = self.get_parameter('cx').value
        # self.cy = self.get_parameter('cy').value

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create publisher for depth information
        self.depth_pub = self.create_publisher(Image, '/robot_2/landmark_distance', 10)

        # Create subscribers for image and bounding boxes
        self.image_sub = self.create_subscription(Image, '/robot_2/image_raw', self.callback, 10)
        self.bbox_subscription = self.create_subscription(
            BoundingBoxes,
            "/robot_2/darknet_ros/bounding_boxes",
            self.bbox_callback,
            10
        )

        # Initialize bounding box variables
        self.bbx = self.bby = self.bbw = self.bbh = self.bbx1 = self.bby1 = self.bbw1 = self.bbh1 = self.bbx2 = self.bby2 = self.bbw2 = self.bbh2 = -1
        self.last_z = self.last_z1 = self.last_z2 = 0

        # Download the MiDaS model
        self.midas = torch.hub.load('intel-isl/MiDaS', 'DPT_hybrid')
        self.midas.to('cuda')
        self.midas.eval()

        # Input transformation pipeline
        self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.transform = self.transforms.default_transform

        self.get_logger().info('DepthEstimatorNode is started...')

    def bbox_callback(self, msg):
        if len(msg.bounding_boxes) > 0:
            bbox = msg.bounding_boxes[2]
            if bbox.class_id == "Ball":
                self.bbx = bbox.xmin
                self.bby = bbox.ymin
                self.bbw = bbox.xmax - bbox.xmin
                self.bbh = bbox.ymax - bbox.ymin
            if bbox.class_id == "T_Pole":
                self.bbx1 = bbox.xmin
                self.bby1 = bbox.ymin
                self.bbw1 = bbox.xmax - bbox.xmin
                self.bbh1 = bbox.ymax - bbox.ymin
            if bbox.class_id == "B_Pole":
                self.bbx2 = bbox.xmin
                self.bby2 = bbox.ymin
                self.bbw2 = bbox.xmax - bbox.xmin
                self.bbh2 = bbox.ymax - bbox.ymin
            # Process the bounding box data here

    def callback(self, msg):
        # Convert ROS image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Transform input for MiDaS
        imgbatch = self.transform(cv_image).to('cuda')

        # Make a prediction
        with torch.no_grad():
            prediction = self.midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=cv_image.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()

            # Get depth map
            depth_map = prediction.cpu().numpy()
            # depth_map1 = prediction.cpu().numpy()
            # depth_map2 = prediction.cpu().numpy()
        # Bounding box data
        x, y, w, h, x1, y1, w1, h1, x2, y2, w2, h2 = self.bbx, self.bby, self.bbw, self.bbh, self.bbx1, self.bby1, self.bbw1, self.bbh1, self.bbx2, self.bby2, self.bbw2, self.bbh2

        # Compute center point of the bounding box
        cx = (x + w) / 2
        cy = (y + h) / 2
        cx1 = (x1 + w1) / 2
        cy1 = (y1 + h1) / 2
        cx2 = (x2 + w2) / 2
        cy2 = (y2 + h2) / 2
        z = self.last_z
        z1 = self.last_z1
        z2 = self.last_z2
        # Compute 3D coordinates of the center point
        depth = depth_map[int(cy), int(cx)]
        depth1 = depth_map[int(cy1), int(cx1)]
        depth2 = depth_map[int(cy2), int(cx2)]
        if depth >= 0 and self.bbx != -1 and self.bby != -1:
            # Use the actual distance formula (in centimeters) from depth and camera intrinsic parameters
            x_normalized = (self.fx + self.fy) / 2 / depth  # Normalization of x
            z = (
                0.00000000001 * x_normalized**5 -
                0.00000002 * x_normalized**4 +
                0.00001 * x_normalized**3 -
                0.0228 * x_normalized**2 +
                3.4115 * x_normalized - 
                9.5276
            )
            z *= 10   # Conversion to centimeters (1 meter = 100 centimeters)

            # Publish distance as a ROS Image message
            distance_msg = Image()
            distance_msg.header = msg.header
            distance_msg.encoding = 'mono16'
            distance_msg.width = 1
            distance_msg.height = 1
            distance_msg.step = 2
            # Convert the z value to uint16 data type
            z_uint16 = np.uint16(z)
            distance_msg.data = z_uint16.tobytes()
            self.depth_pub.publish(distance_msg)
        elif depth >= 0 and self.bbx1 != -1 and self.bby1 != -1:
            # Use the actual distance formula (in centimeters) from depth and camera intrinsic parameters
            x_normalized1 = (self.fx1 + self.fy1) / 2 / depth1  # Normalization of x
            z1 = (
                0.00000000001 * x_normalized1**5 -
                0.00000002 * x_normalized1**4 +
                0.00001 * x_normalized1**3 -
                0.0228 * x_normalized1**2 +
                3.4115 * x_normalized1 - 
                9.5276
            )
            z1 *= 10   # Conversion to centimeters (1 meter = 100 centimeters)

            # Publish distance as a ROS Image message
            distance_msg = Image()
            distance_msg.header = msg.header
            distance_msg.encoding = 'mono16'
            distance_msg.width = 1
            distance_msg.height = 1
            distance_msg.step = 2
            # Convert the z value to uint16 data type
            z1_uint16 = np.uint16(z1)
            distance_msg.data = z1_uint16.tobytes()
            self.depth_pub.publish(distance_msg)
        elif depth >= 0 and self.bbx2 != -1 and self.bby2 != -1:
            # Use the actual distance formula (in centimeters) from depth and camera intrinsic parameters
            x_normalized = (self.fx2 + self.fy2) / 2 / depth2  # Normalization of x
            z2 = (
                0.00000000001 * x_normalized**5 -
                0.00000002 * x_normalized**4 +
                0.00001 * x_normalized**3 -
                0.0228 * x_normalized**2 +
                3.4115 * x_normalized - 
                9.5276
            )
            z2 *= 10   # Conversion to centimeters (1 meter = 100 centimeters)

            # Publish distance as a ROS Image message
            distance_msg = Image()
            distance_msg.header = msg.header
            distance_msg.encoding = 'mono16'
            distance_msg.width = 1
            distance_msg.height = 1
            distance_msg.step = 2
            # Convert the z value to uint16 data type
            z2_uint16 = np.uint16(z2)
            distance_msg.data = z2_uint16.tobytes()
            self.depth_pub.publish(distance_msg)
        # Display depth map and frame
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        # depth_map1 = cv2.normalize(depth_map1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # depth_map1 = cv2.applyColorMap(depth_map1, cv2.COLORMAP_JET)
        # depth_map2 = cv2.normalize(depth_map2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # depth_map2 = cv2.applyColorMap(depth_map2, cv2.COLORMAP_JET)
        cv2.rectangle(cv_image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(cv_image, f"Distance: {z:.1f} cm", (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Depth Map", depth_map)
        cv2.imshow("Frame", cv_image)

        cv2.waitKey(1)
        self.bbx, self.bby, self.bbw, self.bbh = -1, -1, -1, -1

def main(args=None):
    rclpy.init(args=args)
    depth_estimator = DepthEstimator()
    rclpy.spin(depth_estimator)
    depth_estimator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

