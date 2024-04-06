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
        self.declare_parameter('fx3', 1050.00)
        self.declare_parameter('fy3', 1050.00)
        self.declare_parameter('cx3', 960)
        self.declare_parameter('cy3', 540)
        self.declare_parameter('fx4', 1050.00)
        self.declare_parameter('fy4', 1050.00)
        self.declare_parameter('cx4', 960)
        self.declare_parameter('cy4', 540)
        self.declare_parameter('fx5', 1050.00)
        self.declare_parameter('fy5', 1050.00)
        self.declare_parameter('cx5', 960)
        self.declare_parameter('cy5', 540)
        self.declare_parameter('fx6', 1050.00)
        self.declare_parameter('fy6', 1050.00)
        self.declare_parameter('cx6', 960)
        self.declare_parameter('cy6', 540)
        
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
        self.fx3 = self.get_parameter('fx3').value
        self.fy3 = self.get_parameter('fy3').value
        self.cx3 = self.get_parameter('cx3').value
        self.cy3 = self.get_parameter('cy3').value
        self.fx4 = self.get_parameter('fx4').value
        self.fy4 = self.get_parameter('fy4').value
        self.cx4 = self.get_parameter('cx4').value
        self.cy4 = self.get_parameter('cy4').value
        self.fx5 = self.get_parameter('fx5').value
        self.fy5 = self.get_parameter('fy5').value
        self.cx5 = self.get_parameter('cx5').value
        self.cy5 = self.get_parameter('cy5').value
        self.fx6 = self.get_parameter('fx6').value
        self.fy6 = self.get_parameter('fy6').value
        self.cx6 = self.get_parameter('cx6').value
        self.cy6 = self.get_parameter('cy6').value

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
        self.bbx = self.bby = self.bbw = self.bbh = self.bbx1 = self.bby1 = self.bbw1 = self.bbh1 = self.bbx2 = self.bby2 = self.bbw2 = self.bbh2 = self.bbx3 = self.bby3 = self.bbw3 = self.bbh3 = self.bbx4 = self.bby4 = self.bbw4 = self.bbh4 = self.bbx5 = self.bby5 = self.bbw5 = self.bbh5 = self.bbx6 = self.bby6 = self.bbw6 = self.bbh6 = -1
        self.last_z = self.last_z1 = self.last_z2 = self.last_z3 = self.last_z4 = self.last_z5 = self.last_z6 = 0

        # Download the MiDaS model
        self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        self.midas.to('cuda')
        self.midas.eval()

        # Input transformation pipeline
        self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.transform = self.transforms.default_transform

        self.get_logger().info('DepthEstimatorNode is started...')

    def bbox_callback(self, msg):
        for bbox in msg.bounding_boxes:
            if bbox.class_id == "Ball":
                self.bbx = bbox.xmin
                self.bby = bbox.ymin
                self.bbw = bbox.xmax - bbox.xmin
                self.bbh = bbox.ymax - bbox.ymin
            elif bbox.class_id == "T_Pole":
                self.bbx1 = bbox.xmin
                self.bby1 = bbox.ymin
                self.bbw1 = bbox.xmax - bbox.xmin
                self.bbh1 = bbox.ymax - bbox.ymin
            elif bbox.class_id == "B_Pole":
                self.bbx2 = bbox.xmin
                self.bby2 = bbox.ymin
                self.bbw2 = bbox.xmax - bbox.xmin
                self.bbh2 = bbox.ymax - bbox.ymin
            elif bbox.class_id == "X_Cross":
                self.bbx3 = bbox.xmin
                self.bby3 = bbox.ymin
                self.bbw3 = bbox.xmax - bbox.xmin
                self.bbh3 = bbox.ymax - bbox.ymin
            elif bbox.class_id == "Corner":
                self.bbx4 = bbox.xmin
                self.bby4 = bbox.ymin
                self.bbw4 = bbox.xmax - bbox.xmin
                self.bbh4 = bbox.ymax - bbox.ymin
            elif bbox.class_id == "Robot":
                self.bbx5 = bbox.xmin
                self.bby5 = bbox.ymin
                self.bbw5 = bbox.xmax - bbox.xmin
                self.bbh5 = bbox.ymax - bbox.ymin
            elif bbox.class_id == "Goal":
                self.bbx6 = bbox.xmin
                self.bby6 = bbox.ymin
                self.bbw6 = bbox.xmax - bbox.xmin
                self.bbh6 = bbox.ymax - bbox.ymin

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

        # List to store distances for each landmark
        distances = []

        # Iterate through bounding box data
        for bbox_data in [(self.bbx, self.bby, self.bbw, self.bbh, self.fx, self.fy, self.cx, self.cy),
                          (self.bbx1, self.bby1, self.bbw1, self.bbh1, self.fx1, self.fy1, self.cx1, self.cy1),
                          (self.bbx2, self.bby2, self.bbw2, self.bbh2, self.fx2, self.fy2, self.cx2, self.cy2),
                          (self.bbx3, self.bby3, self.bbw3, self.bbh3, self.fx3, self.fy3, self.cx3, self.cy3),
                          (self.bbx4, self.bby4, self.bbw4, self.bbh4, self.fx4, self.fy4, self.cx4, self.cy4),
                          (self.bbx5, self.bby5, self.bbw5, self.bbh5, self.fx5, self.fy5, self.cx5, self.cy5),
                          (self.bbx6, self.bby6, self.bbw6, self.bbh6, self.fx6, self.fy6, self.cx6, self.cy6)]:
            x, y, w, h, fx, fy, cx, cy = bbox_data
            # Compute center point of the bounding box
            cx = (x + w) / 2
            cy = (y + h) / 2

            # Compute 3D coordinates of the center point
            depth = depth_map[int(cy), int(cx)]

            # Check if depth is valid and bounding box is detected
            if depth >= 0 and x != -1 and y != -1:
                # Use the actual distance formula (in centimeters) from depth and camera intrinsic parameters
                x_normalized = (fx + fy) / 2 / depth  # Normalization of x
                z = (
                    0.00000000001 * x_normalized**5 -
                    0.00000002 * x_normalized**4 +
                    0.00001 * x_normalized**3 -
                    0.0228 * x_normalized**2 +
                    3.4115 * x_normalized - 
                    9.5276
                )
                z *= 100   # Conversion to centimeters (1 meter = 100 centimeters)

                # Append distance to the list
                distances.append((x, y, w, h, z))

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

        # Display depth map and frame
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

        # Draw bounding boxes and distance information
        for (x, y, w, h, z) in distances:
            if x != -1 and y != -1:
                cv2.rectangle(cv_image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.putText(cv_image, f"Distance: {z:.1f} cm", (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Depth Map", depth_map)
        cv2.imshow("Frame", cv_image)
        cv2.waitKey(1)

        # Reset bounding box variables
        self.bbx, self.bby, self.bbw, self.bbh = -1, -1, -1, -1
        self.bbx1, self.bby1, self.bbw1, self.bbh1 = -1, -1, -1, -1
        self.bbx2, self.bby2, self.bbw2, self.bbh2 = -1, -1, -1, -1
        self.bbx3, self.bby3, self.bbw3, self.bbh3 = -1, -1, -1, -1
        self.bbx4, self.bby4, self.bbw4, self.bbh4 = -1, -1, -1, -1
        self.bbx5, self.bby5, self.bbw5, self.bbh5 = -1, -1, -1, -1
        self.bbx6, self.bby6, self.bbw6, self.bbh6 = -1, -1, -1, -1


def main(args=None):
    rclpy.init(args=args)
    depth_estimator = DepthEstimator()
    rclpy.spin(depth_estimator)
    depth_estimator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
