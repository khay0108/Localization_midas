#=====Library=====
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from darknet_ros_msgs.msg import BoundingBoxes
import cv2
import numpy as np
import torch
from torchvision import transforms
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet

#=================

class DepthEstimator(Node):
    def __init__(self):
        super().__init__('depth_estimator')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create publisher for depth information
        self.depth_pub = self.create_publisher(Image, '/robot_2/landmark_distance', 10)

        # Create subscriber for image and bounding boxes
        self.image_sub = self.create_subscription(Image, '/robot_2/image_raw', self.image_callback, 10)
        self.bbox_subscription = self.create_subscription(
            BoundingBoxes,
            "/robot_2/darknet_ros/bounding_boxes",
            self.bbox_callback,
            10
        )

        # Initialize bounding box variables
        self.bounding_boxes = {}

        # Initialize MIDAS model
        self.midas_model = MidasNet().eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas_model.to(self.device)

        self.get_logger().info('DepthEstimatorNode is started...')

    def bbox_callback(self, msg):
        # Reset bounding boxes
        self.bounding_boxes = {}
        # Iterate through bounding boxes and find the one for each class
        for bbox in msg.bounding_boxes:
            self.bounding_boxes[bbox.Class] = bbox

    def image_callback(self, msg):
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

    # Process bounding boxes and compute distance
    for (bx, by, bw, bh, z_var, fx_var, fy_var, cx_var, cy_var) in [(self.bbx, self.bby, self.bbw, self.bbh, self.last_z, self.fx, self.fy, self.cx, self.cy),
                                                                    (self.bbx1, self.bby1, self.bbw1, self.bbh1, self.last_z1, self.fx1, self.fy1, self.cx1, self.cy1),
                                                                    (self.bbx2, self.bby2, self.bbw2, self.bbh2, self.last_z2, self.fx2, self.fy2, self.cx2, self.cy2),
                                                                    (self.bbx3, self.bby3, self.bbw3, self.bbh3, self.last_z3, self.fx3, self.fy3, self.cx3, self.cy3),
                                                                    (self.bbx4, self.bby4, self.bbw4, self.bbh4, self.last_z4, self.fx4, self.fy4, self.cx4, self.cy4),
                                                                    (self.bbx5, self.bby5, self.bbw5, self.bbh5, self.last_z5, self.fx5, self.fy5, self.cx5, self.cy5),
                                                                    (self.bbx6, self.bby6, self.bbw6, self.bbh6, self.last_z6, self.fx6, self.fy6, self.cx6, self.cy6),
                                                                    (self.bbx7, self.bby7, self.bbw7, self.bbh7, self.last_z7, self.fx7, self.fy7, self.cx7, self.cy7),
                                                                    (self.bbx8, self.bby8, self.bbw8, self.bbh8, self.last_z8, self.fx8, self.fy8, self.cx8, self.cy8),
                                                                    (self.bbx9, self.bby9, self.bbw9, self.bbh9, self.last_z9, self.fx9, self.fy9, self.cx9, self.cy9)]:
        if bx != -1 and by != -1:
            depth = depth_map[int((by + bh / 2)), int((bx + bw / 2))]
            if depth >= 0:
                # Compute distance using the formula
                distance = (self.fx * self.real_object_width) / (depth * self.pixel_size)
                distance *= 100  # Convert to centimeters

                # Publish distance as a ROS Image message
                distance_msg = Image()
                distance_msg.header = msg.header
                distance_msg.encoding = 'mono16'
                distance_msg.width = 1
                distance_msg.height = 1
                distance_msg.step = 2
                # Convert the distance value to uint16 data type
                distance_uint16 = np.uint16(distance)
                distance_msg.data = distance_uint16.tobytes()
                self.depth_pub.publish(distance_msg)

    # Display depth map and frame
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

    for (bx, by, bw, bh, z_var) in [(self.bbx, self.bby, self.bbw, self.bbh, self.last_z),
                                    (self.bbx1, self.bby1, self.bbw1, self.bbh1, self.last_z1),
                                    (self.bbx2, self.bby2, self.bbw2, self.bbh2, self.last_z2),
                                    (self.bbx3, self.bby3, self.bbw3, self.bbh3, self.last_z3),
                                    (self.bbx4, self.bby4, self.bbw4, self.bbh4, self.last_z4),
                                    (self.bbx5, self.bby5, self.bbw5, self.bbh5, self.last_z5),
                                    (self.bbx6, self.bby6, self.bbw6, self.bbh6, self.last_z6),
                                    (self.bbx7, self.bby7, self.bbw7, self.bbh7, self.last_z7),
                                    (self.bbx8, self.bby8, self.bbw8, self.bbh8, self.last_z8),
                                    (self.bbx9, self.bby9, self.bbw9, self.bbh9, self.last_z9)]:
        if bx != -1 and by != -1:
            cv2.rectangle(cv_image, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 255, 0), 2)
            cv2.putText(cv_image, f"Distance: {z_var:.1f} cm", (int(bx), int(by) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Depth Map", depth_map)
    cv2.imshow("Frame", cv_image)
    cv2.waitKey(1)

    def estimate_depth(self, roi):
        # Resize and normalize ROI for input to MIDAS model
        input_transform = transforms.Compose([
            Resize(384),
            NormalizeImage(),
            PrepareForNet()
        ])
        input_image = input_transform(roi).unsqueeze(0).to(self.device)

        # Perform depth estimation using MIDAS model
        with torch.no_grad():
            prediction = self.midas_model(input_image)

        # Extract depth from prediction
        depth = prediction.squeeze().cpu().numpy()

        return depth

def main(args=None):
    rclpy.init(args=args)
    depth_estimator = DepthEstimator()
    rclpy.spin(depth_estimator)
    depth_estimator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
