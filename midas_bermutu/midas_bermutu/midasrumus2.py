#=====Library=====
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from darknet_ros_msgs.msg import BoundingBoxes
import cv2
import numpy as np
import torch
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet

#=================

class DepthEstimator(Node):
    def __init__(self):
        super().__init__('depth_estimator')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create publisher for depth information
        self.depth_pub = self.create_publisher(Image, '/robot_4/landmark_distance', 10)

        # Create subscriber for image and bounding boxes
        self.image_sub = self.create_subscription(Image, '/robot_4/image_raw', self.image_callback, 10)
        self.bbox_subscription = self.create_subscription(
            BoundingBoxes,
            "/robot_4/darknet_ros/bounding_boxes",
            self.bbox_callback,
            10
        )

        # Initialize bounding box variables
        self.ball_bbox = None
        self.t_pole_bbox = None
        self.b_pole_bbox = None
	self.ga_l_cross_bbox = None
	self.pa_l_cross_bbox = None
	self.x_cross_bbox = None
	self.goal_bbox = None
	self.pinalti_bbox = None
	self.t_corner_bbox = None
        self.corner_bbox = None

        # Initialize MIDAS model
        self.midas_model = MidasNet().eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas_model.to(self.device)

        self.get_logger().info('DepthEstimatorNode is started...')

    def bbox_callback(self, msg):
        # Iterate through bounding boxes and find the one for each class
        for bbox in msg.bounding_boxes:
            if bbox.Class == "Ball":
                self.ball_bbox = bbox
            elif bbox.Class == "T_Pole":
                self.t_pole_bbox = bbox
            elif bbox.Class == "B_Pole":
                self.b_pole_bbox = bbox
	    elif bbox.Class == "GA_L_Cross":
                self.ga_l_cross_bbox = bbox
	    elif bbox.Class == "PA_L_Cross":
                self.pa_l_cross_bbox = bbox
	    elif bbox.Class == "X_Cross":
                self.x_cross_bbox = bbox
	    elif bbox.Class == "Goal":
                self.goal_bbox = bbox
	    elif bbox.Class == "Pinalti":
                self.pinalti_bbox = bbox
	    elif bbox.Class == "T_Corner":
                self.t_corner_bbox = bbox
            elif bbox.Class == "Corner":
                self.corner_bbox = bbox

    def image_callback(self, msg):
        # List to store bounding boxes
        bounding_boxes = [self.ball_bbox, self.t_pole_bbox, self.b_pole_bbox, self.ga_l_cross_bbox, self.pa_l_cross_bbox, self.x_cross_bbox, self.goal_bbox, self.pinalti_bbox, self.t_corner_bbox, self.corner_bbox]

        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Perform object detection and depth estimation for each bounding box
        for bbox in bounding_boxes:
            if bbox is not None:
                x, y, w, h = bbox.xmin, bbox.ymin, bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin
                
                # Extract region of interest (ROI) from the image
                roi = cv_image[y:y+h, x:x+w]

                # Perform depth estimation using MIDAS
                depth = self.estimate_depth(roi)

                # Publish depth as a ROS Image message
                depth_msg = Image()
                depth_msg.header = msg.header
                depth_msg.encoding = 'mono16'
                depth_msg.width = 1
                depth_msg.height = 1
                depth_msg.step = 2
                depth_msg.data = depth.tobytes()
                self.depth_pub.publish(depth_msg)

                # Display bounding box and depth on the image
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(cv_image, f"Depth: {depth:.1f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the processed image
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
