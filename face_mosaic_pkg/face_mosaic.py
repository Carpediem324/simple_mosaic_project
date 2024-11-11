import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class FaceMosaicNode(Node):
    def __init__(self):
        super().__init__('face_mosaic_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # 이미지 토픽 이름을 지정
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.face_cascade = cv2.CascadeClassifier('./face_mosaic_pkg/model/haarcascade_frontalface_default.xml')


    def image_callback(self, msg):
        # ROS 이미지 메시지를 OpenCV 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # 얼굴 탐지
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 모자이크 처리
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            # 모자이크 효과: 이미지를 작은 크기로 줄였다가 다시 키움
            face_region = cv2.resize(face_region, (w//10, h//10))
            face_region = cv2.resize(face_region, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y+h, x:x+w] = face_region

        # 화면에 이미지 표시 (디버깅 용도)
        cv2.imshow('Face Mosaic', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    face_mosaic_node = FaceMosaicNode()

    try:
        rclpy.spin(face_mosaic_node)
    except KeyboardInterrupt:
        pass

    # 노드 종료
    face_mosaic_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
