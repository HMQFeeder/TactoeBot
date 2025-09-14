import cv2
import cv2.aruco as aruco
import numpy as np
import os
import json
from datetime import datetime

class CameraCalibrator:
    def __init__(self, dictionary_type=aruco.DICT_4X4_50):
        """
        Initialize camera calibrator with ChArUco board.
        
        Args:
            dictionary_type: ArUco dictionary type to use
        """
        self.dictionary_type = dictionary_type
        self.aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
        self.aruco_params = aruco.DetectorParameters()
        
        # Calibration data
        self.obj_points = []  # 3D points in real world space
        self.img_points = []  # 2D points in image plane
        self.calibration_images = []
        
        # Camera matrix and distortion coefficients
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_error = None
        
    def detect_charuco_board(self, image, board):
        """
        Detect ChArUco board in an image.
        
        Args:
            image: Input image
            board: ChArUco board object
            
        Returns:
            tuple: (corners, ids) or (None, None) if no board detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, rejected = detector.detectMarkers(image)
        
        if len(corners) > 0:
            # Interpolate ChArUco corners
            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            if ret:
                return charuco_corners, charuco_ids
            else:
                return None, None
        else:
            return None, None
    
    def create_charuco_board(self, squares_x=7, squares_y=5, square_length=30, marker_length=23):
        """
        Create a ChArUco board.
        
        Args:
            squares_x (int): Number of squares in X direction
            squares_y (int): Number of squares in Y direction
            square_length (float): Length of square side in meters
            marker_length (float): Length of marker side in meters
            
        Returns:
            ChArUco board object
        """
        board_size = (squares_x, squares_y)
    
        board = aruco.CharucoBoard(
            board_size,
            squareLength=square_length,
            markerLength=marker_length,
            dictionary=self.aruco_dict
        )
        return board
    
    def process_calibration_image(self, image, board, draw_board=True):
        """
        Process a single calibration image with ChArUco board.
        
        Args:
            image: Input calibration image
            board: ChArUco board object
            draw_board (bool): Whether to draw detected board
            
        Returns:
            tuple: (success, processed_image, object_points, image_points)
        """
        # Detect ChArUco board
        charuco_corners, charuco_ids = self.detect_charuco_board(image, board)
        
        if charuco_ids is None or len(charuco_ids) < 8:
            print(f"Only detected {0 if charuco_ids is None else len(charuco_ids)} corners — skipping.")
            return False, image, None, None
        
        # Get object points for detected corners
        obj_points = board.getChessboardCorners()
        obj_points = obj_points[charuco_ids.flatten()]
        
        # Get image points
        img_points = charuco_corners.reshape(-1, 2)
        
        # Draw board if requested
        if draw_board:
            aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)
        
        return True, image, obj_points, img_points
    
    def add_calibration_image(self, image, board, draw_board=True):
        """
        Add a calibration image to the calibration dataset.
        
        Args:
            image: Input calibration image
            board: ChArUco board object
            draw_board (bool): Whether to draw detected board
            
        Returns:
            bool: True if image was successfully added, False otherwise
        """
        success, processed_img, obj_points, img_points = self.process_calibration_image(
            image, board, draw_board
        )
        
        if success:
            self.obj_points.append(obj_points)
            self.img_points.append(img_points)
            self.calibration_images.append(processed_img)
            print(f"Added calibration image. Total images: {len(self.calibration_images)}")
            return True
        else:
            print("Failed to add calibration image.")
            return False
    
    def calibrate_camera(self, image_size):
        """
        Perform camera calibration using collected data.
        
        Args:
            image_size (tuple): Size of images (width, height)
            
        Returns:
            bool: True if calibration successful, False otherwise
        """
        if len(self.obj_points) < 3:
            print("Need at least 3 calibration images for calibration.")
            return False
        
        print(f"Calibrating camera with {len(self.obj_points)} images...")
        print(f"Object points count: {len(self.obj_points)}")
        print(f"Image points count: {len(self.img_points)}")
        for i, pts in enumerate(self.img_points):
            print(f"Image {i} has {pts.shape[0]} points")
            print(f"Image size: {image_size}")
        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, image_size, None, None
        )
        
        if ret:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            
            # Calculate reprojection error
            total_error = 0
            for i in range(len(self.obj_points)):
                # img_points[i]: có thể shape (N,2), chuyển thành (N,1,2)
                img_pts = self.img_points[i].astype(np.float32)
                if img_pts.ndim == 2 and img_pts.shape[1] == 2:
                    img_pts = img_pts.reshape(-1,1,2)
                
                img_points_proj, _ = cv2.projectPoints(
                    self.obj_points[i], rvecs[i], tvecs[i], 
                    self.camera_matrix, self.dist_coeffs
                )
                # img_points_proj đã có shape (N,1,2)
                
                error = cv2.norm(img_pts, img_points_proj, cv2.NORM_L2) / len(img_points_proj)
                total_error += error

            
            self.calibration_error = total_error / len(self.obj_points)
            print(f"Calibration successful! Average reprojection error: {self.calibration_error:.4f}")
            return True
        else:
            print("Calibration failed.")
            return False
    
    def save_calibration(self, filename="camera_calibration.json"):
        """
        Save calibration parameters to a JSON file.
        
        Args:
            filename (str): Name of the file to save calibration data
        """
        if self.camera_matrix is None:
            print("No calibration data to save. Run calibration first.")
            return
        
        calibration_data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "calibration_error": float(self.calibration_error),
            "num_images": len(self.calibration_images),
            "dictionary_type": str(self.dictionary_type),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename="camera_calibration.json"):
        """
        Load calibration parameters from a JSON file.
        
        Args:
            filename (str): Name of the file to load calibration data from
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
            
            self.camera_matrix = np.array(calibration_data["camera_matrix"])
            self.dist_coeffs = np.array(calibration_data["dist_coeffs"])
            self.calibration_error = calibration_data["calibration_error"]
            
            print(f"Calibration loaded from {filename}")
            print(f"Calibration error: {self.calibration_error:.4f}")
            return True
        except FileNotFoundError:
            print(f"Calibration file {filename} not found.")
            return False
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def undistort_image(self, image):
        """
        Undistort an image using calibration parameters.
        
        Args:
            image: Input image to undistort
            
        Returns:
            numpy.ndarray: Undistorted image
        """
        if self.camera_matrix is None:
            print("No calibration data available. Load or perform calibration first.")
            return image
        
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        
        # Crop the image
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def get_calibration_info(self):
        """
        Get information about the current calibration.
        
        Returns:
            dict: Calibration information
        """
        if self.camera_matrix is None:
            return {"status": "No calibration data"}
        
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        return {
            "status": "Calibrated",
            "focal_length_x": fx,
            "focal_length_y": fy,
            "principal_point_x": cx,
            "principal_point_y": cy,
            "calibration_error": self.calibration_error,
            "num_images": len(self.calibration_images)
        }

def interactive_calibration():
    """
    Interactive camera calibration using ChArUco board.
    """
    print("=== Interactive Camera Calibration with ChArUco Board ===")
    print("Press 'c' to capture calibration image")
    print("Press 'q' to quit")
    print("Press 's' to start calibration")
    print("Press 'l' to load existing calibration")
    
    # Initialize camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    width, height = 1280, 1024
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Tắt auto exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)      # Giá trị exposure

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)         # Tắt auto focus
    cap.set(cv2.CAP_PROP_FOCUS, 160)           # Giá trị focus

    cap.set(cv2.CAP_PROP_GAIN, 192)            # Gain
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 220)     # Brightness

    # Một số camera không cho chỉnh sharpness qua OpenCV, nhưng nếu được:
    cap.set(cv2.CAP_PROP_SHARPNESS, 1671.4)    # Sharpness   # 0 = focus gần, tăng số = focus xa (tùy cam hỗ trợ)

    # ----- Khóa white balance -----
    cap.set(cv2.CAP_PROP_AUTO_WB, 0) 
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize calibrator and create ChArUco board
    calibrator = CameraCalibrator()
    board = calibrator.create_charuco_board(squares_x=7, squares_y=5, 
                                          square_length=0.03, marker_length=0.023)
    
    # Load existing calibration if available
    if os.path.exists("camera_calibration.json"):
        print("Found existing calibration file. Press 'l' to load it.") 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect ChArUco board in real-time
        charuco_corners, charuco_ids = calibrator.detect_charuco_board(frame, board)

        if charuco_ids is not None:
            aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
            cv2.putText(frame, f"ChArUco corners: {len(charuco_ids)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No ChArUco board detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show calibration info
        info = calibrator.get_calibration_info()
        cv2.putText(frame, f"Images: {info.get('num_images', 0)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Camera Calibration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            if calibrator.add_calibration_image(frame.copy(), board):
                print(f"Captured calibration image {len(calibrator.calibration_images)}")
        elif key == ord('s'):
            if len(calibrator.calibration_images) >= 10:
                height, width = frame.shape[:2]
                if calibrator.calibrate_camera((width, height)):
                    calibrator.save_calibration()
            else:
                print("Need at least 10 calibration images")
        elif key == ord('l'):
            calibrator.load_calibration()
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    
    """Main function to run camera calibration."""
    print("ChArUco Camera Calibration Tool")
    print("1. Interactive calibration")
    print("2. Load existing calibration")
    print("3. Test undistortion")
    print("Note: Use charucoboardmaker.py to create ChArUco boards")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        interactive_calibration()
    elif choice == "2":
        calibrator = CameraCalibrator()
        if calibrator.load_calibration():
            print("Calibration loaded successfully!")
            info = calibrator.get_calibration_info()
            print(f"Calibration error: {info['calibration_error']:.4f}")
    elif choice == "3":
        calibrator = CameraCalibrator()
        if calibrator.load_calibration():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            width, height = 1280, 1024
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Tắt auto exposure
            cap.set(cv2.CAP_PROP_EXPOSURE, -6)      # Giá trị exposure

            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)         # Tắt auto focus
            cap.set(cv2.CAP_PROP_FOCUS, 160)           # Giá trị focus

            cap.set(cv2.CAP_PROP_GAIN, 192)            # Gain
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 220)     # Brightness

            # Một số camera không cho chỉnh sharpness qua OpenCV, nhưng nếu được:
            cap.set(cv2.CAP_PROP_SHARPNESS, 1671.4)    # Sharpness   # 0 = focus gần, tăng số = focus xa (tùy cam hỗ trợ)

            # ----- Khóa white balance -----
            cap.set(cv2.CAP_PROP_AUTO_WB, 0) 
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                undistorted = calibrator.undistort_image(frame)
                cv2.imshow("Original", frame)
                cv2.imshow("Undistorted", undistorted)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
