import cv2
import cv2.aruco as aruco
import numpy as np

def create_charuco_board_a4(squares_x=7, squares_y=5, square_length_mm=30, marker_length_mm=23,
                            dictionary_type=aruco.DICT_4X4_50, output_path="charuco_board_a4.png"):
    """
    Create a ChArUco board image for A4 printing at 300 DPI.
    
    Args:
        squares_x, squares_y: Number of squares in X/Y
        square_length_mm: Square side length in mm
        marker_length_mm: Marker side length in mm
        dictionary_type: ArUco dictionary
        output_path: PNG output path
    """
    dpi = 300
    # Convert mm -> pixels: px = mm / 25.4 * DPI
    square_px = int(square_length_mm / 25.4 * dpi)
    marker_px = int(marker_length_mm / 25.4 * dpi)

    # Total board size in pixels
    board_width_px = squares_x * square_px
    board_height_px = squares_y * square_px

    # Create ArUco dictionary and Charuco board
    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    board = aruco.CharucoBoard(
        (squares_x, squares_y),
        squareLength=square_px,
        markerLength=marker_px,
        dictionary=aruco_dict
    )

    # Generate board image
    board_image = board.generateImage((board_width_px, board_height_px))

    # Create A4 background (landscape) with 1 inch margins
    a4_width_px = int(297 / 25.4 * dpi)   # 297 mm → px
    a4_height_px = int(210 / 25.4 * dpi)  # 210 mm → px
    margin_px = int(25.4 / 25.4 * dpi)    # 1 inch margin = 300 px

    a4_image = np.ones((a4_height_px, a4_width_px), dtype=np.uint8) * 255

    # Center the board on A4
    start_x = (a4_width_px - board_width_px) // 2
    start_y = (a4_height_px - board_height_px) // 2
    a4_image[start_y:start_y+board_height_px, start_x:start_x+board_width_px] = board_image

    # Optional: add border
    cv2.rectangle(a4_image, (start_x, start_y),
                  (start_x+board_width_px, start_y+board_height_px), 0, 5)

    # Save image
    cv2.imwrite(output_path, a4_image)
    print(f"Saved board to {output_path}")
    print(f"Square: {square_length_mm} mm, Marker: {marker_length_mm} mm")
    print(f"A4 image size: {a4_width_px}x{a4_height_px} px @ {dpi} DPI")

    return board

if __name__ == "__main__":
    create_charuco_board_a4()
