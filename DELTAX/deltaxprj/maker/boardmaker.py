import cv2
import cv2.aruco as aruco
import numpy as np
import os

def draw_board_axes_full(board_image, mm_to_pixel_x, mm_to_pixel_y, a4_width, a4_height):
    """
    Vẽ trục tọa độ XY với mũi tên 1 đầu (hướng dương), kéo dài hết tờ giấy,
    có chia vạch cm và số cho cả dương và âm.
    """
    height_pixels, width_pixels = board_image.shape[:2]

    # Tâm board (pixel)
    center_x = int((a4_width / 2) * mm_to_pixel_x)
    center_y = int((a4_height / 2) * mm_to_pixel_y)

    # ===== Trục X ===== (mũi tên sang phải)
    cv2.line(board_image, (0, center_y), (width_pixels-1, center_y), (0, 0, 255), 2)
    cv2.arrowedLine(board_image, (center_x, center_y), (width_pixels-1, center_y),
                    (0, 0, 255), 2, tipLength=0.015)

    # ===== Trục Y ===== (mũi tên lên trên)
    cv2.line(board_image, (center_x, height_pixels-1), (center_x, 0), (0, 255, 0), 2)
    cv2.arrowedLine(board_image, (center_x, center_y), (center_x, 0),
                    (0, 255, 0), 2, tipLength=0.015)

    # Vẽ tâm
    cv2.circle(board_image, (center_x, center_y), 5, (0, 0, 0), -1)

    # ===== Vạch chia cm =====
    cm_step_x = int(10 * mm_to_pixel_x)  # 1 cm theo pixel (X)
    cm_step_y = int(10 * mm_to_pixel_y)  # 1 cm theo pixel (Y)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1

    # --- Trục X ---
    for i in range(0, width_pixels//2, cm_step_x):
        # Dương X
        if center_x + i < width_pixels:
            cv2.line(board_image, (center_x + i, center_y - 4),
                     (center_x + i, center_y + 4), (0, 0, 255), 1)
            if i > 0:
                cv2.putText(board_image, f"{i/cm_step_x:.0f}", (center_x + i - 5, center_y - 8),
                            font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

        # Âm X
        if center_x - i >= 0:
            cv2.line(board_image, (center_x - i, center_y - 4),
                     (center_x - i, center_y + 4), (0, 0, 255), 1)
            if i > 0:
                cv2.putText(board_image, f"-{i/cm_step_x:.0f}", (center_x - i - 10, center_y - 8),
                            font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    # --- Trục Y ---
    for j in range(0, height_pixels//2, cm_step_y):
        # Dương Y
        if center_y - j >= 0:
            cv2.line(board_image, (center_x - 4, center_y - j),
                     (center_x + 4, center_y - j), (0, 255, 0), 1)
            if j > 0:
                cv2.putText(board_image, f"{j/cm_step_y:.0f}", (center_x + 6, center_y - j + 4),
                            font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        # Âm Y
        if center_y + j < height_pixels:
            cv2.line(board_image, (center_x - 4, center_y + j),
                     (center_x + 4, center_y + j), (0, 255, 0), 1)
            if j > 0:
                cv2.putText(board_image, f"-{j/cm_step_y:.0f}", (center_x + 6, center_y + j + 4),
                            font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    return board_image




def create_custom_aruco_board(marker_size=0.04, marker_ids=[0, 8, 16, 32, 40],
                             dictionary_type=cv2.aruco.DICT_4X4_50, board_name="aruco_board"):
    """
    Create a custom ArUco board with 4 markers positioned at the 4 corners of an A4 paper.
    
    Args:
        marker_size (float): Size of each marker in meters
        marker_ids (list): List of marker IDs to use (should have 4 elements)
        dictionary_type: ArUco dictionary type
        board_name (str): Name for the output files
    
    Returns:
        tuple: (board_image, board_object)
    """
    
    # A4 paper dimensions in meters (210mm x 297mm)
    a4_width = 297 
    a4_height = 210 #mm
    
    # Calculate marker positions (corners of A4)
    # Top-left, top-right, bottom-left, bottom-right
    offset = 20
    marker_positions = [
        (offset, offset),  # Top-left
        (a4_width - marker_size - offset, offset),  # Top-right
        (offset, a4_height - marker_size - offset),  # Bottom-left
        (a4_width - marker_size - offset, a4_height - marker_size - offset)  # Bottom-right
    ]
    
    # Get the ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    
    # Create A4-sized white background (in pixels)
    # A4 ratio is 1:1.414 (width:height)
    # Using 800 pixels width, height will be 1131 pixels
    height_pixels = 2481  # fixed height
    width_pixels = int(height_pixels * (a4_width / a4_height))  # adjust width for A4 landscape ratio

    
    # Calculate pixel positions for markers
    # Convert meter positions to pixel positions
    mm_to_pixel_x = width_pixels / a4_width
    mm_to_pixel_y = height_pixels / a4_height
    
    #Calculate markers size in pixel
    marker_pixel_sizex = marker_size * mm_to_pixel_x  
    marker_pixel_sizey = marker_size * mm_to_pixel_y 
    marker_pixel_size = int(min(marker_pixel_sizex, marker_pixel_sizey)) #size of the marker in pixels

    # Create individual markers
    markers = []
    for i, (marker_id, position) in enumerate(zip(marker_ids, marker_positions)):
        marker = aruco_dict.generateImageMarker(marker_id, marker_pixel_size)
        markers.append((marker, position, marker_id))
    
    # Create white background
    board_image = np.ones((height_pixels, width_pixels, 3), dtype=np.uint8) * 255
    
    # Place markers on the board
    for marker_img, (x_m, y_m), marker_id in markers:
        # Convert meter position to pixel position
        x_pixel = int(x_m * mm_to_pixel_x)
        y_pixel = int(y_m * mm_to_pixel_y)
        
        # Ensure marker fits within bounds
        if x_pixel + marker_pixel_size > width_pixels:
            x_pixel = width_pixels - marker_pixel_size
        if y_pixel + marker_pixel_size > height_pixels:
            y_pixel = height_pixels - marker_pixel_size
        
        # Convert grayscale marker to RGB
        marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB)
        
        # Place marker on board
        board_image[y_pixel:y_pixel+marker_pixel_size, 
                   x_pixel:x_pixel+marker_pixel_size] = marker_rgb
        
    #board_image = draw_board_axes_full(board_image, mm_to_pixel_x, mm_to_pixel_y, a4_width, a4_height)

    return board_image

def save_board(board_image, board_name="custom_board", output_dir="."):
    """
    Save the board image to a file.
    
    Args:
        board_image: The board image to save
        board_name (str): Name for the output file
        output_dir (str): Directory to save the file
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the board image
    output_path = os.path.join(output_dir, f"{board_name}.png")
    cv2.imwrite(output_path, board_image)
    print(f"Board saved to: {output_path}")
    
    return output_path

def create_custom_board_with_4_markers(marker_size=40, marker_ids=[0, 1, 2, 3],
                                      dictionary_type=cv2.aruco.DICT_4X4_50, 
                                      board_name="custom_4_marker_board",
                                      output_dir="."):
    """
    Create and save a custom ArUco board with 4 markers at A4 corners.
    
    Args:
        marker_size (float): Size of each marker in meters
        marker_ids (list): List of 4 marker IDs to use
        dictionary_type: ArUco dictionary type
        board_name (str): Name for the output files
        output_dir (str): Directory to save the file
    
    Returns:
        str: Path to the saved board image
    """
    
    print(f"Creating custom ArUco board with 4 markers at A4 corners...")
    print(f"Marker IDs: {marker_ids}")
    print(f"Marker size: {marker_size}mm")
    
    # Create the board
    board_image = create_custom_aruco_board(
        marker_size=marker_size,
        marker_ids=marker_ids,
        dictionary_type=dictionary_type,
        board_name=board_name
    )
    
    # Save the board
    output_path = save_board(board_image, board_name, output_dir)
    
    # Print board information
    print(f"\nBoard Information:")
    print(f"- Layout: 4 markers at A4 paper corners")
    print(f"- Marker IDs: {marker_ids}")
    print(f"- Marker size: {marker_size}m")
    print(f"- A4 landscape dimensions: 297mm x 210mm")
    print(f"- Dictionary: {dictionary_type}")
    
    return output_path

def main():
    """Main function to run the custom ArUco board generator."""
    
    print("Creating custom 4-marker board with specific IDs...")
    create_custom_board_with_4_markers(
        marker_ids=[0, 8, 16, 32],
        marker_size=40,
        dictionary_type=cv2.aruco.DICT_4X4_50,
        board_name="custom_ids_board1",
        output_dir="custom_boards"
    )
    

if __name__ == "__main__":
    main()