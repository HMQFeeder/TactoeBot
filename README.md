DeltaX Caro Robot Project

This project demonstrates how to use computer vision, calibration, and robotic control to let a DeltaX robot play Caro (Gomoku). The workflow includes calibrating a USB camera, calculating transformation matrices, and controlling the robot to execute moves based on detected board states.

##Workflow overview:

1.Generate calibration boards

2.Calibrate the camera

3.Compute transformation matrices (Board to Robot, Camera to Robot)

4.Initialize the system

5.Configure gameplay settings

6.Run the game with the DeltaX robot

##Step by step instructions:

1.Generate marker boards

  Go to the maker folder inside deltaxprj. Run the two Python scripts to create a ChArUco board and a custom A4 ArUco board.
	
2.Calibrate the camera

  In the deltaxprj folder, run calibration.py. This script calibrates the USB camera and produces the camera matrix.
	
3.Compute transformation matrices

  First, run CalBoardtoRobot.py and use manually collected data to calculate the Board to Robot matrix. Then, keeping the calibration board in the same position, run CalCamtoRobot.py to    compute the Camera to Robot matrix.
	
4.Initialize the system

  Run innit.py. It will start Acamera.py first. After pressing Enter, it will then launch Arobot.py.
	
5.Configure gameplay

  In the terminal, you can adjust difficulty level, game mode, and other options as needed.
	
6.Start the game

  After Acamera.py generates a JSON file containing the Board to Robot matrix, return to the innit.py terminal and press Enter. The robot is now ready. After you make a move, press F and   the DeltaX robot will calculate and execute its move.

  !!!Note: remember to create your env file with your roboflow api key in it
