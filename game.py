"""
A game that uses hand tracking to 
hit and destroy green circle enemies.

@author: Nandhini Namasivayam
@version: March 2024

edited from: https://i-know-python.com/computer-vision-game-using-mediapipe-and-python/
"""

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import random
import time
import numpy as np
import matplotlib.pyplot as plt

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Library Constants
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkPoints = mp.solutions.hands.HandLandmark
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils

      
class Game:
    def __init__(self):
        # Create the hand detector
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=2)
        self.detector = HandLandmarker.create_from_options(options)

        # TODO: Load video
        self.video = cv2.VideoCapture(1)

    
    def draw_landmarks_on_hand(self, image, detection_result):
        """
        Draws all the landmarks on the hand
        Args:
            image (Image): Image to draw on
            detection_result (HandLandmarkerResult): HandLandmarker detection results
        """
        # Get a list of the landmarks
        hand_landmarks_list = detection_result.hand_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Save the landmarks into a NormalizedLandmarkList
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            # Draw the landmarks on the hand
            DrawingUtil.draw_landmarks(image,
                                       hand_landmarks_proto,
                                       solutions.hands.HAND_CONNECTIONS,
                                       solutions.drawing_styles.get_default_hand_landmarks_style(),
                                       solutions.drawing_styles.get_default_hand_connections_style())

    
    def identify_index_finger(self, image, detection_result):
        """
        Draws a green circle on the index finger 
        and calls a method to check if we've intercepted
        with the enemy
        Args:
            image (Image): The image to draw on
            detection_result (HandLandmarkerResult): HandLandmarker detection results
        """
        # Get image details
        imageHeight, imageWidth = image.shape[:2]

        # Get a list of the landmarks
        hand_landmarks_list = detection_result.hand_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # get coordinates of just the index finger
            finger = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_TIP.value]
            pixelCoord = DrawingUtil._normalized_to_pixel_coordinates(finger.x, finger.y, imageWidth, imageHeight)
            if pixelCoord:
                #draw the circle around the index finger
                cv2.circle(image, (pixelCoord[0], pixelCoord[1]), 25, GREEN, 5)
            return pixelCoord
            
    
    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        coefficients
        finger_locations = []
        # TODO: Modify loop condition  
        while self.video.isOpened():
            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = cv2.flip(image, 1)

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw the hand landmarks & add index finger location to list
            finger_locations.append(self.identify_index_finger(image, results))

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Hand Tracking', image)

            #Draw line of where the finger has been
            for pixelCoord in finger_locations:
                if pixelCoord:
                    #draw the circle around the index finger
                    #cv2.circle(image, (pixelCoord[0], pixelCoord[1]), 25, GREEN, 5)
                    cv2.line(image,(pixelCoord[0],pixelCoord[1]),(pixelCoord[0],pixelCoord[1]),(0,0,0),5)
                    cv2.line(image,(400, 0),(400, 800),(0,0,0),5)
                    cv2.line(image,(100, 0),(100, 800),(0,0,0),5)

                    if pixelCoord[0] > 400:
                        # Separate x and y values
                        x = []
                        y = []
                        for location in finger_locations:
                            if location:
                                x.append(location[0])
                                y.append(location[1])

                        # Fit a parabola
                        coefficients = np.polyfit(x, y, 2) 
                        if coefficients[0] > 0:
                            coefficients[0] = coefficients[0]

                        # Plot the results
                        plt.scatter(x, y, label='Data Points')
                        plt.xlabel('X')
                        plt.ylabel('Y')

                        # Generate points along the fitted parabola for plotting
                        x_values = np.linspace(min(x), max(x), 100)
                        y_values = np.polyval(coefficients, x_values)

                        plt.plot(x_values, y_values, color='red', label='Fitted Parabola')
                        plt.legend()
                        plt.title('Parabola of Best Fit')
                        plt.grid(True)
                        plt.show()

                        #clear the points
                        finger_locations.clear()
                    elif pixelCoord[0] < 100:
                        finger_locations.clear()
            

            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print(self.score)
                
        
                self.video.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":        
    g = Game()
    g.run()