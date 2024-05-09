"""
Joshua Benyo Baker
A game that uses hand tracking to 
throw a ball/dart at a target

edited from: hand-tracking game
"""

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import time
# Importing the dart class from dart.py
from dart import dart

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
        score = 0
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

            #Draw the target
            cv2.line(image,(1250, 225),(1250, 525),(255,0,0),5)

            # Draw the hand landmarks & add index finger location to list
            finger_locations.append(self.identify_index_finger(image, results))
            #update the score
            cv2.putText(image, str(score), (50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color = GREEN, thickness=2)

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print("first imshow reached")
            cv2.imshow('Hand Tracking', image)

            #Draw line of where the finger has been
            for pixelCoord in finger_locations:
                if pixelCoord:
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
                        if coefficients[0] < 0:
                            coefficients[0] = coefficients[0] * -1

                        # Generate points along the fitted parabola for plotting
                        x_values = []
                        for i in range(1250):
                            x_values.append(i)
                        y_values = np.polyval(coefficients, x_values)

                        print(x_values)
                        print(y_values)
                        
                        #create the dart
                        the_dart = dart(x_values, y_values)
                        the_dart.set_points()
                        print(the_dart.locations)
                        loopcount = 0
                        while the_dart.get_x() < 1250:
                            loopcount=loopcount + 5

                            # Get the current frame
                            frame = self.video.read()[1]

                            # Convert it to an RGB image
                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            #flip the image back to normal
                            frame = cv2.flip(frame, 1)
        
                            #cv2.circle(frame, (int(x_values[loopcount + 400]), height - (int(y_values[loopcount + 400]) + pixelCoord[1])), 25, RED, 5)
                            cv2.circle(frame, (int(x_values[loopcount + 400]), (int(y_values[loopcount + 400]))), 25, RED, 5)
                            cv2.putText(frame, str(score), (50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color = GREEN, thickness=2)
                            
                            if int(x_values[loopcount + 400]) == 1240 and int(y_values[loopcount + 400]) > 225 and int(y_values[loopcount + 400]) < 525:
                                cv2.line(frame,(1250, 225),(1250, 525),(0,255,0),5)
                                score = score + 1
                                cv2.imshow('Hand Tracking', frame)
                                break
                            else:
                                cv2.line(frame,(1250, 225),(1250, 525),(0,0,255),5)
                            
                            #display the frame
                            print("second imshow reached")
                            print(loopcount)
                            cv2.imshow('Hand Tracking', frame)                          

                            if len(the_dart.x_values) > 0:
                                #move the points
                                the_dart.move_points()

                            #close the window if user presses q
                            if cv2.waitKey(50) & 0xFF == ord('q'):
                                self.video.release()
                                cv2.destroyAllWindows()
                        
                        #clear the points
                        finger_locations.clear()

                    elif pixelCoord[0] < 100:
                        finger_locations.clear()
            

            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                
        
                self.video.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":        
    g = Game()
    g.run()