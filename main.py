import math
import cv2
import inputfunction
import mediapipe as mp

# Initialize Mediapipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh  # For face tracking

# Colors (BGR format)
REMOTE_COLOR_RED = (0, 0, 255)  # Red color for remote
REMOTE_COLOR_BLUE = (255, 0, 0)  # Blue color for remote

font = cv2.FONT_HERSHEY_SIMPLEX


def relall():
    # Release all control keys
    keys_to_release = ['d', 'w', 'q', 's', 'a', 'z', 'x', 'c', 'left', 'right', 'e']
    for key in keys_to_release:
        inputfunction.release_key(key)


def main():
    # Open the webcam (ensure the correct index; usually 0 or 1)
    cap = cv2.VideoCapture(1)  # Changed to 0; adjust if necessary

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    with mp_hands.Hands(
            model_complexity=1,  # Increased for better accuracy
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as hands, \
            mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert the BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process hand and face landmarks
            hand_results = hands.process(image)
            face_results = face_mesh.process(image)

            # Convert back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            imageHeight, imageWidth, _ = image.shape

            # Virtual hand rendering (more advanced)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Draw hand landmarks with custom styles
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                    )
                    # Optional: Add more advanced rendering here

            # Face mesh rendering (detailed face tracking)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1)
                    )
                    # Optional: Add facial gesture controls here

            # Flight control logic based on hand gestures
            co = []
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Extract the wrist landmark
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    wrist_coords = mp_drawing._normalized_to_pixel_coordinates(
                        wrist.x, wrist.y, imageWidth, imageHeight
                    )
                    if wrist_coords:
                        co.append(list(wrist_coords))

            if len(co) == 2:
                # Calculate the midpoint between both wrists
                xm, ym = (co[0][0] + co[1][0]) / 2, (co[0][1] + co[1][1]) / 2
                radius = 120

                # Calculate slope (m) between the two wrists
                try:
                    m = (co[1][1] - co[0][1]) / (co[1][0] - co[0][0])
                except ZeroDivisionError:
                    m = 0

                # Quadratic equation coefficients for intersection
                a = 1 + m ** 2
                b = -2 * xm - 2 * co[0][0] * (m ** 2) + 2 * m * co[0][1] - 2 * m * ym
                c = (xm ** 2 + (m ** 2) * (co[0][0] ** 2) + co[0][1] ** 2 +
                     ym ** 2 - 2 * co[0][1] * ym - 2 * co[0][1] * co[0][0] * m +
                     2 * m * ym * co[0][0] - radius ** 2)

                discriminant = b ** 2 - 4 * a * c
                if discriminant >= 0:
                    sqrt_discriminant = math.sqrt(discriminant)
                    xa = (-b + sqrt_discriminant) / (2 * a)
                    xb = (-b - sqrt_discriminant) / (2 * a)
                    ya = m * (xa - co[0][0]) + co[0][1]
                    yb = m * (xb - co[0][0]) + co[0][1]
                else:
                    xa, ya, xb, yb = xm, ym, xm, ym  # Default positions if no real roots

                # Draw control circles and lines with new colors
                cv2.circle(image, (325, 275), 80, REMOTE_COLOR_RED, 15)  # Red circle
                cv2.circle(image, (int(xm), int(ym)), 30, REMOTE_COLOR_BLUE, 25)  # Blue circle
                cv2.ellipse(image, (int(xa), int(ya)), (100, 50), 270, 20, 160, 10, -1)
                cv2.ellipse(image, (int(xb), int(yb)), (100, 50), 90, 20, 160, 10, -1)
                l = (int(math.sqrt((co[0][0] - co[1][0]) ** 2 + (co[0][1] - co[1][1]) ** 2)) - 150) // 2
                cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), REMOTE_COLOR_BLUE, 35)  # Blue line
                cv2.line(image, (325, 275), (int(xm), int(ym)), REMOTE_COLOR_RED, 35)  # Red line
                relall()
                cv2.circle(image, (int(co[0][0]), int(co[0][1])), 30, REMOTE_COLOR_BLUE, 25)  # Blue circle

                # Gesture-based key presses
                if co[0][0] > co[1][0] and co[0][1] > co[1][1] and co[0][1] - co[1][1] > 65:
                    inputfunction.press_key('a')
                    print("a")
                elif co[1][0] > co[0][0] and co[1][1] > co[0][1] and co[1][1] - co[0][1] > 65:
                    print("a")
                    inputfunction.press_key('a')
                elif co[0][0] > co[1][0] and co[1][1] > co[0][1] and co[1][1] - co[0][1] > 65:
                    print("d")
                    inputfunction.press_key('d')
                elif co[1][0] > co[0][0] and co[0][1] > co[1][1] and co[0][1] - co[1][1] > 65:
                    print("d")
                    inputfunction.press_key('d')
                elif xm < 270 and ym > 300:
                    inputfunction.press_key('z')
                    print("z")
                elif xm > 380 and 300 > ym > 250:
                    print("Turn left.")
                    inputfunction.press_key('left')
                elif xm > 380 and ym < 250:
                    inputfunction.press_key('e')
                    print("e")
                elif 270 < xm < 380 and ym < 250:
                    inputfunction.press_key('w')
                    print("w")
                elif xm < 270 and 250 < ym < 300:
                    print("Turn right.")
                    inputfunction.press_key('right')
                elif xm < 270 and ym < 250:
                    print("q")
                    inputfunction.press_key('q')
                elif xm < 270 and 300 < ym:
                    print("z")
                    inputfunction.press_key('z')
                elif 270 < xm < 380 and 300 < ym:
                    print("x")
                    inputfunction.press_key('x')
                elif xm > 380 and ym > 300:
                    inputfunction.press_key('c')
                    print("c")
                else:
                    print("stable")


            # Display the frame
            cv2.imshow('Hand and Face Tracking', image)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
