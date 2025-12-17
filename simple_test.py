import cv2

print("Testing webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
else:
    print("✓ Webcam works!")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Webcam Test - Press Q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()