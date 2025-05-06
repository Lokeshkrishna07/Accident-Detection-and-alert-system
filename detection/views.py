import os
import cv2
import tempfile
import numpy as np
import easyocr
import imutils
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from ultralytics import YOLO
from django.shortcuts import render
from django.core.mail import send_mail

# Load YOLO model and EasyOCR
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'detection', 'models', 'best.pt')
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'])

# Fake Databases
VEHICLE_DB = {
    "AP16BG1234": "lokeshone818@gmail.com",
    "TS09CH9876": "lokeshone818@gmail.com"
}

NEARBY_HOSPITAL = {
    "name": "City Emergency Hospital",
    "contact": "+917288017558",
    'email': 'lokeshone818@gmail.com'
}

def index(request):
    return render(request, 'detection/index.html')

class VideoUploadView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        video_file = request.FILES.get('video')
        location = request.POST.get('location', 'Not available')
        if not video_file:
            return Response({"error": "No video uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            for chunk in video_file.chunks():
                temp_video.write(chunk)
            temp_path = temp_video.name

        cap = cv2.VideoCapture(temp_path)
        frame_count = 0
        accident_frames = []
        saved_images = []
        extracted_plates = []

        success, frame = cap.read()
        while success:
            frame_count += 1
            temp_img_path = os.path.join(tempfile.gettempdir(), f"frame_{frame_count}.jpg")
            cv2.imwrite(temp_img_path, frame)

            results = model.predict(temp_img_path, conf=0.7)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    if label.lower() == "accident":
                        accident_frames.append(frame_count)

                        frame_filename = f"accident_frame_{frame_count}.jpg"
                        output_path = os.path.join('media/accidents', frame_filename)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        cv2.imwrite(output_path, frame)
                        saved_images.append(f"/media/accidents/{frame_filename}")

                        # Advanced Number Plate Detection Logic
                        plate_text = self.detect_number_plate(frame)
                        if plate_text:
                            print(f'Number Plate Extracted : {plate_text}')
                            extracted_plates.append(plate_text)
                            if plate_text in VEHICLE_DB:
                                self.send_alert(VEHICLE_DB[plate_text], plate_text, location, "owner")
                        else:
                            print('Number Plate Not Extracted')

                        self.send_alert(NEARBY_HOSPITAL["email"], location=location, alert_type="hospital")
                        break

            success, frame = cap.read()

        cap.release()
        os.remove(temp_path)

        return Response({
            "total_frames": frame_count,
            "accident_detected_frames": accident_frames,
            "accident_frame_images": saved_images,
            "detected_number_plates": extracted_plates
        })

    def detect_number_plate(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edge = cv2.Canny(bfilter, 30, 200)

        key = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(key)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for c in contours:
            approx = cv2.approxPolyDP(c, 10, True)
            if len(approx) == 4:
                location = approx
                break

        if location is None:
            return None

        mask = np.zeros(gray.shape, np.uint8)
        new_img = cv2.drawContours(mask, [location], 0, 255, -1)
        new_img = cv2.bitwise_and(image, image, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_img = gray[x1:x2+1, y1:y2+1]

        result = reader.readtext(cropped_img)
        if result:
            return result[0][-2].replace(" ", "").upper()
        return None

    def send_alert(self, mail, number_plate=None, location=None, alert_type=None):
        message = "🚨 Accident detected"
        if number_plate:
            message += f" involving vehicle {number_plate}."
        if location:
            message += f" 📍 Location: {location}"
        if alert_type == "hospital":
            message += " Please prepare for emergency response."
        elif alert_type == "owner":
            message += " Please check on your vehicle immediately."

        send_mail(
            subject='Accident Alert',
            message=message,
            from_email='your_email@gmail.com',  # Replace with your Gmail
            recipient_list=[mail],
            fail_silently=False,
        )