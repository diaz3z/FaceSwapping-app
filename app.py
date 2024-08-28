import cv2
import mediapipe as mp
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog

mp_face_mesh = mp.solutions.face_mesh

def detect_face_landmarks(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
    return results.multi_face_landmarks

def get_face_landmarks(landmarks, image_shape):
    h, w = image_shape[:2]
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]

def extract_face_region(image, landmarks):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    face_landmarks = get_face_landmarks(landmarks, image.shape)
    convexhull = cv2.convexHull(np.array(face_landmarks))
    cv2.fillConvexPoly(mask, convexhull, 255)
    
    face_region = cv2.bitwise_and(image, image, mask=mask)
    return face_region, mask

def align_face(source_image, target_image, source_landmarks, target_landmarks):
    s_points = np.array(get_face_landmarks(source_landmarks, source_image.shape), dtype=np.float32)
    t_points = np.array(get_face_landmarks(target_landmarks, target_image.shape), dtype=np.float32)
    
    # Use more landmarks for better alignment
    landmark_indices = [1, 33, 61, 199, 263, 291]  # Nose, chin, eyes, cheeks
    s_points = s_points[landmark_indices]
    t_points = t_points[landmark_indices]
    
    M, _ = cv2.estimateAffinePartial2D(s_points, t_points)
    aligned_face = cv2.warpAffine(source_image, M, (target_image.shape[1], target_image.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return aligned_face

def blend_faces(target_image, aligned_face, face_mask):
    mask_blurred = cv2.GaussianBlur(face_mask, (25, 25), 15)
    mask_normalized = mask_blurred.astype(float) / 255.0
    
    # Color correction
    aligned_face_lab = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB)
    
    aligned_face_lab[:,:,0] = target_lab[:,:,0]  # Match luminance
    color_corrected_face = cv2.cvtColor(aligned_face_lab, cv2.COLOR_LAB2BGR)
    
    result = (1.0 - mask_normalized[:, :, np.newaxis]) * target_image + \
             mask_normalized[:, :, np.newaxis] * color_corrected_face
    
    return result.astype(np.uint8)

def swap_face(source_image, target_image):
    source_landmarks = detect_face_landmarks(source_image)
    target_landmarks = detect_face_landmarks(target_image)

    if source_landmarks and target_landmarks:
        source_face, source_mask = extract_face_region(source_image, source_landmarks[0])
        target_face, target_mask = extract_face_region(target_image, target_landmarks[0])

        aligned_face = align_face(source_face, target_image, source_landmarks[0], target_landmarks[0])
        aligned_mask = align_face(source_mask, target_image, source_landmarks[0], target_landmarks[0])
        
        swapped_image = blend_faces(target_image, aligned_face, aligned_mask)

        return swapped_image
    return target_image

class FaceSwapApp:
    def __init__(self):
        self.app = ctk.CTk()
        self.app.title("Face Swap App")
        self.app.geometry("800x600")

        self.source_image = None
        self.target_image = None

        self.load_source_button = ctk.CTkButton(self.app, text="Load Source Image", command=self.load_source_image)
        self.load_target_button = ctk.CTkButton(self.app, text="Load Target Image", command=self.load_target_image)
        self.swap_button = ctk.CTkButton(self.app, text="Swap Face", command=self.swap_and_display)

        self.source_label = ctk.CTkLabel(self.app)
        self.target_label = ctk.CTkLabel(self.app)

        self.load_source_button.pack(pady=10)
        self.load_target_button.pack(pady=10)
        self.swap_button.pack(pady=10)
        self.source_label.pack(side="left", padx=10)
        self.target_label.pack(side="right", padx=10)

    def load_source_image(self):
        self.load_image('source')

    def load_target_image(self):
        self.load_image('target')

    def load_image(self, image_type):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            if image_type == 'source':
                self.source_image = image
                self.display_image(self.source_image, self.source_label)
            else:
                self.target_image = image
                self.display_image(self.target_image, self.target_label)

    def display_image(self, image, label):
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = img.resize((300, 300))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)
        label.configure(image=img_tk)
        label.image = img_tk

    def swap_and_display(self):
        if self.source_image is not None and self.target_image is not None:
            swapped_image = swap_face(self.source_image, self.target_image)
            self.display_image(swapped_image, self.target_label)

    def run(self):
        self.app.mainloop()

if __name__ == "__main__":
    app = FaceSwapApp()
    app.run()
