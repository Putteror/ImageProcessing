
from insightface.app import FaceAnalysis
from faceEmbList import face_emb_list
import numpy as np
import glob, os
import insightface
import json
import cv2

app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU

def get_face_embedding(image_path):

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    faces = app.get(img)
    
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    
    return faces[0].embedding

def compare_faces(emb1, emb2, threshold=0.55): # Adjust this threshold according to your usecase.
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold

def calculate_face_embedding(folderPath):

	image_extensions = ["*.jpg", "*.jpeg", "*.png"]

	face_emb_list = {}
	image_paths = []
	for ext in image_extensions:
		face_emb_data = {}
		image_paths.extend(glob.glob(os.path.join(folderPath, ext)))

	for image_path in image_paths :

		try :
			emb = get_face_embedding(image_path)
			face_emb_list[image_path] = emb.tolist()
			# print(face_emb_data['name'])

		except Exception as e:
			print(image_path, str(e))

	with open("faceEmbeded.json", "w") as f:
		json.dump(face_emb_list, f)


def analysis_video(videoPath):

	cap = cv2.VideoCapture("video.mp4")  # or 0

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		faces = app.get(frame)
		for face in faces:
			emb = face.embedding
			best_match = None
			highest_score = 0

			for known_face in face_emb_list:
				known_emb = known_face['vector']
				sim, is_match = compare_faces(emb, np.array(known_emb))
				if is_match and sim > highest_score:
					highest_score = sim
					best_match = known_face['name']

			if best_match :
				box = face.bbox.astype(int)
				cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
				label = best_match if best_match else "Unknown"
				cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

		cv2.imshow("Face Recognition", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
		    break

	cap.release()
	cv2.destroyAllWindows()


def people_face_recognition(personFacePath) :

	face_image_for_recog_path = personFacePath
	emb_recog = get_face_embedding(face_image_for_recog_path)

	with open("faceEmbeded.json", "r") as f:
		face_emb_list = json.load(f)

	for face_name in face_emb_list :
		similarity_score, is_same_person = compare_faces(np.array(face_emb_list[face_name]), np.array(emb_recog))
		print(face_name, similarity_score)
		if is_same_person :
			print(face_name)


def compare_face_1_1(faceImagePath1, faceImagePath2) :

	face_embed_1 = get_face_embedding(faceImagePath1)
	face_embed_2 = get_face_embedding(faceImagePath2)

	print(compare_faces(face_embed_1, face_embed_2))


if __name__ == '__main__':

	# calculate_face_embedding("nvk")
	# people_face_recognition("testImage/may.jpg")
	compare_face_1_1("testImage/may.jpg", "testImage/putter.jpg")




	