# VidLip-Reader

Deep-learning lip-reading application with video preprocessing, model inference (TensorFlow/Keras), and a sleek Streamlit UI.
Supports offline/online translations, emotion detection, user authentication, admin tools, and custom model uploads.
Built using Streamlit, TensorFlow, OpenCV, FER, bcrypt, and gTTS.

🚀 Features
🎥 Lip Reading

Upload a video (MP4 / AVI / MOV)

AI model predicts spoken text from lip movements

Supports multiple .h5 and .keras models

😊 Emotion Detection

Uses FER to detect emotion from selected frames

Shows emotion + confidence percentage

🌍 Multi-Language Translation

Offline dictionary + online fallback (GoogleTrans)

Text-to-Speech output (if gTTS installed)

🔒 User Authentication

Sign Up / Login / Logout

Bcrypt password hashing

“Remember me” support

Password reset using tokens or email

🧑‍💼 Admin Panel

Promote / Demote users

Reset any user's password

View & clear user history

Delete users

Backup & clear uploaded videos

📁 Model Manager

Upload new ML models from UI

List installed models

📤 Export & Sharing

Export reports as .txt

Download prediction & translation audio
<img width="735" height="342" alt="image" src="https://github.com/user-attachments/assets/169d715c-8558-4dee-aeb7-a3f1903ac112" />

Auto-generate email draft for sharing

🛠️ Installation
1️⃣ Clone the repository
git clone https://github.com/srinivasyad/VidLip-Reader.git
cd VidLip-Reader

2️⃣ Install dependencies
pip install -r requirements.txt

▶️ Run the App
streamlit run full_app_with_password_reset.py

🔑 Default Login

The app automatically creates an admin account on first run:

Username: admin
Password: password


⚠️ It is recommended to change the admin password from the Admin Panel.
