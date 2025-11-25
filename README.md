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

Auto-generate email draft for sharing

📂 Project Structure
├── full_app_with_password_reset.py   # Main Streamlit application
├── models/                           # Place .h5 / .keras models here
├── uploaded_videos/                  # User uploaded videos
├── utils/
│   ├── preprocessing.py              # Video preprocessing logic
│   ├── translations.py               # Offline dictionary
│   └── translations_custom.json      # User-added words
├── users.json                        # User database
├── remember.json                     # Remember-me storage
├── reset_tokens.json                 # Password reset tokens
└── requirements.txt                  # Dependencies

🛠️ Installation
1️⃣ Clone the repository
git clone https://github.com/yourusername/VidLip-Reader.git
cd VidLip-Reader

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Install system dependencies (Important)

Required for video processing & emotion detection.

Windows

Install FFmpeg → https://ffmpeg.org/download.html

Add FFmpeg to PATH

Linux
sudo apt install ffmpeg libsm6 libxext6

▶️ Run the App
streamlit run full_app_with_password_reset.py

🔑 Default Login

The app automatically creates an admin account on first run:

Username: admin
Password: password


⚠️ It is recommended to change the admin password from the Admin Panel.
