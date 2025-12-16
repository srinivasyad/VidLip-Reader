import streamlit as st
import cv2
import numpy as np
import base64
import tempfile
import os
import glob
import urllib.parse
import json
import hashlib
from pathlib import Path
from tensorflow.keras.models import load_model
from config import LABELS, IMG_SIZE, MAX_LEN
from utils.preprocessing import preprocess_video
import importlib
import atexit
from typing import Optional
from datetime import datetime, date, timedelta
import shutil
import secrets
import zipfile
from email.message import EmailMessage
import logging

from utils import translations as translations_mod
from utils.translations import get_translation, AVAILABLE_LANGS

# Optional: load .env in dev if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Optional: bcrypt for secure password hashing ---
BCRYPT_AVAILABLE = False
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except Exception:
    BCRYPT_AVAILABLE = False

# --- TTS: gTTS (optional) ---
TTS_AVAILABLE = False
TTS_IMPORT_ERROR = None
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except Exception as e:
    TTS_AVAILABLE = False
    import traceback
    TTS_IMPORT_ERROR = "".join(traceback.format_exception_only(type(e), e)).strip()

# ------------------ Paths & setup ------------------
ROOT = Path(__file__).parent

# logging should reference ROOT AFTER it's defined
LOGFILE = ROOT / "app.log"
logging.basicConfig(filename=str(LOGFILE), level=logging.INFO)

MODELS_DIR = ROOT / "models"
UPLOADED_VIDEO_DIR = ROOT / "uploaded_videos"
USERS_FILE = ROOT / "users.json"
REMEMBER_FILE = ROOT / "remember.json"
TRANSLATIONS_CUSTOM = ROOT / "utils" / "translations_custom.json"
RESET_TOKENS_FILE = ROOT / "reset_tokens.json"
BACKUPS_DIR = ROOT / "backups"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "utils").mkdir(parents=True, exist_ok=True)
BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
if not TRANSLATIONS_CUSTOM.exists():
    TRANSLATIONS_CUSTOM.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")
if not RESET_TOKENS_FILE.exists():
    RESET_TOKENS_FILE.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------ Users utilities ------------------
def load_users():
    if not USERS_FILE.exists():
        return {}
    try:
        return json.load(open(USERS_FILE, "r", encoding="utf-8"))
    except Exception:
        return {}

def save_users(users):
    """
    Atomic save to avoid corruption: write to tmp file then replace.
    """
    try:
        def json_serial(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            raise TypeError ("Type %s not serializable" % type(obj))
        tmp = USERS_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(users, f, default=json_serial, indent=2, ensure_ascii=False)
        tmp.replace(USERS_FILE)
        return True
    except Exception:
        return False

def load_remember():
    if not REMEMBER_FILE.exists():
        return {"remember": False, "username": ""}
    try:
        return json.load(open(REMEMBER_FILE, "r", encoding="utf-8"))
    except Exception:
        return {"remember": False, "username": ""}

def save_remember(username, remember):
    try:
        json.dump({"remember": bool(remember), "username": username if remember else ""}, open(REMEMBER_FILE, "w", encoding="utf-8"), indent=2)
    except Exception:
        pass

# ------------------ Reset tokens storage helpers ------------------
def _load_reset_tokens():
    try:
        return json.load(open(RESET_TOKENS_FILE, "r", encoding="utf-8"))
    except Exception:
        return {}

def _save_reset_tokens(d):
    try:
        tmp = RESET_TOKENS_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        tmp.replace(RESET_TOKENS_FILE)
    except Exception:
        pass

# ------------------ Password hashing & verification ------------------
def hash_password_bcrypt(pw: str) -> str:
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password_bcrypt(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def hash_password_fallback_sha256(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def verify_password_fallback_sha256(pw: str, hashed: str) -> bool:
    try:
        return hashlib.sha256(pw.encode("utf-8")).hexdigest() == hashed
    except Exception:
        return False

def hash_password(pw: str) -> str:
    if BCRYPT_AVAILABLE:
        return hash_password_bcrypt(pw)
    else:
        return hash_password_fallback_sha256(pw)

def verify_password(pw: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False
    if isinstance(stored_hash, str) and stored_hash.startswith("$2"):
        # bcrypt
        if not BCRYPT_AVAILABLE:
            return False
        return verify_password_bcrypt(pw, stored_hash)
    return verify_password_fallback_sha256(pw, stored_hash)

# ------------------ Migration helper ------------------
def migrate_users_defaults():
    users = load_users()
    modified = False
    if "admin" in users:
        if isinstance(users["admin"], str):
            users["admin"] = {"password": users["admin"], "email": "admin@lipread.pro", "bio": "System Administrator", "phone": "", "dob": "", "history": []}
            modified = True
        if isinstance(users["admin"], dict):
            if not users["admin"].get("is_admin", False):
                users["admin"]["is_admin"] = True
                modified = True
    else:
        users["admin"] = {
            "password": hash_password("password"),
            "email": "admin@lipread.pro",
            "bio": "System Administrator",
            "phone": "",
            "dob": "",
            "history": [],
            "is_admin": True
        }
        modified = True

    for uname, u in list(users.items()):
        if isinstance(u, dict):
            if "is_admin" not in u:
                u["is_admin"] = False
                modified = True
        else:
            users[uname] = {"password": u, "history": [], "is_admin": False}
            modified = True

    if modified:
        save_users(users)

migrate_users_defaults()

# ------------------ Translations helpers ------------------
def load_and_merge_custom_translations():
    try:
        data = {}
        if TRANSLATIONS_CUSTOM.exists():
            data = json.load(open(TRANSLATIONS_CUSTOM, "r", encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
    except Exception:
        data = {}
    try:
        for eng, mapping in data.items():
            if eng in translations_mod.OFFLINE_DICT and isinstance(mapping, dict):
                translations_mod.OFFLINE_DICT[eng].update(mapping)
            else:
                translations_mod.OFFLINE_DICT[eng] = mapping
    except Exception:
        pass

def reload_translations_module():
    try:
        importlib.reload(translations_mod)
        load_and_merge_custom_translations()
        return True, "Reloaded translations module and merged custom translations."
    except Exception as e:
        return False, f"Reload failed: {e}"

def save_custom_translation(english_text: str, language_name: str, translation_text: str) -> bool:
    try:
        cur = {}
        if TRANSLATIONS_CUSTOM.exists():
            try:
                cur = json.load(open(TRANSLATIONS_CUSTOM, "r", encoding="utf-8"))
            except Exception:
                cur = {}
        if english_text not in cur:
            cur[english_text] = {}
        cur[english_text][language_name] = translation_text
        json.dump(cur, open(TRANSLATIONS_CUSTOM, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        if english_text in translations_mod.OFFLINE_DICT:
            translations_mod.OFFLINE_DICT[english_text][language_name] = translation_text
        else:
            translations_mod.OFFLINE_DICT[english_text] = {language_name: translation_text}
        return True
    except Exception:
        return False

def try_online_translate(text: str, target_lang: str):
    """
    Online translation using deep-translator (Google backend).
    Returns translated text on success, or a string starting with "__ERR__:" on failure.
    """
    try:
        from deep_translator import GoogleTranslator
    except Exception as e:
        logging.error(f"deep-translator import failed: {e}")
        return "__ERR__: deep-translator import failed: " + str(e)

    code = AVAILABLE_LANGS.get(target_lang)
    if not code:
        return "__ERR__: Invalid language code for: " + str(target_lang)

    try:
        # GoogleTranslator will perform network request to Google Translate (no official API key)
        translated = GoogleTranslator(source='auto', target=code).translate(text)
        return translated
    except Exception as e:
        logging.exception("deep-translator translation failed")
        return "__ERR__: " + str(e)

# ------------------ Models & prediction ------------------
def discover_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    candidates += glob.glob(os.path.join(script_dir, "models", "*.keras"))
    candidates += glob.glob(os.path.join(script_dir, "models", "*.h5"))
    candidates += glob.glob(os.path.join(script_dir, "..", "models", "*.keras"))
    candidates += glob.glob(os.path.join(script_dir, "..", "models", "*.h5"))
    d = {}
    for p in candidates:
        try:
            d[os.path.basename(p)] = p
        except Exception:
            pass
    if not d:
        return {"No model found (upload one to models/)": None}
    return d

@st.cache_resource
def cached_load_model(path):
    if not path:
        return None
    try:
        model = load_model(path)
        return model
    except Exception:
        return None

def map_prediction_to_label(prediction_array):
    try:
        idx = int(np.argmax(prediction_array))
    except Exception:
        return "Unknown"
    keys = list(LABELS.keys())
    if idx < 0 or idx >= len(keys):
        return "Unknown"
    return LABELS.get(keys[idx], "Unknown")

# ------------------ TTS helpers ------------------
_tts_temp_files = []

def _cleanup_tts_files():
    for p in list(_tts_temp_files):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    _tts_temp_files.clear()

atexit.register(_cleanup_tts_files)

def text_to_speech_bytes(text: str, lang_code: str = "en") -> Optional[bytes]:
    if not TTS_AVAILABLE:
        return None
    try:
        safe_text = (text or "").strip()
        if not safe_text:
            safe_text = " "
        safe_text = safe_text[:4000]
        t = gTTS(safe_text, lang=lang_code)
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tf.close()
        t.save(tf.name)
        _tts_temp_files.append(tf.name)
        with open(tf.name, "rb") as f:
            data = f.read()
        return data
    except Exception:
        return None

# ------------------ Note: Emotion detection removed ------------------
# All FER-related code and UI bits were removed to avoid import errors and runtime issues.
# If you want to re-enable emotion detection later, re-add the fer import and associated helpers.

# ---------- Streamlit setup & session ----------
st.set_page_config(page_title="LipRead Pro", layout="wide", initial_sidebar_state="expanded", page_icon="")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_bio" not in st.session_state:
    st.session_state.user_bio = ""
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "app"

if "_prediction_done" not in st.session_state:
    st.session_state["_prediction_done"] = False
if "_last_prediction" not in st.session_state:
    st.session_state["_last_prediction"] = ""
if "_last_translation" not in st.session_state:
    st.session_state["_last_translation"] = ""
# emotion removed; do NOT set _last_emotion
if "_admin_action_confirm" not in st.session_state:
    st.session_state["_admin_action_confirm"] = {}
if "pending_password_change_user" not in st.session_state:
    st.session_state["pending_password_change_user"] = None

if "_tts_pred_bytes" not in st.session_state:
    st.session_state["_tts_pred_bytes"] = None
if "_tts_trans_bytes" not in st.session_state:
    st.session_state["_tts_trans_bytes"] = None
if "_tts_manual_bytes" not in st.session_state:
    st.session_state["_tts_manual_bytes"] = None

if "_last_video_path" not in st.session_state:
    st.session_state["_last_video_path"] = None

if "_tts_error" not in st.session_state:
    st.session_state["_tts_error"] = TTS_IMPORT_ERROR or ""

load_and_merge_custom_translations()

# --- helper to clear last prediction ---
def clear_prediction():
    st.session_state["_prediction_done"] = False
    st.session_state["_last_prediction"] = ""
    st.session_state["_last_translation"] = ""
    # emotion removed - do not touch _last_emotion
    st.session_state["_tts_pred_bytes"] = None
    st.session_state["_tts_trans_bytes"] = None
    st.session_state["_tts_manual_bytes"] = None
    st.session_state["_last_video_path"] = None

# ------------------ ADMIN: helpers ------------------
def is_admin_user():
    uname = st.session_state.get("username", "")
    if not uname:
        return False
    users = load_users()
    u = users.get(uname)
    if not u:
        return False
    if isinstance(u, dict):
        return bool(u.get("is_admin", False)) or uname == "admin"
    return uname == "admin"

def admin_set_user_flag(username: str, flag: str, value):
    users = load_users()
    if username in users:
        if isinstance(users[username], str):
            users[username] = {"password": users[username]}
        users[username][flag] = value
        save_users(users)
        return True
    return False

# ------------------ Admin-requested password reset helpers ------------------
def request_admin_reset(username: str):
    """
    Mark a username as requesting an admin reset. Returns True if request recorded.
    """
    users = load_users()
    if username not in users:
        return False
    if isinstance(users[username], str):
        users[username] = {"password": users[username]}
    users[username]["admin_reset_requested"] = True
    save_users(users)
    return True

def clear_admin_reset_request(username: str):
    users = load_users()
    if username in users and isinstance(users[username], dict):
        users[username].pop("admin_reset_requested", None)
        save_users(users)

def get_admin_reset_requests():
    """
    Return list of usernames which have admin_reset_requested == True
    """
    users = load_users()
    reqs = []
    for uname, u in users.items():
        if isinstance(u, dict) and u.get("admin_reset_requested"):
            reqs.append(uname)
    return reqs

def admin_reset_user_with_temp_password(username: str, temp_pw: str = None):
    """
    Admin-triggered reset: sets a temporary password (generated if not provided),
    sets must_change_password True, clears the request flag, and saves users.
    Returns (ok: bool, temp_password_or_msg).
    """
    users = load_users()
    if username not in users:
        return False, "User not found"
    pw = temp_pw or secrets.token_urlsafe(10)
    ok = reset_user_password(username, pw, set_must_change=True)
    if not ok:
        return False, "Failed to reset password"
    # clear request flag (reload users to be safe)
    users = load_users()
    if username in users and isinstance(users[username], dict):
        users[username].pop("admin_reset_requested", None)
        save_users(users)
    return True, pw

# ------------------ Password reset functions ------------------ 
def generate_reset_token(length=24):
    return secrets.token_urlsafe(length)

def create_password_reset_token(username: str, expire_minutes: int = 20):
    users = load_users()
    if username not in users:
        return None
    tokens = _load_reset_tokens()
    token = generate_reset_token()
    expires_at = (datetime.now() + timedelta(minutes=expire_minutes)).isoformat()
    tokens[token] = {"username": username, "expires_at": expires_at}
    _save_reset_tokens(tokens)
    # Also append to fallback for easy dev visibility (non-sensitive dev use)
    try:
        fallback_path = ROOT / "reset_tokens_fallback.txt"
        with open(fallback_path, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now().isoformat()} TO:{username} TOKEN:{token}\n")
    except Exception:
        pass
    return token

def verify_and_consume_token(token: str):
    tokens = _load_reset_tokens()
    info = tokens.get(token)
    if not info:
        return None
    try:
        if datetime.fromisoformat(info["expires_at"]) < datetime.now():
            tokens.pop(token, None)
            _save_reset_tokens(tokens)
            return None
    except Exception:
        tokens.pop(token, None)
        _save_reset_tokens(tokens)
        return None
    username = info["username"]
    tokens.pop(token, None)
    _save_reset_tokens(tokens)
    return username

def reset_user_password(username: str, new_password: str, set_must_change=False):
    users = load_users()
    if username not in users:
        return False
    if isinstance(users[username], str):
        users[username] = {"password": users[username]}
    users[username]["password"] = hash_password(new_password)
    if set_must_change:
        users[username]["must_change_password"] = True
    save_users(users)
    return True

# ------------------ UI STYLES (auto-switch for light/dark) ------------------
# Replace previous CSS block with this conditional block to ensure text is visible in light mode.
# It detects Streamlit's theme base and applies a light-friendly palette when appropriate.

# Try to read Streamlit theme base (falls back to 'dark' if unavailable)
try:
    theme_base = st.get_option("theme.base")  # 'light' or 'dark'
except Exception:
    theme_base = "dark"

if theme_base == "light":
    # Light-mode CSS
# ---------- Replace only the light-mode CSS string with this block ----------
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    :root{
        --bg: #FFFFFF;
        --surface: #F7FAFC;
        --card: #FFFFFF;
        --accent: #0ea5a4;      /* teal */
        --accent-2: #6366F1;    /* indigo */
        --muted: #475569;       /* darker muted text for readability */
        --text: #071428;        /* very dark text for high contrast */
        --border: rgba(7,20,40,0.08);
        --glass: rgba(7,20,40,0.03);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        color: var(--text);
        background: linear-gradient(180deg, var(--surface), #F1F5F9 140%);
    }

    .main > div.block-container {
        padding-top: 28px;
        padding-bottom: 28px;
    }

    /* Buttons - clearer, higher contrast */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 14px;
        border: 1px solid var(--border);
        background: linear-gradient(90deg, rgba(7,20,40,0.02), rgba(7,20,40,0.01));
        color: var(--text);
        box-shadow: none;
    }
    .stButton>button[type="primary"] {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: white;
        border: none;
        box-shadow: 0 6px 18px rgba(6,182,212,0.08);
    }
    /* Disabled buttons - visibly disabled but legible */
    .stButton>button:disabled {
        background: #f3f4f6;
        color: #9ca3af;
        border: 1px solid rgba(7,20,40,0.04);
        box-shadow: none;
    }

    /* Input fields - darker placeholder and text */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stDateInput>div>div>input {
        background: #ffffff;
        border: 1px solid var(--border);
        color: var(--text);
        padding: 10px 12px;
        border-radius: 8px;
    }
    .stTextInput>div>div>input::placeholder,
    .stTextArea>div>div>textarea::placeholder {
        color: #94a3b8;
    }

    /* File uploader - make card clear and text readable */
    div[data-testid="stFileUploader"] {
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(7,20,40,0.02), rgba(7,20,40,0.00));
        padding: 14px;
        border: 1px dashed rgba(7,20,40,0.06);
    }
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] p {
        color: var(--muted);
    }

    /* Header and brand */
    .app-header {
        display:flex; align-items:center; gap:18px;
        padding:18px 22px; border-radius:12px;
        background: linear-gradient(90deg, rgba(7,20,40,0.02), rgba(7,20,40,0.01));
        border: 1px solid var(--border);
        margin-bottom: 22px;
    }
    .app-title { font-weight:800; font-size:20px; color:var(--text); letter-spacing: -0.3px; }
    .app-sub { color:var(--muted); font-size:13px; font-weight:400; }

    .brand-icon {
        width:48px; height:48px; border-radius:12px;
        display:flex; align-items:center; justify-content:center;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        color: white; font-weight:800; font-size:20px;
        box-shadow: 0 6px 12px rgba(99,102,241,0.06);
    }

    /* Cards & Containers - remove heavy blur and keep crisp borders */
    .content-card {
        background: linear-gradient(180deg, var(--card), var(--surface));
        border: 1px solid var(--border);
        padding: 22px; border-radius:12px;
        box-shadow: 0 2px 8px rgba(7,20,40,0.04);
        margin-bottom: 16px;
    }

    /* Sidebar: make menu items clearly visible */
    .sidebar-profile-box {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        margin-bottom: 15px;
        background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(255,255,255,0.95));
    }
    /* Streamlit renders sidebar buttons as normal buttons — ensure contrast in sidebar */
    .stSidebar .stButton>button {
        width:100%;
        text-align:left;
        padding:14px 18px;
        background: #ffffff;
        color: var(--text);
        border: 1px solid rgba(7,20,40,0.06);
        box-shadow: 0 6px 18px rgba(7,20,40,0.03);
    }
    .stSidebar .stButton>button[type="primary"] {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: white;
        border: none;
    }

    /* Login card - lighter, clearer */
    .login-container { max-width: 640px; margin: auto; padding-top: 28px; }
    .auth-card {
        background: #ffffff;
        padding: 28px;
        border-radius: 12px;
        border: 1px solid var(--border);
        box-shadow: 0 8px 32px rgba(7,20,40,0.04);
    }

    /* Tabs & small text */
    .stTabs [role="tab"] { color: var(--muted); font-weight:600; }
    .stTabs [role="tab"][aria-selected="true"] { color: var(--text); }

    .stMarkdown p, .stMarkdown span, .stText {
        color: var(--muted);
    }
    .st-header { color: var(--text); }

    /* Ensure form labels and small controls are readable */
    label, .stLabel, .css-1v0mbdj, .css-1xt0b0l {
        color: var(--text) !important;
    }

    /* Dataframe header & text readability */
    .stDataFrame table th { color: var(--text); }
    .stDataFrame table td { color: var(--text); }

    /* Make small helper links visible */
    a, .stLink { color: var(--accent-2); text-decoration: none; }

    </style>
    """

else:
    # Original Dark-mode CSS (keeps behaviour you had before)
    css = """
    <style>
      /* -------------------------------------------------
         New visual theme: 'Calm Teal' — clean, higher contrast,
         larger spacing, clearer buttons and compact cards.
         Only affects styling — no code logic changed.
         ------------------------------------------------- */

      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

      :root{
        --bg: #0F172A;          /* deep slate background for app chrome */
        --surface: #0B1220;     /* slightly lighter panel */
        --card: #071022;        /* card background */
        --accent: #06B6D4;      /* teal-cyan accent */
        --accent-2: #7C3AED;    /* secondary purple accent for highlights */
        --muted: #94A3B8;       /* muted text */
        --text: #E6EEF6;        /* primary text */
        --border: rgba(255,255,255,0.06);
        --glass: rgba(255,255,255,0.02);
      }

      html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        color: var(--text);
        background: linear-gradient(180deg, var(--bg), #061725 140%);
      }

      .main > div.block-container {
        padding-top: 28px;
        padding-bottom: 28px;
      }

      .stButton>button {
        border-radius: 10px;
        font-weight: 650;
        padding: 10px 14px;
        border: 1px solid var(--border);
        background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        color: var(--text);
        box-shadow: 0 6px 18px rgba(3,7,18,0.6);
      }
      .stButton>button[type="primary"] {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: white;
        border: none;
      }
      .stButton>button[type="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(6,182,212,0.12);
      }

      .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stDateInput>div>div>input {
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--border);
        color: var(--text);
        padding: 10px 12px;
        border-radius: 8px;
      }

      div[data-testid="stFileUploader"] {
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(255,255,255,0.01), transparent);
        padding: 14px;
        border: 1px dashed rgba(255,255,255,0.04);
      }

      .app-header {
        display:flex; align-items:center; gap:18px;
        padding:18px 22px; border-radius:12px;
        background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid var(--border);
        margin-bottom: 22px;
      }
      .app-title { font-weight:800; font-size:20px; color:var(--text); letter-spacing: -0.3px; }
      .app-sub { color:var(--muted); font-size:13px; font-weight:400; }

      .brand-icon {
        width:48px; height:48px; border-radius:12px;
        display:flex; align-items:center; justify-content:center;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        color: white; font-weight:800; font-size:20px;
        box-shadow: 0 8px 30px rgba(12,18,30,0.6);
      }

      .content-card {
        background: linear-gradient(180deg, var(--card), var(--surface));
        border: 1px solid var(--border);
        padding: 22px; border-radius:14px;
        box-shadow: 0 6px 18px rgba(2,6,12,0.6);
        margin-bottom: 18px;
      }

      .upload-zone {
        border: 1px dashed rgba(255,255,255,0.04);
        background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));
        padding: 12px 18px 18px 18px;
        border-radius:12px; text-align:center;
      }
      .upload-zone:hover { border-color: rgba(255,255,255,0.08); }

      .sidebar-profile-box {
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 14px;
          text-align: center;
          margin-bottom: 15px;
          background: linear-gradient(180deg, rgba(255,255,255,0.01), transparent);
      }

      .login-container { max-width: 460px; margin: auto; padding-top: 28px; }
      .auth-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        padding: 28px;
        border-radius: 14px;
        border: 1px solid var(--border);
        box-shadow: 0 10px 30px rgba(2,6,12,0.6);
      }

      .stMarkdown p, .stMarkdown span, .stText {
        color: var(--muted);
      }
      .st-header {
        color: var(--text);
      }
    </style>
    """

st.markdown(css, unsafe_allow_html=True)

# ------------------ AUTH page ------------------ (same as before)
def show_auth_page():
    c_side1, c_main, c_side2 = st.columns([1, 1, 1])
    
    with c_main:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='auth-card'>
                <div style='text-align:center; margin-bottom:24px;'>
                    <div style='font-size:40px;'></div>
                    <h2 style='margin:0; margin-top:10px;'>LipRead Pro</h2>
                    <p style='color:#94A3B8; margin:0;'>Intelligent Lip Reading & Analysis</p>
                </div>
            """, 
            unsafe_allow_html=True
        )

        remembered = load_remember()
        default_username = remembered.get("username", "") if remembered.get("remember", False) else ""

        tab_login, tab_signup = st.tabs(["Sign In", "New Account"])
        
        with tab_login:
            st.write("")
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", value=default_username, placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                remember = st.checkbox("Remember for 30 days", value=remembered.get("remember", False))
                
                if st.form_submit_button("Sign In", type="primary", use_container_width=True):
                    users = load_users()
                    if username in users:
                        user_data = users[username]
                        stored_hash = user_data if isinstance(user_data, str) else user_data.get("password")
                        
                        if verify_password(password, stored_hash):
                            # Migrate to bcrypt on first successful login if bcrypt is available
                            if BCRYPT_AVAILABLE and (not isinstance(stored_hash, str) or not stored_hash.startswith("$2")):
                                try:
                                    users[username] = user_data if isinstance(user_data, dict) else {"password": user_data}
                                    users[username]["password"] = hash_password(password)
                                    save_users(users)
                                except Exception:
                                    pass

                            # Check must_change_password
                            must_change = False
                            if isinstance(user_data, dict):
                                must_change = bool(user_data.get("must_change_password", False))

                            st.session_state.logged_in = True
                            st.session_state.username = username
                            if isinstance(user_data, dict):
                                st.session_state.user_email = user_data.get("email", "")
                                st.session_state.user_bio = user_data.get("bio", "")
                            else:
                                st.session_state.user_email = ""
                                st.session_state.user_bio = ""
                            save_remember(username, remember)

                            if must_change:
                                st.session_state["pending_password_change_user"] = username
                                st.session_state.view_mode = "change_password"
                                st.rerun()
                            else:
                                st.rerun()
                        else:
                            st.error("Incorrect password.")
                    else:
                        st.error("Account does not exist.")

            # Forgot password flow UI (token-based)
            st.markdown("---")
            with st.expander("Forgot Password?"):
                fp_user = st.text_input("Enter your username to reset password", key="fp_user")
                # ---- REPLACED: Request token handler (no SMTP, no token display) ----
                if st.button("Request password reset token", key="req_token"):
                    if not fp_user:
                        st.warning("Enter username")
                    else:
                        users = load_users()
                        if fp_user not in users:
                            st.error("Username not found")
                        else:
                            token = create_password_reset_token(fp_user, expire_minutes=20)
                            if not token:
                                st.error("Failed to create reset token.")
                            else:
                                # create_password_reset_token already appends to reset_tokens_fallback.txt
                                st.success("A password reset token has been created. If you need assistance, contact an administrator.")
                # ---- END REPLACED BLOCK ----

                st.markdown("### Reset with token")
                token_input = st.text_input("Reset token", key="reset_token")
                new_pw = st.text_input("New password", type="password", key="reset_new_pw")
                confirm_pw = st.text_input("Confirm new password", type="password", key="reset_new_pw_conf")
                if st.button("Reset password with token", key="apply_token"):
                    if not token_input or not new_pw:
                        st.warning("Provide token and new password")
                    elif new_pw != confirm_pw:
                        st.error("Passwords do not match")
                    else:
                        who = verify_and_consume_token(token_input)
                        if not who:
                            st.error("Invalid or expired token")
                        else:
                            ok = reset_user_password(who, new_pw)
                            if ok:
                                users = load_users()
                                if who in users and isinstance(users[who], dict) and users[who].get("must_change_password"):
                                    users[who]["must_change_password"] = False
                                    save_users(users)
                                st.success("Password changed. You may now login.")
                            else:
                                st.error("Failed to reset password.")

                st.markdown("---")
                st.write("Or request an admin to reset your password (admin will generate a temporary password).")
                if st.button("Request admin reset", key="req_admin_reset"):
                    if not fp_user:
                        st.warning("Enter username above first.")
                    else:
                        if request_admin_reset(fp_user):
                            st.success("Admin reset requested. Please contact an admin to complete the reset.")
                            st.info("Admin will set a temporary password and you must change it at next login.")
                        else:
                            st.error("Failed to request admin reset (username may not exist).")

        with tab_signup:
            st.write("")
            with st.form("signup_form", clear_on_submit=False):
                new_user = st.text_input("Choose Username")
                new_email = st.text_input("Email Address")
                new_pass = st.text_input("Create Password", type="password")
                confirm_pass = st.text_input("Confirm Password", type="password") 
                new_bio = st.text_input("Short Bio (Optional)", placeholder="Job title or usage...")
                
                if st.form_submit_button("Create Account", use_container_width=True):
                    if not new_user or not new_pass or not new_email:
                        st.warning("Username, Email, and Password are required.")
                    elif new_pass != confirm_pass: 
                        st.error("Passwords do not match.")
                    else:
                        users = load_users()
                        if new_user in users:
                            st.error("Username taken.")
                        else:
                            users[new_user] = {
                                "password": hash_password(new_pass),
                                "email": new_email,
                                "bio": new_bio,
                                "phone": "",
                                "dob": "",
                                "history": [],
                                "is_admin": False,
                                "must_change_password": False
                            }
                            if save_users(users):
                                st.success("Account created! You can now log in.")
                                st.rerun()
                            else:
                                st.error("Storage error.")

        st.markdown("</div></div>", unsafe_allow_html=True)


# ------------------ MAIN page ------------------
def show_main_page():
    user_initial = st.session_state.username[0].upper() if st.session_state.username else "?"
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-profile-box">
            <div style="font-size:32px; background:linear-gradient(135deg,var(--accent),var(--accent-2)); width:70px; height:70px; color:white; border-radius:14px; display:flex; align-items:center; justify-content:center; margin:0 auto 10px auto;">
                {user_initial}
            </div>
            <div style="font-weight:700; color:var(--text);">{st.session_state.username}</div>
            <div style="font-size:12px; color:var(--muted);">{st.session_state.user_email or 'User'}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Menu")
        if st.button("📊  Prediction Studio", use_container_width=True):
            st.session_state.view_mode = "app"
            st.rerun()

        if st.button("⚙️  Account Settings", use_container_width=True):
            st.session_state.view_mode = "account"
            st.rerun()

        st.markdown("---")
        
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            clear_prediction()
            st.rerun()

        # Admin area
        if is_admin_user():
            st.markdown("---")
            st.markdown("#### ⚙️ Admin Panel")
            if st.button("👥 Manage Users", use_container_width=True):
                st.session_state.view_mode = "admin_users"
                st.rerun()
            if st.button("🗑️ Clear All Uploads", use_container_width=True):
                st.session_state["_admin_action_confirm"] = {"action": "clear_uploads"}
            if st.button("🧭 Reload Translations", use_container_width=True):
                ok, msg = reload_translations_module()
                if ok: st.toast(msg, icon="✅")
                else: st.error(msg)
            if st.button("📤 Export users.json", use_container_width=True):
                users = load_users()
                st.download_button("Download users.json", json.dumps(users, indent=2, ensure_ascii=False), file_name="users.json", use_container_width=True)

            # Admin: Reset user password quick shortcut expander
            with st.expander("Admin: Reset user password"):
                u_to_reset = st.text_input("Username to reset ", key="admin_reset_user")
                temp_pw = st.text_input("Temporary password to set (leave blank to auto generate)", type="password", key="admin_temp_pw")
                gen_btn_col1, gen_btn_col2 = st.columns([1,1])
                with gen_btn_col1:
                    if st.button("Generate Random Temp Password", key="gen_temp_pw"):
                        st.session_state["_admin_temp_pw_preview"] = secrets.token_urlsafe(10)
                with gen_btn_col2:
                    if "_admin_temp_pw_preview" in st.session_state:
                        st.markdown(f"**Generated:** `{st.session_state['_admin_temp_pw_preview']}`")
                set_must_change = st.checkbox("Force user to change password on next login", value=True, key="admin_must_change")
                if st.button("Apply Reset", key="apply_admin_reset"):
                    if not u_to_reset:
                        st.error("Enter username")
                    else:
                        pw_to_set = temp_pw or st.session_state.get("_admin_temp_pw_preview") or secrets.token_urlsafe(10)
                        ok = reset_user_password(u_to_reset, pw_to_set, set_must_change)
                        if ok:
                            # show generated pw if admin didn't supply one
                            st.success(f"Password for {u_to_reset} reset.")
                            if not temp_pw:
                                st.info(f"Generated temporary password: `{pw_to_set}` (share it securely with the user).")
                        else:
                            st.error("User not found or error.")

            # Render confirmation UI for "clear_uploads" action
            if st.session_state.get("_admin_action_confirm", {}).get("action") == "clear_uploads":
                st.warning("Confirm: Delete all files in uploaded_videos/ ? This cannot be undone.")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Confirm Delete Uploads", key="confirm_clear_uploads"):
                        try:
                            # backup before deleting
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            out = BACKUPS_DIR / f"uploads_backup_{ts}.zip"
                            with zipfile.ZipFile(out, "w") as z:
                                for f in UPLOADED_VIDEO_DIR.glob("*"):
                                    if f.is_file():
                                        z.write(f, arcname=f.name)
                            removed = 0
                            for f in UPLOADED_VIDEO_DIR.glob("*"):
                                if f.is_file():
                                    f.unlink()
                                    removed += 1
                            st.toast(f"Backed up to {out.name} and deleted {removed} files.", icon="✅")
                        except Exception as e:
                            st.error(f"Error deleting files: {e}")
                        st.session_state["_admin_action_confirm"] = {}
                with c2:
                    if st.button("Cancel", key="cancel_clear_uploads"):
                        st.session_state["_admin_action_confirm"] = {}

    # Header
    st.markdown(
        """
        <div class="app-header">
          <div class="brand-icon">LR</div>
          <div style="flex:1">
            <div class="app-title">LipRead Pro</div>
            <div class="app-sub">AI-Powered Video Speech Recognition</div>
          </div>
          <div style="padding: 6px 14px; background:rgba(255,255,255,0.02); border-radius:8px; color:var(--muted); font-size:12px; font-weight:600;">
             v1.2
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Router: change_password view
    if st.session_state.view_mode == "change_password" or st.session_state.get("pending_password_change_user"):
        user = st.session_state.get("pending_password_change_user")
        st.header("Change your password")
        st.markdown("You must change your password before continuing.")
        p1 = st.text_input("New password", type="password", key="cp_new1")
        p2 = st.text_input("Confirm new password", type="password", key="cp_new2")
        if st.button("Change password", key="cp_apply"):
            if not p1:
                st.warning("Enter a new password")
            elif p1 != p2:
                st.error("Passwords do not match")
            else:
                reset_user_password(user, p1)
                users = load_users()
                if user in users and isinstance(users[user], dict) and users[user].get("must_change_password"):
                    users[user]["must_change_password"] = False
                    save_users(users)
                st.success("Password changed. Continue to app.")
                st.session_state["pending_password_change_user"] = None
                st.session_state.view_mode = "app"
                st.rerun()

    # Account view
    elif st.session_state.view_mode == "account":
        st.subheader(f"Account Management")
        users = load_users()
        user_obj = users.get(st.session_state.username, {})
        if isinstance(user_obj, str):
             user_obj = {"password": user_obj}
        current_phone = user_obj.get("phone", "")
        current_dob_str = user_obj.get("dob", "")
        default_date = date.today()
        if current_dob_str:
            try: default_date = datetime.strptime(current_dob_str, "%Y-%m-%d").date()
            except: pass
        min_dob = date(1920, 1, 1)
        max_dob = date.today()

        acc_tab1, acc_tab2 = st.tabs(["✏️ Edit Profile", "📜 Activity History"])
        with acc_tab1:
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            with st.form("profile_edit_form"):
                st.caption("Update your personal details below.")
                col_gr1, col_gr2 = st.columns(2)
                with col_gr1:
                    new_email = st.text_input("Email Address", value=st.session_state.user_email)
                    new_phone = st.text_input("Phone Number", value=current_phone)
                with col_gr2:
                    new_dob = st.date_input("Date of Birth", value=default_date, min_value=min_dob, max_value=max_dob)
                    new_bio = st.text_area("Bio / About", value=st.session_state.user_bio, height=73)
                st.write("")
                if st.form_submit_button("Save Changes", type="primary", use_container_width=True):
                    user_obj["email"] = new_email
                    user_obj["bio"] = new_bio
                    user_obj["phone"] = new_phone
                    user_obj["dob"] = new_dob.isoformat()
                    users[st.session_state.username] = user_obj
                    if save_users(users):
                        st.session_state.user_email = new_email
                        st.session_state.user_bio = new_bio
                        st.toast("Profile updated successfully!", icon="✅")
                    else:
                        st.error("Failed to save.")
            st.markdown("</div>", unsafe_allow_html=True)

        with acc_tab2:
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            hist = user_obj.get("history", [])
            c_h1, c_h2 = st.columns([4, 1])
            with c_h1:
                st.markdown(f"**Total Records:** {len(hist)}")
            with c_h2:
                if st.button("Clear History", use_container_width=True):
                     if st.session_state.username in users:
                         users[st.session_state.username]["history"] = []
                         save_users(users)
                         st.rerun()

            if hist:
                st.dataframe(
                    hist,
                    use_container_width=True,
                    column_config={
                        "timestamp": st.column_config.TextColumn("Date/Time"),
                        "prediction": st.column_config.TextColumn("Text"),
                        "translation": "Trans",
                        "video_path": st.column_config.TextColumn("Saved File (Path)")
                    }
                )
            else:
                st.info("No activity recorded yet.")
            st.markdown("</div>", unsafe_allow_html=True)

    # Admin users management
    elif st.session_state.view_mode == "admin_users":
        if not is_admin_user():
            st.error("Admin access required.")
            if st.button("Back"):
                st.session_state.view_mode = "app"
                st.rerun()
        else:
            st.header("Admin — Manage Users")
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            users = load_users()
            for uname in list(users.keys()):
                if isinstance(users[uname], str):
                    users[uname] = {"password": users[uname], "history": [], "is_admin": False}
            save_users(users)
            st.markdown(f"**Total users:** {len(users)}")
            st.write("")

            # New: show pending password-reset requests
            st.markdown("### 🔔 Password reset requests")
            pending = get_admin_reset_requests()
            if not pending:
                st.info("No pending admin reset requests.")
            else:
                for req_user in pending:
                    cols = st.columns([3,1,1])
                    with cols[0]:
                        st.write(f"**{req_user}** requested an admin reset.")
                    with cols[1]:
                        if st.button(f"Reset & Show Temp##reset_{req_user}", key=f"reset_show_{req_user}"):
                            ok, val = admin_reset_user_with_temp_password(req_user)
                            if ok:
                                st.success(f"Password reset for {req_user}. Temporary password:")
                                st.code(val)  # admin copies and shares this with the user
                            else:
                                st.error(f"Reset failed: {val}")
                            st.rerun()
                    with cols[2]:
                        if st.button(f"Cancel Request##cancel_{req_user}", key=f"cancel_req_{req_user}"):
                            clear_admin_reset_request(req_user)
                            st.success(f"Cleared request for {req_user}")
                            st.rerun()
            st.markdown("---")

            for uname, uobj in users.items():
                is_admin_flag = False
                if isinstance(uobj, dict):
                    is_admin_flag = bool(uobj.get("is_admin", False))
                header_label = f"{uname} {'(admin)' if (uname=='admin' or is_admin_flag) else ''}"
                with st.expander(header_label):
                    if isinstance(uobj, dict):
                        st.write("Email:", uobj.get("email", ""))
                        st.write("Bio:", uobj.get("bio", ""))
                        st.write("Phone:", uobj.get("phone", ""))
                        st.write("DOB:", uobj.get("dob", ""))
                        st.write("History count:", len(uobj.get("history", [])))
                        st.write("Admin reset requested:", bool(uobj.get("admin_reset_requested", False)))
                    else:
                        st.write("Legacy account (no metadata).")
                    cols = st.columns([1,1,1,1])
                    with cols[0]:
                        if st.button(f"Promote to admin {uname}", key=f"prom_{uname}"):
                            admin_set_user_flag(uname, "is_admin", True)
                            st.success(f"{uname} promoted to admin.")
                            st.rerun()
                    with cols[1]:
                        if st.button(f"Demote admin  {uname}", key=f"dem_{uname}"):
                            if uname == st.session_state.username:
                                st.warning("You cannot demote yourself while logged in.")
                            else:
                                admin_set_user_flag(uname, "is_admin", False)
                                st.success(f"{uname} demoted.")
                                st.rerun()
                    with cols[2]:
                        if st.button(f"Clear history  {uname}", key=f"clearhist_{uname}"):
                            if isinstance(users[uname], dict):
                                users[uname]["history"] = []
                                save_users(users)
                                st.success("History cleared.")
                                st.rerun()
                    with cols[3]:
                        if st.button(f"Delete user  {uname}", key=f"del_{uname}"):
                            st.session_state["_admin_action_confirm"] = {"action": "delete_user", "target": uname}
            st.markdown("</div>", unsafe_allow_html=True)

            conf = st.session_state.get("_admin_action_confirm", {})
            if conf.get("action") == "delete_user":
                target = conf.get("target")
                st.warning(f"Confirm deletion of user '{target}'? This cannot be undone.")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Confirm Delete User", key="confirm_delete_user"):
                        users = load_users()
                        if target in users:
                            if target == st.session_state.username:
                                st.error("You cannot delete your own currently-logged-in admin account.")
                            else:
                                try:
                                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    shutil.copy2(USERS_FILE, BACKUPS_DIR / f"users_backup_{ts}.json")
                                except Exception:
                                    pass
                                users.pop(target, None)
                                save_users(users)
                                st.toast(f"Deleted user {target}", icon="✅")
                        else:
                            st.error("User not found.")
                        st.session_state["_admin_action_confirm"] = {}
                        st.rerun()
                with c2:
                    if st.button("Cancel", key="cancel_delete_user"):
                        st.session_state["_admin_action_confirm"] = {}

            if st.button("Back to App"):
                st.session_state.view_mode = "app"
                st.rerun()

    # App tools view (main functionality)
    else:
        # Discover models once
        MODEL_OPTIONS = discover_models()

        # Dynamic tabs: show Model Manager tab only to admins
        tab_labels = ["🎥 Prediction Studio"]
        show_model_tab = is_admin_user()
        if show_model_tab:
            tab_labels.append("📂 Model Manager")
        tab_labels += ["🌐 Smart Translator", "📥 Export Results"]

        tabs = st.tabs(tab_labels)
        # map tab objects
        tab_index = 0
        tab1 = tabs[tab_index]; tab_index += 1
        tab2 = None
        if show_model_tab:
            tab2 = tabs[tab_index]; tab_index += 1
        tab3 = tabs[tab_index]; tab_index += 1
        tab4 = tabs[tab_index]

        # TAB 1: Prediction Studio (same logic)
        with tab1:
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            c_input, c_out = st.columns([1, 1.3], gap="large")
            
            with c_input:
                st.markdown("#### 1. Upload Video")
                st.markdown("<div class='upload-zone'>", unsafe_allow_html=True)
                video_file = st.file_uploader("", type=["mp4","avi","mov","mpeg4"], label_visibility="collapsed")
                if not video_file:
                     st.markdown("<small style='color:#94A3B8;'>Drag and drop file here<br>Limit 200MB per file • MP4, AVI, MOV, MPEG4<br>Supported: MP4, AVI, MOV. Max 10s.</small>", unsafe_allow_html=True)
                else:
                    st.caption("✅ File selected: " + video_file.name)
                st.markdown("</div>", unsafe_allow_html=True)

                if video_file:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
                    tfile.write(video_file.read())
                    tfile.flush()
                    temp_vid_path = tfile.name
                    tfile.close()
                    st.video(temp_vid_path)
                elif st.session_state.get("_last_video_path") and os.path.exists(st.session_state["_last_video_path"]):
                    st.markdown("**(Last Analyzed Video)**")
                    st.video(st.session_state["_last_video_path"])

            with c_out:
                st.write("#### 2. Analysis & Output")
                sel_mod, sel_lang = st.columns([2, 1])
                with sel_mod:
                    model_choice = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
                with sel_lang:
                    lang_choice = st.selectbox("Translate", ["None"] + list(AVAILABLE_LANGS.keys()))
                
                predict_btn = st.button("Start Analysis", type="primary", use_container_width=True, disabled=not video_file)

                if video_file and predict_btn and MODEL_OPTIONS.get(model_choice):
                    model_path = MODEL_OPTIONS[model_choice]
                    with st.status("Processing...", expanded=True) as status:
                        try:
                            video_file.seek(0)
                            unique_name = f"{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                            permanent_save_path = UPLOADED_VIDEO_DIR / unique_name
                            with open(permanent_save_path, "wb") as f_perm:
                                f_perm.write(video_file.read())
                            
                            st.session_state["_last_video_path"] = str(permanent_save_path)
                            st.write("Reading video stream...")
                            loaded_model = cached_load_model(model_path)
                            input_data = preprocess_video(temp_vid_path)
                            st.write("Inferencing speech patterns...")
                            preds = loaded_model.predict(input_data)
                            res_label = map_prediction_to_label(preds)
                            
                            # Emotion detection removed; no emotion text
                            
                            tr_txt = "—"
                            if lang_choice != "None":
                                st.write(f"Translating to {lang_choice}...")
                                t_val = get_translation(res_label, lang_choice)
                                if t_val == "Translation not available":
                                    web_try = try_online_translate(res_label, lang_choice)
                                    if web_try and isinstance(web_try, str) and not web_try.startswith("__ERR__:"):
                                        t_val = web_try + " (Web)"
                                    elif web_try and isinstance(web_try, str) and web_try.startswith("__ERR__:"):
                                        # leave t_val as the offline text (Translation not available) but log
                                        logging.info(f"Translation API error for '{res_label}' -> {web_try}")
                                tr_txt = t_val

                            st.session_state["_prediction_done"] = True
                            st.session_state["_last_prediction"] = res_label
                            # emotion removed - do not set _last_emotion
                            st.session_state["_last_translation"] = tr_txt
                            
                            usrs = load_users()
                            if st.session_state.username in usrs:
                                uobj = usrs[st.session_state.username]
                                if isinstance(uobj, str): uobj = {"password": uobj, "history": []}
                                if "history" not in uobj: uobj["history"] = []
                                uobj["history"].insert(0, {
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "prediction": res_label,
                                    "translation": tr_txt,
                                    "video_path": str(permanent_save_path)
                                })
                                save_users(usrs)
                            
                            st.session_state["_tts_pred_bytes"] = text_to_speech_bytes(res_label, "en") if TTS_AVAILABLE else None
                            st.session_state["_tts_trans_bytes"] = text_to_speech_bytes(tr_txt, AVAILABLE_LANGS.get(lang_choice, "en")) if (TTS_AVAILABLE and tr_txt != "—") else None

                            status.update(label="Complete!", state="complete", expanded=False)
                            os.remove(temp_vid_path)

                        except Exception as e:
                            st.error(f"Error: {e}")

                if st.session_state.get("_prediction_done"):
                    st.divider()

                    col_speech, col_trans = st.columns([1,1], gap="large")

                    with col_speech:
                        st.markdown(
                            f"""
                            <div style='padding:12px; border-radius:10px; background:#14532d; color:white; font-weight:600;'>
                                Speech: {st.session_state['_last_prediction']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if st.session_state.get("_tts_pred_bytes"):
                            st.audio(st.session_state["_tts_pred_bytes"], format="audio/mp3")

                    with col_trans:
                        st.markdown(
                            f"""
                            <div style='padding:12px; border-radius:10px; background:#4d4a14; color:white; font-weight:600;'>
                                Transl: {st.session_state['_last_translation']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if st.session_state.get("_tts_trans_bytes"):
                            st.audio(st.session_state["_tts_trans_bytes"], format="audio/mp3")

                    if st.button("Start New Analysis", type="secondary"):
                        clear_prediction()
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # TAB 2: Models (only if admin)
        if tab2 is not None:
            with tab2:
                col_m1, col_m2 = st.columns([1, 2])
                with col_m1:
                    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
                    st.write("##### Upload Model")
                    st.caption("Files (.h5 / .keras)")
                    u_model = st.file_uploader("", type=["h5", "keras"], key="mod_up", label_visibility="collapsed")
                    if u_model:
                        mp = MODELS_DIR / u_model.name
                        with open(mp, "wb") as f: f.write(u_model.read())
                        st.toast("Model Uploaded!", icon="✅")
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_m2:
                    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
                    st.write("##### Installed Models")
                    if not MODEL_OPTIONS:
                        st.warning("No models found in folder.")
                    else:
                        for k, v in MODEL_OPTIONS.items():
                            st.code(k)
                    st.markdown("</div>", unsafe_allow_html=True)

        # TAB 3: Translator (API-only manual translation)
        with tab3:
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            c_txt, c_opt = st.columns([3, 1])

            with c_txt:
                manual_text = st.text_area("Input Text", height=100)

            with c_opt:
                manual_lang = st.selectbox("Language", list(AVAILABLE_LANGS.keys()), key="man_l")
                st.write("")
                if st.button("Translate", use_container_width=True):
                    if not manual_text.strip():
                        st.warning("Please enter some text to translate.")
                    else:
                        api_result = try_online_translate(manual_text, manual_lang)

                        if isinstance(api_result, str) and api_result.startswith("__ERR__:"):
                            st.session_state["_last_man_trans"] = "Translation Error: " + api_result[8:]
                            st.session_state["_tts_manual_bytes"] = None
                        else:
                            st.session_state["_last_man_trans"] = api_result
                            lang_code = AVAILABLE_LANGS.get(manual_lang, "en")
                            if TTS_AVAILABLE:
                                st.session_state["_tts_manual_bytes"] = text_to_speech_bytes(api_result, lang_code)
                            else:
                                st.session_state["_tts_manual_bytes"] = None

            if "_last_man_trans" in st.session_state:
                st.info(f"Result: {st.session_state['_last_man_trans']}")
                if st.session_state.get("_tts_manual_bytes"):
                    st.audio(st.session_state["_tts_manual_bytes"], format="audio/mp3")

            st.divider()
            with st.expander("Add Custom Word to Dictionary"):
                cx1, cx2, cx3 = st.columns([1, 2, 1])
                with cx1:
                    cl = st.selectbox("Lang", list(AVAILABLE_LANGS.keys()), key="cl_k")
                with cx2:
                    ct = st.text_input("Correct Word", key="ct_k")
                with cx3:
                    st.write("")
                    if st.button("Save", use_container_width=True):
                        if manual_text and ct:
                            save_custom_translation(manual_text.strip(), cl, ct.strip())
                            st.success("Saved!")
            st.markdown("</div>", unsafe_allow_html=True)

        # TAB 4: Export & Email
        with tab4:
             st.markdown("<div class='content-card'>", unsafe_allow_html=True)
             st.subheader("Results Export")
             pred_t = st.session_state.get('_last_prediction', '—')
             tra_t = st.session_state.get('_last_translation', '—')
             rpt = f"""LipRead Pro Analysis Report
=============================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
User: {st.session_state.username}

Prediction: {pred_t}
Translation: {tra_t}
=============================
"""
             st.text_area("Text Preview", rpt, height=150)

             st.write("###### 1. Download Assets")
             col_d1, col_d2, col_d3 = st.columns(3)
             with col_d1:
                 st.download_button("📄 Report (.txt)", rpt, "lipread_report.txt", use_container_width=True)
             with col_d2:
                 aud_pred = st.session_state.get("_tts_pred_bytes")
                 if aud_pred:
                     st.download_button("🔊 Audio (Orig)", aud_pred, file_name="prediction_audio.mp3", mime="audio/mpeg", use_container_width=True)
                 else:
                     st.button("No Audio", disabled=True, use_container_width=True, key="no_aud_1")
             with col_d3:
                 aud_trans = st.session_state.get("_tts_trans_bytes")
                 if aud_trans:
                     st.download_button("🔊 Audio (Trans)", aud_trans, file_name="translation_audio.mp3", mime="audio/mpeg", use_container_width=True)
                 else:
                     st.button("No Translation", disabled=True, use_container_width=True, key="no_aud_2")
             st.write("###### 2. Share")
             subject = urllib.parse.quote("LipRead Analysis Result")
             email_body = rpt + "\n\n(Note: Please attach the downloaded audio files manually)"
             encoded_body = urllib.parse.quote(email_body)
             mailto_link = f"mailto:?subject={subject}&body={encoded_body}"
             st.link_button("📧 Open Email Draft (Attach files manually if needed)", mailto_link, use_container_width=True)
             st.markdown("</div>", unsafe_allow_html=True)

# Entry point
if not st.session_state.logged_in:
    show_auth_page()
else:
    show_main_page()
