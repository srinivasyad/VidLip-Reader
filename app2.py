# app.py — Modified UI per your request (final corrected)
# - Brand centered near top (small margin)
# - Form submit buttons large and full-width, default color green
# - All buttons change to light-blue on hover
# - Accent palette refreshed; logic unchanged

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

# translations module
from utils import translations as translations_mod
from utils.translations import get_translation, AVAILABLE_LANGS

# ------------------ Paths & setup ------------------
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
USERS_FILE = ROOT / "users.json"
REMEMBER_FILE = ROOT / "remember.json"
TRANSLATIONS_CUSTOM = ROOT / "utils" / "translations_custom.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "utils").mkdir(parents=True, exist_ok=True)
if not TRANSLATIONS_CUSTOM.exists():
    TRANSLATIONS_CUSTOM.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------ Users utilities ------------------
def load_users():
    if not USERS_FILE.exists(): return {}
    try:
        return json.load(open(USERS_FILE, "r", encoding="utf-8"))
    except Exception:
        return {}

def save_users(users):
    try:
        json.dump(users, open(USERS_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

def load_remember():
    if not REMEMBER_FILE.exists(): return {"remember": False, "username": ""}
    try:
        return json.load(open(REMEMBER_FILE, "r", encoding="utf-8"))
    except Exception:
        return {"remember": False, "username": ""}

def save_remember(username, remember):
    try:
        json.dump({"remember": bool(remember), "username": username if remember else ""}, open(REMEMBER_FILE, "w", encoding="utf-8"), indent=2)
    except Exception:
        pass

def hash_password(pw):
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def ensure_default_admin():
    users = load_users()
    if "admin" not in users:
        users["admin"] = hash_password("password")
        save_users(users)
ensure_default_admin()

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
    try:
        from googletrans import Translator
    except Exception:
        return None
    try:
        t = Translator()
        code = AVAILABLE_LANGS.get(target_lang)
        if not code: return None
        res = t.translate(text, dest=code)
        return res.text
    except Exception:
        return None

# ------------------ Models & prediction ------------------
def discover_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    candidates += glob.glob(os.path.join(script_dir, "models", "*.keras"))
    candidates += glob.glob(os.path.join(script_dir, "models", "*.h5"))
    candidates += glob.glob(os.path.join(script_dir, "..", "models", "*.keras"))
    candidates += glob.glob(os.path.join(script_dir, "..", "models", "*.h5"))
    candidates += glob.glob(r"C:\mnt\data\*.keras")
    candidates += glob.glob(r"C:\mnt\data\*.h5")
    candidates += glob.glob("/mnt/data/*.keras")
    candidates += glob.glob("/mnt/data/*.h5")
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
    if not path: return None
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

# ---------- Streamlit setup & session ----------
st.set_page_config(page_title="Lip Reading App", layout="wide", initial_sidebar_state="expanded")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "_prediction_done" not in st.session_state:
    st.session_state["_prediction_done"] = False
if "_last_prediction" not in st.session_state:
    st.session_state["_last_prediction"] = ""
if "_last_translation" not in st.session_state:
    st.session_state["_last_translation"] = ""

load_and_merge_custom_translations()

# minimal CSS from your improved UI, with requested visual changes
st.markdown("""
    <style>
        :root{--glass: rgba(255,255,255,0.06); --accent-a:#16a34a; --accent-b:#10b981;}
        body { background: linear-gradient(135deg,#f7fbff 0%,#eef6ff 100%); color:#071130; font-family:'Inter',sans-serif; }
        .main-app{background:rgba(6,8,15,0.02); border-radius:14px; padding:24px;}
        .card{background:#ffffff; border-radius:12px; padding:16px; margin-bottom:12px; box-shadow: 0 6px 18px rgba(2,6,23,0.04);}
        .big-title{font-size:28px; font-weight:700; color:#071130;}
        .sep { border-left: 3px solid rgba(7,17,48,0.06); padding-left:12px; margin-bottom:12px; }

        /* BRAND centered with small top margin to avoid huge empty space */
        .brand-centered { text-align:center; margin-top:2cm; }
        .brand-centered .brand { display:inline-block; padding:10px 18px; border-radius:12px; background: linear-gradient(90deg,var(--accent-a),var(--accent-b)); color: white; font-weight:800; font-size:16px; box-shadow: 0 10px 30px rgba(16,185,129,0.12); }

        /* Auth card visual for light mode */
        .auth-wrap { display:flex; justify-content:center; align-items:flex-start; padding-top:26px; padding-bottom:26px; }
        .auth-card { width: 820px; border-radius: 12px; background: #ffffff; border: 1px solid rgba(16,24,40,0.06); box-shadow: 0 12px 36px rgba(2,6,23,0.06); padding: 22px; color: #071130; font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }

        .title { font-size:22px; font-weight:700; margin-bottom:6px; color:#071130; }
        .subtitle { color: #65748b; margin-bottom:18px; font-size:14px; }
        .side-card { background:#fbfdff; border-radius:8px; padding:12px; border:1px solid rgba(16,24,40,0.04); }
        .muted-small { color:#65748b; font-size:13px; }

        /* Inputs: slightly rounded and subtle background */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            border-radius:8px !important; background: #f6f8fb !important; color: #071130 !important;
        }

        /* PRIMARY: make form submit buttons large, full-width and green */
        .stForm button, .stButton>button { width:100% !important; display:block !important; padding:14px 18px !important; font-size:16px !important; border-radius:10px !important; background: #22c55e !important; color: #fff !important; border: none !important; box-shadow: 0 10px 28px rgba(34,197,94,0.12) !important; }

        /* Download/secondary buttons slightly smaller but consistent */
        .stDownloadButton>button { width:100% !important; padding:10px 12px !important; border-radius:10px !important; }

        /* Hover behaviour: ALL buttons change to light-blue on hover as requested */
        .stButton>button:hover, .stForm button:hover, .stDownloadButton>button:hover { transform: scale(1.01); background: #ADD8E6 !important; box-shadow: 0 8px 22px rgba(14,52,92,0.06) !important; color: #05243a !important; }

        /* small note text */
        .small-note { font-size:13px; color:#65748b; }

    </style>
""", unsafe_allow_html=True)

# place the brand centered (moved from inside auth card)
st.markdown("<div class='brand-centered'><div class='brand'>LipRead Pro</div></div>", unsafe_allow_html=True)

# ------------------ AUTH page (unchanged logic but visual tuned) ------------------
def show_auth_page():
    """
    Clean, professional light-mode login / signup UI.
    Keeps existing helper functions: load_users, save_users, load_remember,
    save_remember, hash_password, save_users, etc.
    """

    st.markdown("<div class='auth-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)

    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.markdown("<div class='title'>Welcome — Lip Reading App</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Sign in to predict words from short videos, translate them into local languages, and share results. Account details are stored locally in <code>users.json</code>.</div>", unsafe_allow_html=True)

        # load remembered username if any
        remembered = load_remember()
        default_username = remembered.get("username", "") if remembered.get("remember", False) else ""

        mode = st.radio("Mode", ("Login", "Sign up"), horizontal=True)

        if mode == "Login":
            with st.form("login_form_light", clear_on_submit=False):
                c1, c2 = st.columns(2)
                username = c1.text_input("Username", value=default_username, placeholder="e.g. srinivas")
                password = c2.text_input("Password", type="password", placeholder="Your password")
                remember = st.checkbox("Remember username on this device", value=remembered.get("remember", False))
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Login")
                if submitted:
                    users = load_users()
                    if username in users and users[username] == hash_password(password):
                        st.success("Login successful ✅")
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        save_remember(username, remember)
                        st.rerun()
                    else:
                        st.error("Invalid username or password. If you don't have an account, choose Sign up.")
        else:
            with st.form("signup_form_light", clear_on_submit=False):
                new_user = st.text_input("Choose a username", placeholder="unique username")
                new_pass = st.text_input("Choose a password", type="password")
                confirm = st.text_input("Confirm password", type="password")
                remember_after = st.checkbox("Remember username after sign up", value=False)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                created = st.form_submit_button("Create account")
                if created:
                    if not new_user or not new_pass:
                        st.warning("Username and password cannot be empty.")
                    elif new_pass != confirm:
                        st.error("Passwords do not match.")
                    else:
                        users = load_users()
                        if new_user in users:
                            st.error("Username already exists — pick another.")
                        else:
                            users[new_user] = hash_password(new_pass)
                            ok = save_users(users)
                            if ok:
                                st.success("Account created 🎉 — you are now logged in.")
                                save_remember(new_user, remember_after)
                                st.session_state.logged_in = True
                                st.session_state.username = new_user
                                st.rerun()
                            else:
                                st.error("Failed to save account (file permissions?), try again.")

    with right_col:
        st.markdown("<div class='side-card'>", unsafe_allow_html=True)
        st.markdown("### Quick Actions", unsafe_allow_html=True)
        users = load_users()
        st.download_button(
            "Export users backup",
            data=json.dumps(users, indent=2, ensure_ascii=False),
            file_name="users.json",
            mime="application/json",
            help="Download a local backup of users.json for safekeeping.",
            use_container_width=True,
        )
        st.markdown("<div class='small-note' style='margin-top:10px;'>Tips:<br>• Use a strong password for production.<br>• Back up users.json regularly.<br>• This demo stores login data locally only.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # small decorative image (optional). If offline, comment this out.
        try:
            st.image("https://images.unsplash.com/photo-1556157382-97eda2d62296?q=80&w=400&auto=format&fit=crop", use_container_width=True)
        except Exception:
            pass

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ MAIN page with visually separated Translate tab ------------------
def show_main_page():
    with st.sidebar:
        st.markdown(f"**Signed in as:** {st.session_state.username}")
        if st.button("🔒 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state["_prediction_done"] = False
            st.session_state["_last_prediction"] = ""
            st.session_state["_last_translation"] = ""
            st.rerun()
        st.markdown("---")
        if st.button("Reload translations"):
            ok, msg = reload_translations_module()
            if ok: st.success(msg)
            else: st.error(msg)
        st.markdown("Custom translations file:")
        st.code(str(TRANSLATIONS_CUSTOM))

    st.markdown("<div class='main-app'>", unsafe_allow_html=True)
    st.markdown("<div class='big-title'> Lip Reading & Translation</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>Predict spoken words from short videos, translate to local languages, and share results.</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Models", "Translate", "Share & Export"])
    MODEL_OPTIONS = discover_models()

    # PREDICT tab (unchanged)
    with tab1:
        left, right = st.columns([2,1])
        with left:
            st.header("Upload & Predict")
            video_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])
            model_choice = st.selectbox("Choose Model", list(MODEL_OPTIONS.keys()))
            model_path = MODEL_OPTIONS.get(model_choice)
            lang_choice = st.selectbox("Translate Prediction To", ["None"] + list(AVAILABLE_LANGS.keys()))
            if video_file:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(video_file.read()); tmp.flush(); tmp_path = tmp.name; tmp.close()
                try:
                    data = open(tmp_path, "rb").read(); b64 = base64.b64encode(data).decode()
                    st.markdown(f"<video width='420' controls><source src='data:video/mp4;base64,{b64}' type='video/mp4'></video>", unsafe_allow_html=True)
                except Exception:
                    st.warning("Preview not available.")
                if not model_path:
                    st.error("No model selected.")
                else:
                    st.info("Loading model...")
                    model_obj = cached_load_model(model_path)
                    if model_obj is None:
                        st.error("Model failed to load.")
                    else:
                        try:
                            input_tensor = preprocess_video(tmp_path)
                            preds = model_obj.predict(input_tensor)
                            predicted_label = map_prediction_to_label(preds)
                            st.success("Prediction complete.")
                            st.markdown(f"**English Prediction:** `{predicted_label}`")
                            st.session_state["_prediction_done"] = True
                            st.session_state["_last_prediction"] = predicted_label
                            if lang_choice != "None":
                                trans = get_translation(predicted_label, lang_choice)
                                if trans == "Translation not available":
                                    online = try_online_translate(predicted_label, lang_choice)
                                    if online:
                                        trans = online + " (online)"
                                st.markdown(f"**{lang_choice} Translation:** `{trans}`")
                                st.session_state["_last_translation"] = trans
                            else:
                                st.session_state["_last_translation"] = "—"
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        with right:
            st.header("Notes")
            st.write("- After a prediction, you'll see it shown in the Translate tab above the manual translator.")
            st.write("- Manual Translation is independent and always available below that card.")

    # MODELS tab (same)
    with tab2:
        st.header("Models & Upload")
        uploaded_model = st.file_uploader("Upload model (.keras/.h5)", type=["keras","h5"], key="model_upload")
        if uploaded_model:
            dest = MODELS_DIR / uploaded_model.name
            with open(dest, "wb") as f: f.write(uploaded_model.read())
            st.success(f"Saved model to {dest}")
            MODEL_OPTIONS = discover_models()
        st.markdown("**Available models**")
        for m in MODEL_OPTIONS: st.write(f"- {m}")

    # TRANSLATE tab — visually separated
    with tab3:
        st.header("Translate — Prediction (top) and Manual (below)")

        # TOP: show last prediction (if any) in a distinct card
        if st.session_state.get("_prediction_done", False):
            st.markdown("<div class='card sep'>", unsafe_allow_html=True)
            st.subheader("Last model prediction")
            last_pred = st.session_state.get("_last_prediction", "—")
            last_tr = st.session_state.get("_last_translation", "—")
            st.markdown(f"**English (predicted):** `{last_pred}`")
            st.markdown(f"**Translation (predicted):** `{last_tr}`")
            if st.button("Clear last prediction (remove this card)"):
                st.session_state["_prediction_done"] = False
                st.session_state["_last_prediction"] = ""
                st.session_state["_last_translation"] = ""
                st.success("Cleared last prediction.")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No recent prediction to show here.")

        # SEPARATOR visual (manual translator card)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Manual Translation (independent)")

        # OUTER form for translation (single form only)
        with st.form("manual_translate_form", clear_on_submit=False):
            manual_text = st.text_area("Enter text to translate", height=140)
            manual_lang = st.selectbox("Translate to", ["None"] + list(AVAILABLE_LANGS.keys()))
            translate_btn = st.form_submit_button("Translate Now")

            # We will display the translation result inside the form area after submit
            if translate_btn:
                if not manual_text.strip():
                    st.warning("Enter text to translate.")
                elif manual_lang == "None":
                    st.warning("Choose a language.")
                else:
                    res = get_translation(manual_text.strip(), manual_lang)
                    if res == "Translation not available":
                        online = try_online_translate(manual_text.strip(), manual_lang)
                        if online:
                            res = online + " (online)"
                    st.markdown("**Translation result:**")
                    st.code(res)

        # ---- Custom-save UI (NOT a nested form) ----
        st.markdown("---")
        st.write("If translation is missing/incorrect, add an offline translation below:")
        custom_lang = st.selectbox("Language for custom translation", list(AVAILABLE_LANGS.keys()), key="custom_lang")
        custom_text = st.text_input("Custom translation text", key="custom_text")
        if st.button("Save custom translation"):
            if not manual_text.strip():
                st.error("No source text to attach the custom translation to. Type text in the translator above first.")
            elif not custom_text.strip():
                st.error("Enter translation text to save.")
            else:
                ok = save_custom_translation(manual_text.strip(), custom_lang, custom_text.strip())
                if ok:
                    st.success("Saved custom translation. Click 'Reload translations' in sidebar if needed.")
                else:
                    st.error("Failed to save custom translation (check file permissions).")

        st.markdown("</div>", unsafe_allow_html=True)

    # SHARE tab (unchanged)
    with tab4:
        st.header("Share & Export")
        pred = st.session_state.get("_last_prediction", "—")
        tr = st.session_state.get("_last_translation", "—")
        st.markdown(f"**Last English Prediction:** `{pred}`")
        st.markdown(f"**Last Translation:** `{tr}`")
        share_text = f"Lip Reading Result:\nEnglish: {pred}\nTranslation: {tr}"
        st.text_area("Result text", value=share_text, height=140)
        st.download_button("Download result", data=share_text.encode("utf-8"), file_name="lip_reading_result.txt", mime="text/plain")
        mailto = f"mailto:?subject=Lip Reading Result&body={urllib.parse.quote(share_text)}"
        # fixed broken line: render mailto link correctly
        st.markdown(f"[📧 Share via Email]({mailto})")

    st.markdown("</div>", unsafe_allow_html=True)

# Entry point
if not st.session_state.logged_in:
    show_auth_page()
else:
    show_main_page()
