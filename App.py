import streamlit as st
import pandas as pd
from PIL import Image
import os
import google.generativeai as genai
from datetime import datetime
from gtts import gTTS
import io
import hashlib

# --- 1. إعدادات الأمان والذكاء الاصطناعي ---

# قراءة المفتاح من secrets فقط (بدون وضع مفتاح مكشوف داخل الكود)
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = None

if API_KEY:
    genai.configure(api_key=API_KEY)

@st.cache_resource
def load_ai_model():
    # تم حذف -latest لأنه يسبب خطأ 404
    return genai.GenerativeModel("gemini-1.5-flash")

def get_ai_response(prompt, image=None):
    try:
        if not API_KEY:
            return "⚠️ لم يتم العثور على GEMINI_API_KEY في secrets."

        model = load_ai_model()

        if image:
            response = model.generate_content([prompt, image])
        else:
            response = model.generate_content(prompt)

        if hasattr(response, "text") and response.text:
            return response.text
        else:
            return "⚠️ لم يتم إرجاع نص من النموذج."

    except Exception as e:
        return f"⚠️ عذراً، هناك مشكلة في الاتصال بالمعلم الذكي. (Error: {str(e)})"

# دالة التشفير لحماية كلمات المرور
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# دالة المعلم الناطق
def speak_text(text):
    try:
        clean_text = text[:250].replace("*", "").replace("#", "").replace("-", "")
        tts = gTTS(text=clean_text, lang='ar')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except:
        return None

# --- 2. نظام المجلدات وقواعد البيانات ---
for folder in ['lessons', 'exams', 'db']:
    os.makedirs(folder, exist_ok=True)

USERS_DB = "db/users.csv"
FILES_DB = "db/files.csv"
GRADES_DB = "db/grades.csv"

def init_db(path, columns):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)

init_db(USERS_DB, ["user", "pass", "role", "grade"])
init_db(FILES_DB, ["name", "grade", "sub", "type", "date"])
init_db(GRADES_DB, ["user", "sub", "score", "date"])

def load_data(path):
    try:
        return pd.read_csv(path)
    except:
        if "users" in path:
            return pd.DataFrame(columns=["user", "pass", "role", "grade"])
        if "files" in path:
            return pd.DataFrame(columns=["name", "grade", "sub", "type", "date"])
        return pd.DataFrame(columns=["user", "sub", "score", "date"])

# --- 3. إدارة الجلسة ---
if "user_data" not in st.session_state:
    st.session_state["user_data"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- 4. الوقت والثيم ---
hour = datetime.now().hour
if 5 <= hour < 18:
    greeting, bg, txt, card = "☀️ صباح الخير", "#F0F2F6", "#000000", "#FFFFFF"
else:
    greeting, bg, txt, card = "🌙 ليلة سعيدة", "#0E1117", "#FFFFFF", "#262730"

st.set_page_config(page_title="منصة حسام الذكية", layout="wide")

st.markdown(f"""
<style>
.stApp {{ background-color: {bg}; color: {txt}; }}
.stButton>button {{
    width: 100%; border-radius: 12px; height: 3.5em;
    background: linear-gradient(45deg, #D32F2F, #B71C1C);
    color: white; font-weight: bold; border: none;
}}
.greeting-box {{
    padding: 20px; background-color: {card}; border-radius: 15px;
    border: 1px solid #D32F2F; text-align: center; margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}}
.plan-box {{
    background-color: #fdf2f2; border-right: 5px solid #D32F2F;
    padding: 15px; border-radius: 8px; color: black; margin-top: 10px; white-space: pre-wrap;
}}
</style>
""", unsafe_allow_html=True)

subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "إنكليزي", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "عربي"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "فرنسي", "عربي"]
}

# --- 5. منطق الدخول ---
if st.session_state["user_data"] is None:

    st.markdown(f'<div class="greeting-box"><h1>{greeting}</h1><p>أهلاً بك في منصة حسام التعليمية المطورة</p></div>', unsafe_allow_html=True)

    t_log, t_sign = st.tabs(["🔐 تسجيل الدخول", "📝 إنشاء حساب"])

    with t_log:
        u = st.text_input("اسم المستخدم", key="login_u")
        p = st.text_input("كلمة المرور", type="password", key="login_p")

        if st.button("دخول المنصة"):
            if u == "Hosam" and p == "Anahosam031007":
                st.session_state["user_data"] = {"user": u, "role": "Owner", "grade": "الكل"}
                st.rerun()
            else:
                users = load_data(USERS_DB)
                hashed_p = hash_password(p)
                match = users[(users["user"] == u) & (users["pass"] == hashed_p)]

                if not match.empty:
                    st.session_state["user_data"] = match.iloc[0].to_dict()
                    st.rerun()
                else:
                    st.error("عذراً، البيانات غير صحيحة")

    with t_sign:
        nu = st.text_input("الاسم الكامل")
        np = st.text_input("كلمة السر", type="password")
        nr = st.selectbox("أنا:", ["طالب", "أستاذ"])
        ng = st.selectbox("الصف:", list(subs_map.keys())) if nr == "طالب" else "الكل"

        if st.button("تأكيد إنشاء الحساب"):
            if nu and np:
                users = load_data(USERS_DB)
                if nu in users['user'].values:
                    st.error("الاسم موجود مسبقاً")
                else:
                    new_user = pd.DataFrame([{
                        "user": nu,
                        "pass": hash_password(np),
                        "role": nr,
                        "grade": ng
                    }])
                    pd.concat([users, new_user]).to_csv(USERS_DB, index=False)
                    st.success("تم بنجاح! سجل دخولك الآن")
