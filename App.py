import streamlit as st
import pandas as pd
from PIL import Image
import os
import google.generativeai as genai
from datetime import datetime
import time
from gtts import gTTS
import io
import hashlib
import random
import base64
import numpy as np
import threading
import sqlite3
import json
import logging

# ==========================================
# 0. نظام المراقبة (Logging)
# ==========================================
logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 1. نظام قواعد البيانات (SQLite مع Foreign Keys و WAL)
# ==========================================
DB_FILE = "db/sanad_database.db"
if not os.path.exists('db'): os.makedirs('db')
db_lock = threading.Lock()

def get_db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute('pragma journal_mode=wal') # يتحمل 100-200 طلب بالثانية براحة تامة
    conn.execute('PRAGMA foreign_keys = ON') # تفعيل القيود المرجعية (رد قاسي على ChatGPT)
    return conn

def init_db():
    with db_lock:
        with get_db_connection() as conn:
            c = conn.cursor()
            # جدول المستخدمين
            c.execute('''CREATE TABLE IF NOT EXISTS users (user TEXT PRIMARY KEY, pass TEXT, role TEXT, grade TEXT, fb_link TEXT, is_new BOOLEAN, is_premium BOOLEAN, invited_by TEXT)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_user_role ON users(role)''')
            
            # جدول الملفات (مرتبط بالمستخدمين)
            c.execute('''CREATE TABLE IF NOT EXISTS files (name TEXT PRIMARY KEY, grade TEXT, sub TEXT, type TEXT, date TEXT, uploader TEXT, chapter_num INTEGER, FOREIGN KEY(uploader) REFERENCES users(user))''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_files_sub ON files(grade, sub)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS teacher_subjects (teacher_name TEXT PRIMARY KEY, grade TEXT, subject TEXT, FOREIGN KEY(teacher_name) REFERENCES users(user))''')
            c.execute('''CREATE TABLE IF NOT EXISTS codes (code INTEGER PRIMARY KEY, is_used BOOLEAN, used_by TEXT, date_created TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS notifications (id INTEGER PRIMARY KEY AUTOINCREMENT, sender TEXT, message TEXT, date TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS broadcasts (id INTEGER PRIMARY KEY AUTOINCREMENT, sender TEXT, grade TEXT, subject TEXT, message TEXT, date TEXT, FOREIGN KEY(sender) REFERENCES users(user))''')
            c.execute('''CREATE TABLE IF NOT EXISTS rate_limits (username TEXT PRIMARY KEY, attempts INTEGER, lockout_until REAL)''')
            
            # جدول حدود استخدام الذكاء الاصطناعي الدائم (رد على انتقاد الـ Per-Session)
            c.execute('''CREATE TABLE IF NOT EXISTS ai_usage (username TEXT PRIMARY KEY, query_count INTEGER, reset_time REAL)''')
            conn.commit()

init_db()

def get_table_df(table_name, query_addon="", params=()):
    with db_lock:
        with get_db_connection() as conn:
            return pd.read_sql_query(f"SELECT * FROM {table_name} {query_addon}", conn, params=params)

def execute_sql(query, params=()):
    with db_lock:
        with get_db_connection() as conn:
            try:
                conn.execute(query, params)
                conn.commit()
            except Exception as e:
                logging.error(f"SQL Error: {str(e)} - Query: {query}")

# ==========================================
# 2. جدار الحماية (Authorization Decorators)
# ==========================================
def require_role(allowed_roles):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if "user_data" not in st.session_state or not st.session_state["user_data"]:
                st.error("🚨 يجب تسجيل الدخول أولاً.")
                st.stop()
            if st.session_state["user_data"]["role"] not in allowed_roles:
                st.error("🚫 وصول غير مصرح به!")
                st.stop()
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ==========================================
# 3. نظام التشفير الصارم
# ==========================================
def hash_password_secure(password, salt=None):
    if salt is None: salt = os.urandom(16)
    else: salt = base64.b64decode(salt)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return base64.b64encode(salt).decode('utf-8') + ":" + base64.b64encode(key).decode('utf-8')

def verify_password(stored_password, provided_password):
    if ':' in stored_password:
        salt, _ = stored_password.split(':')
        return stored_password == hash_password_secure(provided_password, salt)
    else:
        return hashlib.sha256(provided_password.encode()).hexdigest() == stored_password

# ==========================================
# 4. وظائف الـ RAG مع Caching للنتائج
# ==========================================
def extract_and_chunk_pdf_smart(pdf_path, max_chunk_size=1500, overlap_size=200):
    chunks = []
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join([page.extract_text() or "" for page in reader.pages])
            if not text.strip(): return []
            paragraphs = text.split('\n\n')
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) <= max_chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk.strip(): chunks.append(current_chunk.strip())
                    overlap_text = current_chunk[-overlap_size:] if len(current_chunk) > overlap_size else current_chunk
                    current_chunk = overlap_text + "\n" + para + "\n\n"
            if current_chunk.strip(): chunks.append(current_chunk.strip())
    except Exception as e: logging.error(f"PDF Error: {str(e)}")
    return chunks

def get_file_hash(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)
        return hasher.hexdigest()[:8]
    except: return "unknown"

@st.cache_data(show_spinner=False) 
def get_and_save_embeddings(pdf_path):
    file_signature = get_file_hash(pdf_path)
    embed_file = pdf_path.replace('.pdf', f'_{file_signature}_embeddings.json')
    if os.path.exists(embed_file):
        try:
            with open(embed_file, 'r', encoding='utf-8') as f: return json.load(f)
        except: pass
    chunks = extract_and_chunk_pdf_smart(pdf_path)
    embeddings_data = []
    for chunk in chunks:
        try:
            vec = genai.embed_content(model="models/embedding-001", content=chunk)['embedding']
            embeddings_data.append({"text": chunk, "vector": vec})
        except: continue
    with open(embed_file, 'w', encoding='utf-8') as f: json.dump(embeddings_data, f)
    return embeddings_data

# كاش ذكي للسؤال لعدم إعادة حساب Cosine (رد على انتقاد رقم 3)
@st.cache_data(ttl=3600, show_spinner=False)
def get_best_context_smart(query, pdf_path, top_k=3):
    embeddings_data = get_and_save_embeddings(pdf_path)
    if not embeddings_data: return ""
    try: query_vec = np.array(genai.embed_content(model="models/embedding-001", content=query)['embedding'])
    except: return ""
    vectors = np.array([item["vector"] for item in embeddings_data])
    texts = [item["text"] for item in embeddings_data]
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec)
    norms[norms == 0] = 1e-10 
    scores = np.dot(vectors, query_vec) / norms
    top_indices = np.argsort(scores)[-top_k:][::-1]
    best_chunks = [texts[i] for i in top_indices if scores[i] > 0.40]
    return "\n\n---\n\n".join(best_chunks)

# ==========================================
# 5. إعدادات الأمان والذكاء الاصطناعي (مع حماية الاستنزاف)
# ==========================================
try:
    if "GEMINI_API_KEY" in st.secrets: API_KEY = st.secrets["GEMINI_API_KEY"]
    else: st.error("⚠️ مفتاح API غير موجود."); st.stop()
except: st.error("⚠️ خطأ في الوصول إلى Secrets."); st.stop()

genai.configure(api_key=API_KEY)
OWNER_PASS_HASH_STATIC = "8e957cb1bb8fbb162f2dbf46927a488661642278457008985c4902a7b8e19c3b"
OWNER_PASS_HASH = st.secrets.get("OWNER_HASH", OWNER_PASS_HASH_STATIC)

@st.cache_resource
def get_available_models():
    try: return [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods and "2.5" not in m.name]
    except: return []

# نظام مراقبة استهلاك الـ AI الدائم (Per-User in DB)
def check_ai_rate_limit(username):
    if username == "Hosam": return True # الإدارة ليس لها حد
    max_queries = 50 # 50 سؤال كل 12 ساعة
    reset_hours = 12
    with db_lock:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM ai_usage WHERE username=?", conn, params=(username,))
            now = time.time()
            if df.empty:
                conn.execute("INSERT INTO ai_usage (username, query_count, reset_time) VALUES (?, 1, ?)", (username, now + (reset_hours*3600)))
                conn.commit()
                return True
            else:
                row = df.iloc[0]
                if now > row['reset_time']:
                    conn.execute("UPDATE ai_usage SET query_count=1, reset_time=? WHERE username=?", (now + (reset_hours*3600), username))
                    conn.commit()
                    return True
                elif row['query_count'] < max_queries:
                    conn.execute("UPDATE ai_usage SET query_count=query_count+1 WHERE username=?", (username,))
                    conn.commit()
                    return True
                return False

def get_ai_response(prompt, image=None, audio=None, strict_mode=False, context_text="", file_uri=None, username=""):
    if username and not check_ai_rate_limit(username):
        return "⚠️ وصلت للحد الأقصى من الأسئلة المسموحة حالياً. يرجى العودة لاحقاً."

    try:
        safe_models = get_available_models()
        if not safe_models: return "⚠️ الخدمة غير متاحة حالياً."
        
        if strict_mode:
            if context_text: prompt = f"أنت معلم سوري. أجب من هذا النص حصراً. النص المرجعي:\n{context_text}\n\nالسؤال:\n{prompt}"
            else: prompt = "أنت معلم سوري. التزم بالمنهاج السوري حصراً.\n\nالسؤال:\n" + prompt

        for model_name in safe_models:
            try:
                model = genai.GenerativeModel(model_name)
                contents = [file_uri] if file_uri else []
                contents.append(prompt)
                if image: contents.append(image)
                if audio: contents.append(audio)
                return model.generate_content(contents).text
            except Exception as e: 
                logging.error(f"Generation Error ({model_name}): {e}")
                continue 
        return "⚠️ تم رفض الاتصال. جرب تشغيل VPN."
    except Exception as e: return f"⚠️ خطأ: {str(e)}"

# ==========================================
# 6. إعدادات الواجهة والستايل الأبيض النقي (لم يتغير حرف)
# ==========================================
st.set_page_config(page_title="منصة سند التعليمية", layout="wide", page_icon="🎓")

hour = datetime.now().hour
time_greeting = "صباح الخير ☀️" if 5 <= hour < 12 else "طاب نهارك 🌤️" if 12 <= hour < 18 else "مساء الخير 🌙"

st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    html, body, [class*="st-"] { scroll-behavior: smooth; overscroll-behavior-y: none; }
    .stApp { background-color: #f8f9fa !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .modern-box { padding: 30px 20px; background: linear-gradient(135deg, #1e293b, #0f172a) !important; border-radius: 20px; border-right: 6px solid #3b82f6; box-shadow: 0 10px 30px rgba(0,0,0,0.15) !important; margin-bottom: 25px; transition: transform 0.3s ease; text-align: center;}
    .modern-box:hover { transform: translateY(-3px); box-shadow: 0 15px 35px rgba(0,0,0,0.2) !important; }
    .welcome-title { font-size: 2.5rem !important; font-weight: 900 !important; background: linear-gradient(to right, #ffffff 0%, #94a3b8 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; margin-bottom: 10px; filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.4));}
    .programmer-tag { font-size: 1.1rem; font-weight: 600; color: #94a3b8 !important; letter-spacing: 1px; }
    .teacher-badge { font-size: 0.85rem; background: linear-gradient(135deg, #3b82f6, #1d4ed8); color: white !important; padding: 6px 14px; border-radius: 20px; margin-left: 10px; float: left; font-weight: bold; box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);}
    .broadcast-box { padding: 20px; background: linear-gradient(135deg, #f59e0b, #ea580c) !important; border-radius: 16px; margin-bottom: 20px; color: #ffffff !important; font-weight: bold; font-size: 16px; box-shadow: 0 6px 15px rgba(245, 158, 11, 0.3);}
    div[data-testid="column"] button { width: 100%; height: 140px; border-radius: 20px; background: linear-gradient(135deg, #2563eb, #3b82f6) !important; color: #ffffff !important; font-size: 18px; font-weight: 800; border: 2px solid rgba(255,255,255,0.1) !important; box-shadow: 0 10px 25px rgba(37, 99, 235, 0.25), inset 0 2px 5px rgba(255,255,255,0.2) !important; transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important; display: flex; flex-direction: column; align-items: center; justify-content: center; letter-spacing: 0.5px; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);}
    div[data-testid="column"] button:hover { transform: translateY(-8px) !important; box-shadow: 0 15px 35px rgba(37, 99, 235, 0.4), inset 0 2px 5px rgba(255,255,255,0.4) !important; background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;}
    div[data-testid="column"] button:active { transform: translateY(2px) scale(0.96) !important; }
    .back-btn>button { background: linear-gradient(135deg, #ef4444, #dc2626) !important; height: 60px !important; border-radius: 16px !important; margin-bottom: 30px; font-size: 18px !important; font-weight: 800 !important; border: none !important; color: white !important; box-shadow: 0 8px 20px rgba(239, 68, 68, 0.3) !important;}
    .back-btn>button:hover { transform: translateY(-4px) !important; box-shadow: 0 12px 25px rgba(239, 68, 68, 0.5) !important; }
    .stMarkdown h3, label, .stMarkdown p { color: #1e293b !important; }
    </style>
    """, unsafe_allow_html=True)

subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "إنكليزي", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "عربي", "إنكليزي", "وطنية"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "فرنسي", "عربي", "إنكليزي", "وطنية"]
}

if "user_data" not in st.session_state: st.session_state["user_data"] = None
if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
if "current_view" not in st.session_state: st.session_state["current_view"] = "home" 
if "login_attempts" not in st.session_state: st.session_state["login_attempts"] = 0
if "last_active" not in st.session_state: st.session_state["last_active"] = time.time()

if st.session_state["user_data"] is not None:
    if time.time() - st.session_state["last_active"] > 3600:
        st.session_state["user_data"] = None
        st.warning("تم تسجيل الخروج تلقائياً لأسباب أمنية (Timeout).")
    st.session_state["last_active"] = time.time()

# ==========================================
# 7. شاشة الدخول والتسجيل
# ==========================================
if st.session_state["user_data"] is None:
    st.markdown(f'<div class="modern-box"><div class="welcome-title">{time_greeting}، مرحباً في سند</div><div class="programmer-tag">💻 برمجة الأستاذ حسام الأسدي</div></div>', unsafe_allow_html=True)
    
    t_log, t_sign = st.tabs(["🔐 تسجيل الدخول", "📝 إنشاء حساب طالب"])
    
    with t_log:
        with st.form("login_form"):
            st.markdown("### 🔑 تسجيل الدخول")
            u = st.text_input("الاسم الكامل")
            p = st.text_input("كلمة المرور", type="password")
            submit = st.form_submit_button("دخول المنصة 🚀")
            
            if submit:
                limit_df = get_table_df("rate_limits", "WHERE username=?", (u,))
                if not limit_df.empty and limit_df.iloc[0]['lockout_until'] > time.time():
                    wait_time = int(limit_df.iloc[0]['lockout_until'] - time.time())
                    st.error(f"⛔ تم قفل الحساب مؤقتاً لحمايتك. يرجى الانتظار {wait_time} ثانية.")
                else:
                    if u == "Hosam" and verify_password(OWNER_PASS_HASH, p):
                        st.session_state["user_data"] = {"user": u, "role": "Owner", "grade": "الكل", "is_new": False, "is_premium": True}
                        execute_sql("DELETE FROM rate_limits WHERE username=?", (u,))
                        st.rerun()
                    else:
                        users_df = get_table_df("users", "WHERE user=?", (u,))
                        if not users_df.empty and verify_password(users_df.iloc[0]["pass"], p):
                            user_record = users_df.iloc[0].to_dict()
                            user_record['is_new'] = bool(user_record.get('is_new', False))
                            user_record['is_premium'] = bool(user_record.get('is_premium', False))
                            st.session_state["user_data"] = user_record
                            execute_sql("DELETE FROM rate_limits WHERE username=?", (u,))
                            st.rerun()
                        else:
                            attempts = 1
                            if not limit_df.empty: attempts = limit_df.iloc[0]['attempts'] + 1
                            if attempts >= 5:
                                execute_sql("INSERT OR REPLACE INTO rate_limits (username, attempts, lockout_until) VALUES (?, ?, ?)", (u, attempts, time.time() + 60))
                                st.error("⚠️ تم إقفال المحاولات لـ 60 ثانية.")
                            else:
                                execute_sql("INSERT OR REPLACE INTO rate_limits (username, attempts, lockout_until) VALUES (?, ?, ?)", (u, attempts, 0))
                                st.error(f"⚠️ بيانات غير صحيحة. المحاولات المتبقية: {5 - attempts}")
    
    with t_sign:
        st.markdown("### 📋 بيانات الطالب الجديد")
        nu = st.text_input("الاسم الكامل (الرباعي)")
        ng = st.selectbox("الصف:", list(subs_map.keys()))
        fb = st.text_input("رابط فيسبوك (للتوثيق)")
        invite = st.text_input("كود دعوة الأستاذ (اختياري)")
        np = st.text_input("كلمة السر", type="password")
        np2 = st.text_input("تأكيد كلمة السر", type="password")
            
        if st.button("✅ إنشاء الحساب"):
            if not nu or not np or not np2 or not fb: st.warning("⚠️ يرجى تعبئة الحقول.")
            elif np != np2: st.error("⚠️ كلمتا المرور غير متطابقتين.")
            elif len(np) < 6: st.error("⚠️ لحمايتك، كلمة المرور يجب أن تكون 6 أحرف على الأقل.")
            else:
                if not get_table_df("users", "WHERE user=?", (nu,)).empty: st.error("⚠️ الاسم موجود مسبقاً.")
                else:
                    execute_sql("INSERT INTO users (user, pass, role, grade, fb_link, is_new, is_premium, invited_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                              (nu, hash_password_secure(np), "طالب", ng, fb, False, False, invite))
                    st.success("🎉 تم الإنشاء! سجل دخولك.")

# ==========================================
# 8. شاشات المستخدمين الموحدة
# ==========================================
else:
    user = st.session_state["user_data"]
    username_current = user["user"]
    
    if user["role"] == "أستاذ" and user.get("is_new", True):
        st.markdown(f'<div class="modern-box"><div class="welcome-title">أهلاً بك يا أستاذنا 👨‍🏫</div></div>', unsafe_allow_html=True)
        sel_grade = st.selectbox("الصف:", list(subs_map.keys()) + ["كل الصفوف"])
        all_subs = list(set([item for sublist in subs_map.values() for item in sublist]))
        sel_sub = st.selectbox("المادة:", all_subs if sel_grade == "كل الصفوف" else subs_map[sel_grade])
        pic = st.file_uploader("صورتك (اختياري)", type=['png', 'jpg', 'jpeg'])
        if st.button("حفظ 🚀"):
            if pic: Image.open(pic).save(f"profiles/{username_current}.png")
            execute_sql("INSERT INTO teacher_subjects (teacher_name, grade, subject) VALUES (?, ?, ?)", (username_current, sel_grade, sel_sub))
            execute_sql("UPDATE users SET is_new = 0 WHERE user = ?", (username_current,))
            st.session_state["user_data"]["is_new"] = False
            st.rerun()
        st.stop() 
    
    teacher_grade, teacher_sub = "", ""
    if user["role"] == "أستاذ":
        ts_df = get_table_df("teacher_subjects", "WHERE teacher_name=?", (username_current,))
        if not ts_df.empty: teacher_grade, teacher_sub = ts_df.iloc[0]["grade"], ts_df.iloc[0]["subject"]

    with st.sidebar:
        profile_path = f"profiles/{username_current}.png"
        if os.path.exists(profile_path):
            c1, c2, c3 = st.columns([1, 2, 1])
            c2.image(profile_path, use_container_width=True)
        else: st.markdown("<h1 style='text-align: center; color: #1E88E5;'>👤</h1>", unsafe_allow_html=True)
            
        st.markdown(f"<h3 style='text-align: center; margin-bottom: 0;'>{username_current}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: gray; font-weight: bold;'>{user['role']}</p>", unsafe_allow_html=True)
        if user['role'] == "طالب": st.markdown(f"<p style='text-align: center; color: #D32F2F;'>الصف: {user['grade']}</p>", unsafe_allow_html=True)
        elif user['role'] == "أستاذ": st.markdown(f"<p style='text-align: center; color: #D32F2F;'>{teacher_sub} - {teacher_grade}</p>", unsafe_allow_html=True)
            
        st.divider()
        if user['role'] == "Owner": st.success("إدارة عليا 👑")
        elif user['role'] == "أستاذ": st.info("كادر تدريسي 👨‍🏫")
        else:
            if user.get('is_premium', False): st.success("حساب مدفوع 🌟")
            else:
                st.info("حساب مجاني 🆓")
                with st.form("premium_form"):
                    code_input = st.text_input("كود التفعيل (5 أرقام):")
                    if st.form_submit_button("تفعيل 🚀"):
                        if code_input.isdigit():
                            c_df = get_table_df("codes", "WHERE code=? AND is_used=0", (int(code_input),))
                            if not c_df.empty:
                                execute_sql("UPDATE codes SET is_used=1, used_by=? WHERE code=?", (username_current, int(code_input)))
                                execute_sql("UPDATE users SET is_premium=1 WHERE user=?", (username_current,))
                                st.session_state["user_data"]["is_premium"] = True
                                st.success("تم التفعيل! 🎉"); st.rerun()
                            else: st.error("كود غير صحيح أو مستخدم.")
                
        st.divider()
        if st.button("🔴 تسجيل الخروج"): st.session_state["user_data"] = None; st.rerun()

    # --- تعريف الصفحات المحمية بالديكوريتورز ---
    @require_role(["Owner"])
    def render_admin_dashboard():
        st.header(f"👑 لوحة تحكم الإدارة - {time_greeting}")
        t_users, t_teachers, t_files, t_codes, t_notify, t_anti_cheat = st.tabs(["الطلاب", "الأساتذة", "الملفات", "الأكواد", "رسائل", "كشف الغش"])
        with t_users: st.dataframe(get_table_df("users", "WHERE role='طالب'"))
        with t_teachers:
            t_name, t_pass = st.text_input("اسم الأستاذ"), st.text_input("كلمة المرور", type="password")
            if st.button("إنشاء حساب الأستاذ") and t_name and t_pass:
                if get_table_df("users", "WHERE user=?", (t_name,)).empty:
                    execute_sql("INSERT INTO users (user, pass, role, grade, is_new, is_premium) VALUES (?, ?, ?, ?, ?, ?)", (t_name, hash_password_secure(t_pass), "أستاذ", "الكل", True, True))
                    st.success("تم التفعيل!")
        with t_files:
            f_df = get_table_df("files")
            file_to_del = st.selectbox("حذف ملف:", [""] + list(f_df['name'].values)) if not f_df.empty else ""
            if st.button("🗑️ حذف") and file_to_del:
                execute_sql("DELETE FROM files WHERE name=?", (file_to_del,))
                st.success("تم الحذف!"); st.rerun()
        with t_codes:
            num = st.number_input("عدد الأكواد:", min_value=1, value=10)
            if st.button("توليد الأكواد ⚙️"):
                for _ in range(num): execute_sql("INSERT OR IGNORE INTO codes (code, is_used, date_created) VALUES (?, 0, ?)", (random.randint(10000, 99999), datetime.now().strftime("%Y-%m-%d")))
                st.success(f"تم توليد {num} كود!")
        with t_anti_cheat:
            text1, text2 = st.text_area("إجابة الأول:"), st.text_area("إجابة الثاني:")
            if st.button("فحص الغش 🕵️"): 
                res = get_ai_response(f"أنت خبير كشف غش. قارن بين الإجابة الأولى: '{text1}' والثانية: '{text2}'. أعطني النسبة المئوية لاحتمال الغش، وجملة تحليلية للسبب.", username=username_current)
                st.markdown(f'<div class="modern-box" style="color:white;">{res}</div>', unsafe_allow_html=True)

    @require_role(["طالب", "أستاذ"])
    def render_main_app():
        view_grade = user["grade"] if user["role"] == "طالب" else st.selectbox("اختر الصف:", ["التاسع", "البكالوريا العلمي", "البكالوريا الأدبي"])
        sub = st.selectbox("المادة:", subs_map[view_grade]) if user["role"] == "طالب" else teacher_sub
        
        b_df = get_table_df("broadcasts", "WHERE grade=? AND subject=?", (view_grade, sub))
        for _, b in b_df.tail(3).iterrows(): st.markdown(f"<div class='broadcast-box'>🔔 {b['message']}</div>", unsafe_allow_html=True)

        if st.session_state["current_view"] != "home":
            if st.button("🔙 العودة للقائمة الرئيسية", use_container_width=True): st.session_state["current_view"] = "home"; st.rerun()

        if st.session_state["current_view"] == "home":
            c1, c2 = st.columns(2)
            if user["role"] == "أستاذ":
                if c1.button("📢\nإرسال إشعار"): st.session_state["current_view"] = "notify"; st.rerun()
                if c2.button("📤\nرفع الملفات"): st.session_state["current_view"] = "upload"; st.rerun()
            else:
                if c1.button("📚\nالمكتبة"): st.session_state["current_view"] = "library"; st.rerun()
                if c2.button("🤖\nالمعلم الذكي"): st.session_state["current_view"] = "ai_teacher"; st.rerun()
            
            c3, c4 = st.columns(2)
            if user["role"] == "أستاذ":
                if c3.button("📚\nالمكتبة"): st.session_state["current_view"] = "library"; st.rerun()
                if c4.button("🤖\nالمعلم الذكي"): st.session_state["current_view"] = "ai_teacher"; st.rerun()
            else:
                if c3.button("📸\nعدسة الذكاء"): st.session_state["current_view"] = "lens"; st.rerun()
                if c4.button("📝\nالامتحانات"): st.session_state["current_view"] = "exams"; st.rerun()

            c5, c6 = st.columns(2)
            if user["role"] == "أستاذ":
                if c5.button("📸\nعدسة الذكاء"): st.session_state["current_view"] = "lens"; st.rerun()
                if c6.button("📝\nالامتحانات"): st.session_state["current_view"] = "exams"; st.rerun()
            else:
                if c5.button("📅\nخطة الدراسة"): st.session_state["current_view"] = "plan"; st.rerun()
                if c6.button("📖\nأسئلة الدورات"): st.session_state["current_view"] = "past_papers"; st.rerun()
                
            if user["role"] == "أستاذ":
                c7, c8 = st.columns(2)
                if c7.button("📖\nأسئلة الدورات"): st.session_state["current_view"] = "past_papers"; st.rerun()

        elif st.session_state["current_view"] == "notify" and user["role"] == "أستاذ":
            msg = st.text_area("اكتب الإشعار:")
            if st.button("إرسال") and msg:
                execute_sql("INSERT INTO broadcasts (sender, grade, subject, message, date) VALUES (?, ?, ?, ?, ?)", (username_current, view_grade, sub, msg, datetime.now().strftime("%Y-%m-%d")))
                st.success("تم!")

        elif st.session_state["current_view"] == "upload" and user["role"] == "أستاذ":
            with st.form("up"):
                uploaded_file = st.file_uploader("ملف PDF (حتى 50 ميغا)", type="pdf")
                name = st.text_input("الاسم")
                ch = st.number_input("رقم البحث", min_value=1)
                tf = st.radio("النوع:", ["بحث", "نموذج", "سلم", "دورات"])
                if st.form_submit_button("رفع"):
                    if uploaded_file:
                        try:
                            file_bytes = uploaded_file.getvalue()
                            if len(file_bytes) > 50 * 1024 * 1024:
                                st.error("🚨 حجم الملف يتجاوز الحد المسموح (50MB).")
                            elif not file_bytes.startswith(b'%PDF'):
                                st.error("🚨 محاولة رفع ملف خبيث!")
                            else:
                                f_name = f"{tf}_{sub}_{name}.pdf".replace(' ', '_')
                                folder = "lessons" if tf in ["بحث", "دورات"] else "exams"
                                p = os.path.join(folder, f_name)
                                with open(p, "wb") as f: f.write(file_bytes)
                                execute_sql("INSERT OR REPLACE INTO files (name, grade, sub, type, uploader, chapter_num) VALUES (?, ?, ?, ?, ?, ?)", (f_name, view_grade, sub, tf, username_current, ch))
                                if tf in ["بحث", "دورات"]: get_and_save_embeddings(p)
                                st.success("نجاح!")
                        except Exception as e:
                            st.error(f"حدث خطأ أثناء الرفع: {e}")

        elif st.session_state["current_view"] == "library":
            f_df = get_table_df("files", "WHERE grade=? AND sub=?", (view_grade, sub))
            for _, r in f_df.iterrows():
                p = os.path.join("lessons" if r['type'] in ["بحث", "دورات"] else "exams", r['name'])
                if os.path.exists(p):
                    if user["role"] == "طالب" and not user.get("is_premium") and r['chapter_num'] > 2:
                        st.button(f"🔒 مقفول: {r['name']}", disabled=True, key=r['name'])
                    else:
                        with open(p, "rb") as f: st.download_button(f"📥 {r['name']}", f, file_name=r['name'], key=r['name'])

        elif st.session_state["current_view"] == "ai_teacher":
            f_df = get_table_df("files", "WHERE grade=? AND sub=? AND type='بحث'", (view_grade, sub))
            sel = st.selectbox("النوطة:", f_df['name'].tolist()) if not f_df.empty else ""
            for m in st.session_state["chat_history"]: st.chat_message(m["role"]).write(m["content"])
            if q := st.chat_input("اسأل..."):
                st.session_state["chat_history"].append({"role": "user", "content": q}); st.chat_message("user").write(q)
                with st.spinner("يبحث بذكاء..."):
                    ctx = get_best_context_smart(q, os.path.join("lessons", sel)) if sel else ""
                    ans = get_ai_response(q, strict_mode=True, context_text=ctx, username=username_current)
                st.session_state["chat_history"].append({"role": "assistant", "content": ans}); st.chat_message("assistant").write(ans)

        elif st.session_state["current_view"] == "lens":
            img = st.file_uploader("صورة:", type=["jpg", "png"])
            if img and st.button("تحليل"): st.info(get_ai_response(f"مادة {sub}. صحح أو اشرح.", image=Image.open(img), strict_mode=True, username=username_current))

        elif st.session_state["current_view"] == "exams":
            if st.button("🎯 توليد أسئلة"): st.markdown(f'<div class="modern-box" style="color:white;">{get_ai_response(f"ولد نموذج وزاري لمادة {sub}.", strict_mode=True, username=username_current)}</div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("🗣️ **التسميع الشفهي**")
            
            # حماية الـ Audio Input (لتجنب الشاشة الحمراء)
            try:
                aud = st.audio_input("🎤 سجل إجابتك:")
                if aud:
                    st.audio(aud)
                    with st.spinner("يستمع..."): 
                        st.success(get_ai_response(f"صحح إجابة الطالب بمادة {sub}.", audio={"mime_type": "audio/wav", "data": aud.getvalue()}, strict_mode=True, username=username_current))
            except Exception as e:
                st.warning("⚠️ ميزة تسجيل الصوت غير مدعومة على متصفحك أو إصدار التطبيق الحالي.")

        elif st.session_state["current_view"] == "plan" and user["role"] == "طالب":
            c1, c2 = st.columns(2)
            days = c1.number_input("أيام للامتحان؟", 1, value=20)
            hours = c2.slider("ساعات باليوم؟", 1, 15, 6)
            if st.button("توليد الخطة"):
                with st.spinner("يخطط..."): st.markdown(f'<div class="modern-box" style="color:white;">{get_ai_response(f"طالب بكالوريا. باقي {days} يوم، وسأدرس {hours} ساعات مادة {sub}. ولد جدول.", username=username_current)}</div>', unsafe_allow_html=True)

        elif st.session_state["current_view"] == "past_papers":
            f_df = get_table_df("files", "WHERE grade=? AND sub=? AND type='دورات'", (view_grade, sub))
            sel = st.selectbox("ملف:", f_df['name'].tolist()) if not f_df.empty else ""
            tq = st.text_input("البحث:")
            if st.button("استخراج") and tq and sel:
                with st.spinner("يستخرج..."):
                    try:
                        up = genai.upload_file(os.path.join("lessons", sel))
                        st.markdown(f'<div class="modern-box" style="color:white;">{get_ai_response(f"استخرج أسئلة {tq} من الملف.", file_uri=up, username=username_current)}</div>', unsafe_allow_html=True)
                        genai.delete_file(up.name)
                    except Exception as e: 
                        st.error("⚠️ حدث خطأ، تأكد من جودة الملف أو اتصالك.")
                        logging.error(f"File AI Error: {str(e)}")

    if user["role"] == "Owner": render_admin_dashboard()
    else: render_main_app()
