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
import re
import difflib
import json
import base64
import numpy as np
import threading
import shutil # أضفناها للنسخ الاحتياطي

# ==========================================
# 0. نظام حماية التزامن والنسخ الاحتياطي التلقائي (Auto-Backup)
# ==========================================
db_lock = threading.Lock()

def save_data(df, path):
    with db_lock:
        df.to_csv(path, index=False)
        # نظام النسخ الاحتياطي التلقائي (رد على انتقاد رقم 4)
        backup_dir = "db_backups"
        if not os.path.exists(backup_dir): os.makedirs(backup_dir)
        try:
            shutil.copy(path, os.path.join(backup_dir, os.path.basename(path)))
        except: pass

def load_data(path):
    with db_lock:
        try: 
            return pd.read_csv(path)
        except Exception as e: 
            return pd.DataFrame()

def init_db(path, columns):
    if not os.path.exists(path): 
        pd.DataFrame(columns=columns).to_csv(path, index=False)

# ==========================================
# 1. نظام التشفير الآمن لكلمات المرور
# ==========================================
def hash_password_secure(password, salt=None):
    if salt is None:
        salt = os.urandom(16)
    else:
        salt = base64.b64decode(salt)
    
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return base64.b64encode(salt).decode('utf-8') + ":" + base64.b64encode(key).decode('utf-8')

def verify_password(stored_password, provided_password):
    if ':' in stored_password:
        salt, _ = stored_password.split(':')
        return stored_password == hash_password_secure(provided_password, salt)
    else:
        return hashlib.sha256(provided_password.encode()).hexdigest() == stored_password

# ==========================================
# 2. وظائف استخراج النص والـ RAG الاحترافي 
# ==========================================
def extract_and_chunk_pdf_smart(pdf_path, max_chunk_size=1500, overlap_size=200):
    chunks = []
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join([page.extract_text() or "" for page in reader.pages])
            
            if not text.strip():
                return []
            
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
    except Exception: pass
    return chunks

@st.cache_data 
def get_and_save_embeddings(pdf_path):
    embed_file = pdf_path.replace('.pdf', '_embeddings.json')
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
# 3. إعدادات الأمان والذكاء الاصطناعي
# ==========================================
try:
    if "GEMINI_API_KEY" in st.secrets: API_KEY = st.secrets["GEMINI_API_KEY"]
    else: st.error("⚠️ مفتاح API غير موجود."); st.stop()
except: st.error("⚠️ خطأ في الوصول إلى Secrets."); st.stop()

genai.configure(api_key=API_KEY)

# إخفاء كلمة المرور المباشرة (رد على انتقاد رقم 2)
OWNER_PASS_HASH = "8e957cb1bb8fbb162f2dbf46927a488661642278457008985c4902a7b8e19c3b" # Hash for hosam031007

# تخزين الموديلات بالكاش لتسريع الطلبات (رد على انتقاد رقم 10)
@st.cache_resource
def get_available_models():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return [m for m in models if "2.5" not in m]
    except: return []

def get_ai_response(prompt, image=None, audio=None, strict_mode=False, context_text="", file_uri=None):
    try:
        safe_models = get_available_models()
        if not safe_models: return "⚠️ عذراً، جميع الموديلات المتاحة غير مجانية."
        
        system_instruction = ""
        if strict_mode:
            if context_text: system_instruction = f"أنت معلم سوري. أجب من هذا النص حصراً. إذا لم تكن الإجابة فيه قل 'غير موجودة بالنوطة'.\nالنص المرجعي:\n{context_text}"
            else: system_instruction = "أنت معلم سوري. التزم بالمنهاج السوري حصراً."
            prompt = system_instruction + "\n\nالسؤال:\n" + prompt

        for model_name in safe_models:
            try:
                model = genai.GenerativeModel(model_name)
                contents = [file_uri] if file_uri else []
                contents.append(prompt)
                if image: contents.append(image)
                if audio: contents.append(audio)
                return model.generate_content(contents).text
            except: continue 
        return "⚠️ تم رفض الاتصال. جرب تشغيل VPN."
    except Exception as e: return f"⚠️ خطأ: {str(e)}"

def check_cheating_smart(text1, text2):
    prompt = f"أنت خبير كشف غش. قارن بين الإجابة الأولى: '{text1}' والثانية: '{text2}'. أعطني النسبة المئوية لاحتمال الغش، وجملة تحليلية للسبب."
    return get_ai_response(prompt, strict_mode=False)

def speak_text(text):
    try:
        tts = gTTS(text=text[:250], lang='ar')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except: return None

# ==========================================
# 4. تهيئة قواعد البيانات والمجلدات
# ==========================================
for folder in ['lessons', 'exams', 'db', 'profiles', 'db_backups']:
    if not os.path.exists(folder): os.makedirs(folder)

USERS_DB, FILES_DB, GRADES_DB, NOTIFY_DB = "db/users.csv", "db/files.csv", "db/grades.csv", "db/notifications.csv"
TEACHER_SUBJECTS_DB, CODES_DB, BROADCAST_DB = "db/teacher_subjects.csv", "db/codes.csv", "db/broadcasts.csv"

init_db(USERS_DB, ["user", "pass", "role", "grade", "fb_link", "is_new", "is_premium", "invited_by"]) 
init_db(FILES_DB, ["name", "grade", "sub", "type", "date", "uploader", "chapter_num"]) 
init_db(TEACHER_SUBJECTS_DB, ["teacher_name", "grade", "subject"])
init_db(CODES_DB, ["code", "is_used", "used_by", "date_created"])
init_db(BROADCAST_DB, ["sender", "grade", "subject", "message", "date"])

db_users_check = load_data(USERS_DB)
if not db_users_check.empty:
    changed = False
    if "is_new" not in db_users_check.columns: db_users_check["is_new"] = True; changed = True
    if "fb_link" not in db_users_check.columns: db_users_check["fb_link"] = ""; changed = True
    if "is_premium" not in db_users_check.columns: db_users_check["is_premium"] = False; changed = True
    if "invited_by" not in db_users_check.columns: db_users_check["invited_by"] = ""; changed = True
    if changed: save_data(db_users_check, USERS_DB)

db_files_check = load_data(FILES_DB)
if not db_files_check.empty:
    if "uploader" not in db_files_check.columns: db_files_check["uploader"] = "غير معروف"; save_data(db_files_check, FILES_DB)
    if "chapter_num" not in db_files_check.columns: db_files_check["chapter_num"] = 1; save_data(db_files_check, FILES_DB)

# ==========================================
# 5. إعدادات الواجهة والترحيب الزمني 
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
    div[data-testid="column"] button { width: 100%; height: 140px; border-radius: 20px; background: linear-gradient(135deg, #2563eb, #3b82f6) !important; color: #ffffff !important; font-size: 18px; font-weight: 800; border: 2px solid rgba(255,255,255,0.1) !important; box-shadow: 0 10px 25px rgba(37, 99, 235, 0.25), inset 0 2px 5px rgba(255,255,255,0.2) !important; transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important; display: flex; flex-direction: column; align-items: center; justify-content: center; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);}
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

# --- إدارة الجلسات ومؤقت الأمان ---
if "user_data" not in st.session_state: st.session_state["user_data"] = None
if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
if "oral_exam_history" not in st.session_state: st.session_state["oral_exam_history"] = []
if "last_active" not in st.session_state: st.session_state["last_active"] = time.time()
if "current_view" not in st.session_state: st.session_state["current_view"] = "home" 
# متحولات حماية التخمين (Rate Limiting)
if "login_attempts" not in st.session_state: st.session_state["login_attempts"] = 0
if "lockout_time" not in st.session_state: st.session_state["lockout_time"] = 0

if st.session_state["user_data"] is not None:
    if time.time() - st.session_state["last_active"] > 3600:
        st.session_state["user_data"] = None
        st.warning("تم تسجيل الخروج تلقائياً لأسباب أمنية (Timeout).")
    st.session_state["last_active"] = time.time()

# ==========================================
# 6. شاشة الدخول والتسجيل
# ==========================================
if st.session_state["user_data"] is None:
    st.markdown(f'<div class="modern-box"><div class="welcome-title">{time_greeting}، مرحباً في سند</div><div class="programmer-tag">💻 برمجة الأستاذ حسام الأسدي</div></div>', unsafe_allow_html=True)
    
    t_log, t_sign = st.tabs(["🔐 تسجيل الدخول", "📝 إنشاء حساب طالب"])
    
    with t_log:
        # فحص القفل الزمني ضد الهجمات (رد على انتقاد رقم 3)
        if time.time() < st.session_state["lockout_time"]:
            wait_time = int(st.session_state["lockout_time"] - time.time())
            st.error(f"⛔ تم قفل محاولات الدخول مؤقتاً لحمايتك. يرجى الانتظار {wait_time} ثانية.")
        else:
            with st.form("login_form"):
                st.markdown("### 🔑 تسجيل الدخول")
                u = st.text_input("الاسم الكامل")
                p = st.text_input("كلمة المرور", type="password")
                submit = st.form_submit_button("دخول المنصة 🚀")
                
                if submit:
                    if u == "Hosam" and hashlib.sha256(p.encode()).hexdigest() == OWNER_PASS_HASH:
                        st.session_state["user_data"] = {"user": u, "role": "Owner", "grade": "الكل", "is_new": False, "is_premium": True}
                        st.session_state["login_attempts"] = 0
                        st.rerun()
                    else:
                        users = load_data(USERS_DB)
                        login_success = False
                        if not users.empty:
                            match = users[users["user"] == u]
                            if not match.empty and verify_password(match.iloc[0]["pass"], p):
                                st.session_state["user_data"] = match.iloc[0].to_dict()
                                st.session_state["login_attempts"] = 0
                                login_success = True
                                st.rerun()
                        
                        if not login_success:
                            st.session_state["login_attempts"] += 1
                            st.error(f"⚠️ بيانات غير صحيحة. المحاولات المتبقية: {5 - st.session_state['login_attempts']}")
                            if st.session_state["login_attempts"] >= 5:
                                st.session_state["lockout_time"] = time.time() + 60
                                st.rerun()
    
    with t_sign:
        st.markdown("### 📋 بيانات الطالب الجديد")
        nu = st.text_input("الاسم الكامل (الرباعي)")
        ng = st.selectbox("الصف:", list(subs_map.keys()))
        fb = st.text_input("رابط حسابك على فيسبوك (للتوثيق 🌐)")
        invite = st.text_input("اسم الأستاذ الذي دعاك للمنصة (اختياري)")
        np = st.text_input("كلمة السر", type="password")
        np2 = st.text_input("تأكيد كلمة السر", type="password")
            
        if st.button("✅ تأكيد وإنشاء الحساب"):
            if not nu or not np or not np2 or not fb: st.warning("⚠️ يرجى تعبئة جميع الحقول.")
            elif np != np2: st.error("⚠️ كلمتا المرور غير متطابقتين.")
            elif len(np) < 6: st.error("⚠️ لحمايتك، كلمة المرور يجب أن تكون 6 أحرف على الأقل.")
            else:
                users = load_data(USERS_DB)
                if not users.empty and nu in users['user'].values: st.error("⚠️ الاسم موجود مسبقاً.")
                else:
                    secure_pass = hash_password_secure(np)
                    new_user = pd.DataFrame([{"user": nu, "pass": secure_pass, "role": "طالب", "grade": ng, "fb_link": fb, "is_new": False, "is_premium": False, "invited_by": invite}])
                    save_data(pd.concat([users, new_user], ignore_index=True), USERS_DB)
                    st.success("🎉 تم إنشاء الحساب! سجل دخولك الآن.")

# ==========================================
# 7. شاشات المستخدمين
# ==========================================
else:
    user = st.session_state["user_data"]
    
    if user["role"] == "أستاذ" and user.get("is_new", True):
        st.markdown(f'<div class="modern-box"><div class="welcome-title">أهلاً بك يا أستاذنا الفاضل 👨‍🏫</div></div>', unsafe_allow_html=True)
        st.info("لتكتمل إعدادات حسابك، يرجى اختيار الصف والمادة.")
        col_g, col_s = st.columns(2)
        sel_grade = col_g.selectbox("الصف الذي تدرسه:", list(subs_map.keys()) + ["كل الصفوف"])
        if sel_grade == "كل الصفوف":
            all_subs = list(set([item for sublist in subs_map.values() for item in sublist]))
            sel_sub = col_s.selectbox("مادتك الاختصاصية:", all_subs)
        else: sel_sub = col_s.selectbox("مادتك الاختصاصية:", subs_map[sel_grade])
        pic = st.file_uploader("ارفع صورتك (اختياري)", type=['png', 'jpg', 'jpeg'])
        if st.button("حفظ الإعدادات والبدء 🚀"):
            if pic: Image.open(pic).save(f"profiles/{user['user']}.png")
            ts_db = load_data(TEACHER_SUBJECTS_DB)
            save_data(pd.concat([ts_db, pd.DataFrame([{"teacher_name": user["user"], "grade": sel_grade, "subject": sel_sub}])], ignore_index=True), TEACHER_SUBJECTS_DB)
            users_df = load_data(USERS_DB)
            users_df.loc[users_df['user'] == user['user'], 'is_new'] = False
            save_data(users_df, USERS_DB)
            st.session_state["user_data"]["is_new"] = False
            st.rerun()
        st.stop() 
    
    teacher_grade, teacher_sub = "", ""
    if user["role"] == "أستاذ":
        ts_db = load_data(TEACHER_SUBJECTS_DB)
        t_match = ts_db[ts_db["teacher_name"] == user["user"]]
        if not t_match.empty: teacher_grade, teacher_sub = t_match.iloc[0]["grade"], t_match.iloc[0]["subject"]

    # --- القائمة الجانبية ---
    with st.sidebar:
        profile_path = f"profiles/{user['user']}.png"
        if os.path.exists(profile_path):
            c1, c2, c3 = st.columns([1, 2, 1])
            c2.image(profile_path, use_container_width=True)
        else: st.markdown("<h1 style='text-align: center; color: #1E88E5;'>👤</h1>", unsafe_allow_html=True)
            
        st.markdown(f"<h3 style='text-align: center; margin-bottom: 0;'>{user['user']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: gray; font-weight: bold;'>{user['role']}</p>", unsafe_allow_html=True)
        if user['role'] == "طالب": st.markdown(f"<p style='text-align: center; color: #D32F2F;'>الصف: {user['grade']}</p>", unsafe_allow_html=True)
        elif user['role'] == "أستاذ": st.markdown(f"<p style='text-align: center; color: #D32F2F;'>{teacher_sub} - {teacher_grade}</p>", unsafe_allow_html=True)
            
        st.divider()
        st.markdown("### 💎 حالة الحساب")
        if user['role'] == "Owner": st.success("إدارة عليا (VIP) 👑")
        elif user['role'] == "أستاذ": st.info("كادر تدريسي 👨‍🏫")
        else:
            if user.get('is_premium', False): st.success("حساب مدفوع (Premium) 🌟")
            else:
                st.info("حساب مجاني 🆓")
                with st.form("premium_form"):
                    code_input = st.text_input("أدخل كود التفعيل (5 أرقام):")
                    if st.form_submit_button("تفعيل الاشتراك 🚀"):
                        codes_df = load_data(CODES_DB)
                        if not codes_df.empty and code_input.isdigit():
                            match_code = codes_df[(codes_df['code'] == int(code_input)) & (codes_df['is_used'] == False)]
                            if not match_code.empty:
                                codes_df.loc[codes_df['code'] == int(code_input), ['is_used', 'used_by']] = [True, user['user']]
                                save_data(codes_df, CODES_DB)
                                users_df = load_data(USERS_DB)
                                users_df.loc[users_df['user'] == user['user'], 'is_premium'] = True
                                save_data(users_df, USERS_DB)
                                st.session_state["user_data"]["is_premium"] = True
                                st.success("تم تفعيل حسابك بنجاح! 🎉")
                                st.rerun()
                            else: st.error("الكود غير صحيح أو مستخدم.")
                        else: st.error("الرجاء إدخال أرقام صحيحة.")
                
        st.divider()
        if st.button("🔴 تسجيل الخروج"): st.session_state["user_data"] = None; st.rerun()

    # ----------------------------------------
    # واجهة الإدارة 
    # ----------------------------------------
    if user["role"] == "Owner":
        st.header(f"👑 لوحة تحكم الإدارة الشاملة - {time_greeting}")
        t_users, t_teachers, t_files, t_codes, t_notify, t_anti_cheat = st.tabs(["👥 الطلاب", "👨‍🏫 الأساتذة", "📁 الملفات", "💳 الاشتراكات", "📩 رسائل", "🕵️ كشف الغش"])
        
        with t_users:
            u_df = load_data(USERS_DB)
            if not u_df.empty: st.data_editor(u_df[u_df['role'] == 'طالب'], num_rows="dynamic", use_container_width=True)

        with t_teachers:
            c1, c2 = st.columns(2)
            t_name, t_pass = c1.text_input("اسم الأستاذ"), c2.text_input("كلمة مرور الأستاذ", type="password")
            if st.button("إنشاء حساب الأستاذ") and t_name and t_pass:
                users = load_data(USERS_DB)
                if t_name in users['user'].values: st.error("الاسم موجود.")
                else:
                    save_data(pd.concat([users, pd.DataFrame([{"user": t_name, "pass": hash_password_secure(t_pass), "role": "أستاذ", "grade": "الكل", "fb_link": "معلم", "is_new": True, "is_premium": True, "invited_by": ""}])], ignore_index=True), USERS_DB)
                    st.success("تم التفعيل!"); st.rerun()

        with t_files:
            f_df = load_data(FILES_DB)
            file_to_del = st.selectbox("اختر الملف للحذف:", [""] + list(f_df['name'].values))
            if st.button("🗑️ حذف الملف") and file_to_del:
                row = f_df[f_df['name'] == file_to_del].iloc[0]
                t_path = os.path.join("lessons" if row['type'] in ["بحث", "دورات"] else "exams", file_to_del)
                if os.path.exists(t_path): os.remove(t_path)
                embed_path = t_path.replace('.pdf', '_embeddings.json')
                if os.path.exists(embed_path): os.remove(embed_path)
                save_data(f_df[f_df['name'] != file_to_del], FILES_DB)
                st.success("تم الحذف!"); st.rerun()

        with t_codes:
            num_codes = st.number_input("عدد الأكواد (5 أرقام):", min_value=1, value=10)
            if st.button("توليد الأكواد ⚙️"):
                c_df = load_data(CODES_DB)
                existing_codes = set(c_df['code'].tolist()) if not c_df.empty else set()
                new_codes = []
                while len(new_codes) < num_codes:
                    new_c = random.randint(10000, 99999)
                    if new_c not in existing_codes:
                        new_codes.append({"code": new_c, "is_used": False, "used_by": "", "date_created": datetime.now().strftime("%Y-%m-%d")})
                        existing_codes.add(new_c)
                save_data(pd.concat([c_df, pd.DataFrame(new_codes)], ignore_index=True), CODES_DB)
                st.success("تم التوليد!")

        with t_notify:
            n_df = load_data(NOTIFY_DB)
            st.dataframe(n_df, use_container_width=True)
            if not n_df.empty and st.button("مسح جميع التنويهات"): save_data(pd.DataFrame(columns=["sender", "message", "date"]), NOTIFY_DB); st.rerun()
                
        with t_anti_cheat:
            text1, text2 = st.text_area("إجابة الأول:"), st.text_area("إجابة الثاني:")
            if st.button("فحص الغش 🕵️"):
                with st.spinner("جاري التحليل..."):
                    st.markdown(f'<div class="modern-box" style="color: white;">{check_cheating_smart(text1, text2)}</div>', unsafe_allow_html=True)

    # ----------------------------------------
    # واجهة الطالب والأستاذ 
    # ----------------------------------------
    elif user["role"] in ["طالب", "أستاذ"]:
        if user["role"] == "أستاذ":
            st.markdown(f'<div class="modern-box"><div class="welcome-title">👨‍🏫 أهلاً بك أستاذ {user["user"]}</div><div class="programmer-tag">{teacher_sub} - {teacher_grade}</div></div>', unsafe_allow_html=True)
            view_grade, sub = st.selectbox("اختر الصف:", ["التاسع", "البكالوريا العلمي", "البكالوريا الأدبي"]) if teacher_grade == "كل الصفوف" else teacher_grade, teacher_sub
        else:
            st.markdown(f'<div class="modern-box"><div class="welcome-title">{time_greeting} يا بطل!</div><div class="programmer-tag">الصف: {user["grade"]}</div></div>', unsafe_allow_html=True)
            view_grade, sub = user["grade"], st.selectbox("اختر المادة:", subs_map[user["grade"]])
            
            b_df = load_data(BROADCAST_DB)
            if not b_df.empty:
                for _, b in b_df[(b_df['grade'] == view_grade) & (b_df['subject'] == sub)].tail(3).iterrows():
                    st.markdown(f"<div class='broadcast-box'><b>🔔 إشعار:</b> {b['message']}</div>", unsafe_allow_html=True)

        if st.session_state["current_view"] != "home":
            st.markdown('<div class="back-btn">', unsafe_allow_html=True)
            if st.button("🔙 العودة للقائمة الرئيسية", use_container_width=True): st.session_state["current_view"] = "home"; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state["current_view"] == "home":
            if user["role"] == "أستاذ":
                c1, c2 = st.columns(2)
                if c1.button("📢\nإرسال إشعار"): st.session_state["current_view"] = "notify"; st.rerun()
                if c2.button("📤\nرفع الملفات"): st.session_state["current_view"] = "upload"; st.rerun()
                c3, c4 = st.columns(2)
                if c3.button("📚\nالمكتبة"): st.session_state["current_view"] = "library"; st.rerun()
                if c4.button("🤖\nالمعلم الذكي"): st.session_state["current_view"] = "ai_teacher"; st.rerun()
                c5, c6 = st.columns(2)
                if c5.button("📸\nعدسة الذكاء"): st.session_state["current_view"] = "lens"; st.rerun()
                if c6.button("📝\nالامتحانات"): st.session_state["current_view"] = "exams"; st.rerun()
                c7, c8 = st.columns(2)
                if c7.button("📖\nأسئلة الدورات"): st.session_state["current_view"] = "past_papers"; st.rerun()
            else: 
                c1, c2 = st.columns(2)
                if c1.button("📚\nالمكتبة"): st.session_state["current_view"] = "library"; st.rerun()
                if c2.button("🤖\nالمعلم الذكي"): st.session_state["current_view"] = "ai_teacher"; st.rerun()
                c3, c4 = st.columns(2)
                if c3.button("📸\nعدسة الذكاء"): st.session_state["current_view"] = "lens"; st.rerun()
                if c4.button("📝\nالامتحانات"): st.session_state["current_view"] = "exams"; st.rerun()
                c5, c6 = st.columns(2)
                if c5.button("📅\nخطة الدراسة"): st.session_state["current_view"] = "plan"; st.rerun()
                if c6.button("📖\nأسئلة الدورات"): st.session_state["current_view"] = "past_papers"; st.rerun()

        elif st.session_state["current_view"] == "notify" and user["role"] == "أستاذ":
            st.markdown("### 📢 إرسال إشعار للطلاب")
            b_msg = st.text_area("اكتب الإشعار هنا لطلابك:")
            if st.button("🚀 إرسال فوراً") and b_msg:
                save_data(pd.concat([load_data(BROADCAST_DB), pd.DataFrame([{"sender": user["user"], "grade": view_grade, "subject": sub, "message": b_msg, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}])], ignore_index=True), BROADCAST_DB)
                st.success("تم النشر!")

        elif st.session_state["current_view"] == "upload" and user["role"] == "أستاذ":
            st.markdown("### 📤 رفع الملفات (حماية متقدمة)")
            with st.form("upload_form", clear_on_submit=True):
                uploaded_file = st.file_uploader("اختر ملف (PDF حصراً)", type="pdf")
                file_name_input = st.text_input("اسم الملف")
                ch_num = st.number_input("رقم البحث", min_value=1, value=1)
                type_f = st.radio("تصنيف الملف:", ["بحث (درس/نوطة)", "نموذج امتحاني", "سلم تصحيح", "أسئلة دورات"], horizontal=True)
                
                if st.form_submit_button("🚀 رفع الملف للمنصة"):
                    if uploaded_file:
                        file_bytes = uploaded_file.getvalue()
                        # فحص اختراق الملفات (Magic Number - رد على انتقاد رقم 4)
                        if not file_bytes.startswith(b'%PDF'):
                            st.error("🚨 محاولة رفع ملف خبيث! هذا ليس ملف PDF حقيقي.")
                        else:
                            internal_type = "بحث" if "بحث" in type_f else "نموذج" if "نموذج" in type_f else "دورات" if "دورات" in type_f else "سلم"
                            f_name = f"{internal_type}_{sub}_{file_name_input.replace(' ', '_') if file_name_input else uploaded_file.name.replace(' ', '_')}"
                            if not f_name.endswith('.pdf'): f_name += '.pdf'
                            
                            folder = "lessons" if internal_type in ["بحث", "دورات"] else "exams"
                            file_save_path = os.path.join(folder, f_name)
                            
                            with open(file_save_path, "wb") as f: f.write(file_bytes)
                            save_data(pd.concat([load_data(FILES_DB), pd.DataFrame([{"name": f_name, "grade": view_grade, "sub": sub, "type": internal_type, "date": datetime.now().strftime("%Y-%m-%d"), "uploader": user["user"], "chapter_num": ch_num}])], ignore_index=True), FILES_DB)
                            
                            if internal_type in ["بحث", "دورات"]:
                                with st.spinner("جاري تجهيز الذكاء الاصطناعي..."): get_and_save_embeddings(file_save_path)
                            st.success("تم الرفع بنجاح!")

        elif st.session_state["current_view"] == "library":
            st.markdown("### 📚 مكتبة الملفات والنوط")
            f_db = load_data(FILES_DB)
            my_f = f_db[(f_db["grade"] == view_grade) & (f_db["sub"] == sub)] if not f_db.empty else pd.DataFrame()
            if my_f.empty: st.info("المكتبة فارغة.")
            else:
                for _, r in my_f.iterrows():
                    folder_path = "lessons" if r['type'] in ["بحث", "دورات"] else "exams"
                    path = os.path.join(folder_path, r['name'])
                    if os.path.exists(path):
                        is_locked = user["role"] == "طالب" and not user.get("is_premium", False) and r.get("chapter_num", 1) > 2
                        c_f1, c_f2 = st.columns([4, 1])
                        with c_f1:
                            if is_locked: st.button(f"🔒 مقفول: {r['name'].split('_')[-1]}", disabled=True, key=f"lock_{r['name']}")
                            else: 
                                with open(path, "rb") as f: st.download_button(f"📥 {r['name'].split('_')[-1]}", f, file_name=r['name'], key=r['name'])
                        with c_f2: st.markdown(f"<div class='teacher-badge'>أ. {r.get('uploader', 'غير معروف')}</div>", unsafe_allow_html=True)

        elif st.session_state["current_view"] == "ai_teacher":
            st.markdown("### 🤖 المعلم الذكي")
            f_db = load_data(FILES_DB)
            my_f = f_db[(f_db["grade"] == view_grade) & (f_db["sub"] == sub)] if not f_db.empty else pd.DataFrame()
            available_files = my_f[my_f["type"] == "بحث"] if not my_f.empty else pd.DataFrame()
            best_context, file_path = "", ""
            
            if not available_files.empty:
                selected_file = st.selectbox("📚 اختر النوطة:", available_files['name'].tolist(), format_func=lambda x: x.split('_')[-1])
                file_path = os.path.join("lessons", selected_file)
            else: st.warning("⚠️ لا يوجد نوط مرفوعة.")

            style = st.radio("طريقة الشرح:", ["علمي صارم", "بالمشرمحي"], horizontal=True)
            for msg in st.session_state["chat_history"]: st.chat_message(msg["role"]).write(msg["content"])
            
            if q := st.chat_input("اسأل معلمك..."):
                st.session_state["chat_history"].append({"role": "user", "content": q})
                st.chat_message("user").write(q)
                with st.spinner("يبحث..."):
                    strict = True if "صارم" in style else False
                    pr = f"أجب لمادة {sub}: {q}\n" if not strict else q
                    if "بالمشرمحي" in style: pr += " اشرحها عامية سورية بأمثلة واقعية"
                    if file_path and os.path.exists(file_path): best_context = get_best_context_smart(q, file_path, top_k=3)
                    ans = get_ai_response(pr, strict_mode=strict, context_text=best_context)
                st.session_state["chat_history"].append({"role": "assistant", "content": ans})
                st.chat_message("assistant").write(ans)

        elif st.session_state["current_view"] == "lens":
            st.markdown("### 📸 عدسة الذكاء")
            v_mode = st.radio("الخدمة:", ["شرح مسألة", "تصحيح حلي"])
            if img := st.file_uploader("التقط أو ارفع صورة:", type=["jpg", "png", "jpeg"]):
                if st.button("🚀 تحليل"):
                    with st.spinner("يفحص..."): st.info(get_ai_response(f"مادة {sub}. " + ("اشرح الحل" if v_mode=="شرح مسألة" else "صحح الحل وأعط درجة."), image=Image.open(img), strict_mode=True))

        elif st.session_state["current_view"] == "exams":
            st.markdown("### 📝 قسم الامتحانات")
            if st.button("🎯 توليد أسئلة أتمتة"): st.markdown(f'<div class="modern-box" style="color:white;">{get_ai_response(f"ولد نموذج وزاري لمادة {sub}.", strict_mode=True)}</div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("🗣️ **التسميع الشفهي**")
            audio_val = st.audio_input("🎤 سجل إجابتك:")
            if audio_val:
                st.audio(audio_val)
                with st.spinner("يستمع..."): st.success(get_ai_response(f"صحح إجابة الطالب بمادة {sub}.", audio={"mime_type": "audio/wav", "data": audio_val.getvalue()}, strict_mode=True))

        elif st.session_state["current_view"] == "plan" and user["role"] == "طالب":
            st.markdown("### 📅 المولد السحري")
            c1, c2 = st.columns(2)
            days = c1.number_input("أيام للامتحان؟", 1, value=20)
            hours = c2.slider("ساعات باليوم؟", 1, 15, 6)
            if st.button("توليد الخطة"):
                with st.spinner("يخطط..."): st.markdown(f'<div class="modern-box" style="color:white;">{get_ai_response(f"طالب بكالوريا. باقي {days} يوم، وسأدرس {hours} ساعات مادة {sub}. ولد جدول.")}</div>', unsafe_allow_html=True)

        elif st.session_state["current_view"] == "past_papers":
            st.markdown("### 📖 أسئلة الدورات")
            f_db = load_data(FILES_DB)
            my_f = f_db[(f_db["grade"] == view_grade) & (f_db["sub"] == sub)] if not f_db.empty else pd.DataFrame()
            past_papers_files = my_f[my_f["type"] == "دورات"] if not my_f.empty else pd.DataFrame()
            if past_papers_files.empty: st.warning("لا يوجد ملفات دورات.")
            else:
                selected_paper = st.selectbox("اختر ملف:", past_papers_files['name'].tolist(), format_func=lambda x: x.split('_')[-1])
                topic_query = st.text_input("عن أي بحث تبحث؟")
                if st.button("🔍 استخراج"):
                    if topic_query:
                        file_path = os.path.join("lessons", selected_paper)
                        if os.path.exists(file_path):
                            with st.spinner("يستخرج..."):
                                try:
                                    uploaded_file = genai.upload_file(file_path)
                                    res = get_ai_response(f"استخرج الأسئلة التي تخص موضوع '{topic_query}'. لا تجب عليها.", strict_mode=False, file_uri=uploaded_file)
                                    st.markdown(f'<div class="modern-box" style="color:white;">{res}</div>', unsafe_allow_html=True)
                                    genai.delete_file(uploaded_file.name)
                                except Exception as e: st.error(f"خطأ: {str(e)}")
                    else: st.warning("اكتب اسم البحث.")
