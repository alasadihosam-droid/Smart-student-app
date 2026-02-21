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

# ==========================================
# 0. نظام حماية التزامن لملفات الـ CSV (لمنع ضياع البيانات)
# ==========================================
db_lock = threading.Lock()

def save_data(df, path):
    with db_lock:
        df.to_csv(path, index=False)

def load_data(path):
    with db_lock:
        try: 
            return pd.read_csv(path)
        except Exception as e: 
            print(f"Error loading {path}: {e}")
            return pd.DataFrame()

def init_db(path, columns):
    if not os.path.exists(path): 
        pd.DataFrame(columns=columns).to_csv(path, index=False)

# ==========================================
# 1. نظام التشفير الآمن لكلمات المرور (Salting)
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
# 2. وظائف استخراج النص والـ RAG الاحترافي (مُسرّع بـ Numpy)
# ==========================================
def extract_and_chunk_pdf_smart(pdf_path, max_chunk_size=1500, overlap_size=200):
    chunks = []
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join([page.extract_text() or "" for page in reader.pages])
            
            if not text.strip():
                st.warning(f"⚠️ تنبيه: لم نتمكن من استخراج نص من الملف '{os.path.basename(pdf_path)}'.")
                return []
            
            paragraphs = text.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) <= max_chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    overlap_text = current_chunk[-overlap_size:] if len(current_chunk) > overlap_size else current_chunk
                    current_chunk = overlap_text + "\n" + para + "\n\n"
                    
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
    except Exception as e:
        st.error(f"⚠️ حدث خطأ أثناء قراءة الملف: {str(e)}")
    return chunks

@st.cache_data 
def get_and_save_embeddings(pdf_path):
    embed_file = pdf_path.replace('.pdf', '_embeddings.json')
    
    if os.path.exists(embed_file):
        try:
            with open(embed_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading embeddings: {e}")
            
    chunks = extract_and_chunk_pdf_smart(pdf_path)
    embeddings_data = []
    
    for chunk in chunks:
        try:
            vec = genai.embed_content(model="models/embedding-001", content=chunk)['embedding']
            embeddings_data.append({"text": chunk, "vector": vec})
        except Exception as e:
            print(f"Error embedding chunk: {e}")
            continue
            
    with open(embed_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f)
        
    return embeddings_data

def get_best_context_smart(query, pdf_path, top_k=3):
    embeddings_data = get_and_save_embeddings(pdf_path)
    if not embeddings_data: return ""
    
    try:
        query_embed = genai.embed_content(model="models/embedding-001", content=query)['embedding']
        query_vec = np.array(query_embed)
    except Exception as e: 
        print(f"Error embedding query: {e}")
        return ""
    
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
    if "GEMINI_API_KEY" in st.secrets:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    else:
        st.error("⚠️ مفتاح API غير موجود. يرجى إضافة GEMINI_API_KEY في ملف Secrets.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ خطأ في الوصول إلى Secrets: {e}")
    st.stop()

genai.configure(api_key=API_KEY)

OWNER_PASS_RAW = st.secrets.get("OWNER_PASSWORD", "hosam031007")
OWNER_PASS_HASH = hash_password_secure(OWNER_PASS_RAW)

def get_ai_response(prompt, image=None, audio=None, strict_mode=False, context_text="", file_uri=None):
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        safe_models = [m for m in available_models if "2.5" not in m]
        if not safe_models: return "⚠️ عذراً، جميع الموديلات المتاحة في حسابك غير مجانية."
        
        system_instruction = ""
        if strict_mode:
            if context_text:
                system_instruction = f"""تعليمات صارمة: أنت معلم سوري. أجب على السؤال بناءً على هذا النص المرفوع من الأستاذ حصراً. 
                لا تقم بإضافة أي معلومات أو قوانين خارجية. إذا لم تكن الإجابة موجودة في النص، قل: 
                'عذراً، إجابة هذا السؤال غير موجودة في نوطة الأستاذ المرفوعة.'
                
                النص المرجعي:
                {context_text}"""
            else:
                system_instruction = "تعليمات صارمة: أنت معلم سوري. التزم حصراً بالمعلومات الموجودة في المنهاج السوري. لا تقم بإضافة أي معلومات خارجية."
            
            prompt = system_instruction + "\n\nسؤال/طلب الطالب:\n" + prompt

        for model_name in safe_models:
            try:
                model = genai.GenerativeModel(model_name)
                contents = []
                if file_uri: contents.append(file_uri)
                contents.append(prompt)
                if image: contents.append(image)
                if audio: contents.append(audio)
                return model.generate_content(contents).text
            except Exception as e: 
                print(f"Model {model_name} failed: {e}")
                continue 
        return "⚠️ تم رفض الاتصال أو لا يوجد موديل متاح. جرب تشغيل VPN."
    except Exception as e: return f"⚠️ خطأ عام: {str(e)}"

def check_cheating_smart(text1, text2):
    prompt = f"""أنت خبير في كشف الغش الأكاديمي.
    لدينا إجابتان من طالبين مختلفين لنفس السؤال العلمي.
    الإجابة الأولى: "{text1}"
    الإجابة الثانية: "{text2}"
    مهمتك: هل هناك تلاعب واضح أو نسخ ولصق؟ أريد إجابتك بالصيغة التالية حصراً:
    النسبة: [النسبة المئوية لاحتمالية الغش رقماً]
    التحليل: [جملة واحدة سريعة تشرح السبب]"""
    return get_ai_response(prompt, strict_mode=False)

def speak_text(text):
    try:
        tts = gTTS(text=text[:250], lang='ar')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e: 
        print(f"TTS Error: {e}")
        return None

# ==========================================
# 4. تهيئة قواعد البيانات والمجلدات
# ==========================================
for folder in ['lessons', 'exams', 'db', 'profiles']:
    if not os.path.exists(folder): os.makedirs(folder)

USERS_DB = "db/users.csv"
FILES_DB = "db/files.csv"
GRADES_DB = "db/grades.csv"
NOTIFY_DB = "db/notifications.csv" 
TEACHER_SUBJECTS_DB = "db/teacher_subjects.csv" 
CODES_DB = "db/codes.csv" 
BROADCAST_DB = "db/broadcasts.csv" 

init_db(USERS_DB, ["user", "pass", "role", "grade", "fb_link", "is_new", "is_premium", "invited_by"]) 
init_db(FILES_DB, ["name", "grade", "sub", "type", "date", "uploader", "chapter_num"]) 
init_db(GRADES_DB, ["user", "sub", "score", "date"])
init_db(NOTIFY_DB, ["sender", "message", "date"])
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
    changed = False
    if "uploader" not in db_files_check.columns: db_files_check["uploader"] = "غير معروف"; changed = True
    if "chapter_num" not in db_files_check.columns: db_files_check["chapter_num"] = 1; changed = True
    if changed: save_data(db_files_check, FILES_DB)

# ==========================================
# 5. إعدادات الواجهة والترحيب الزمني 
# ==========================================
st.set_page_config(page_title="منصة سند التعليمية", layout="wide", page_icon="🎓")

hour = datetime.now().hour
if 5 <= hour < 12: time_greeting = "صباح الخير ☀️"
elif 12 <= hour < 18: time_greeting = "طاب نهارك 🌤️"
else: time_greeting = "مساء الخير 🌙"

# ==========================================
# ستايل الـ CSS الاحترافي الجديد (متعوب عليه)
# ==========================================
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    html, body, [class*="st-"] { scroll-behavior: smooth; overscroll-behavior-y: none; }
    
    /* لون خلفية التطبيق كامل ليعطي تباين مع الأزرار */
    .stApp { 
        background-color: #f4f6f9; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
    }
    
    /* الصناديق والمربعات البيضاء الأنيقة */
    .modern-box { 
        padding: 25px; 
        background: #ffffff; 
        border-radius: 20px; 
        border-right: 6px solid #1E88E5; 
        box-shadow: 0 10px 25px rgba(0,0,0,0.06); 
        margin-bottom: 25px; 
        transition: transform 0.3s ease;
    }
    .modern-box:hover { transform: translateY(-3px); }
    
    /* صندوق الإشعارات */
    .broadcast-box { 
        padding: 20px; 
        background: linear-gradient(135deg, #FF9800, #FFB74D); 
        border-radius: 16px; 
        margin-bottom: 20px; 
        color: #ffffff; 
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0 6px 15px rgba(255, 152, 0, 0.3);
    }
    
    /* العناوين المتدرجة */
    .welcome-title { 
        font-size: 2.2rem; 
        font-weight: 900; 
        text-align: center; 
        background: linear-gradient(to left, #1E88E5, #8E24AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .programmer-tag { font-size: 1rem; text-align: center; font-weight: 700; color: #78909C; letter-spacing: 1px; }
    .teacher-badge { font-size: 0.85rem; background: linear-gradient(135deg, #1E88E5, #1565C0); color: white; padding: 6px 14px; border-radius: 20px; margin-left: 10px; float: left; font-weight: bold; box-shadow: 0 4px 10px rgba(30, 136, 229, 0.3);}
    
    /* تصميم الأيقونات (الأزرار المربعة) - التحديث السحري */
    div[data-testid="column"] button { 
        width: 100%; 
        height: 130px; 
        border-radius: 24px; 
        /* لون كحلي/أزرق غامق متباين جداً مع الخلفية الفاتحة */
        background: linear-gradient(135deg, #2c3e50, #3498db); 
        color: #ffffff; 
        font-size: 19px; 
        font-weight: 800; 
        border: 2px solid rgba(255,255,255,0.1); 
        box-shadow: 0 10px 25px rgba(52, 152, 219, 0.3), inset 0 2px 5px rgba(255,255,255,0.2); 
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); 
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        justify-content: center; 
        letter-spacing: 0.5px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    div[data-testid="column"] button:hover { 
        transform: translateY(-8px); 
        box-shadow: 0 15px 35px rgba(52, 152, 219, 0.5), inset 0 2px 5px rgba(255,255,255,0.3); 
        background: linear-gradient(135deg, #34495e, #2980b9);
    }
    div[data-testid="column"] button:active { 
        transform: translateY(2px) scale(0.96); 
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4); 
    }
    
    /* زر العودة */
    .back-btn>button { 
        background: linear-gradient(135deg, #FF416C, #FF4B2B) !important; 
        height: 60px !important; 
        border-radius: 16px !important; 
        margin-bottom: 30px; 
        font-size: 18px !important; 
        font-weight: 800 !important; 
        border: none !important; 
        color: white !important; 
        box-shadow: 0 8px 20px rgba(255, 65, 108, 0.4) !important; 
        transition: all 0.3s ease !important;
    }
    .back-btn>button:hover { 
        transform: translateY(-4px) !important; 
        box-shadow: 0 12px 25px rgba(255, 65, 108, 0.6) !important; 
    }
    </style>
    """, unsafe_allow_html=True)

subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "إنكليزي", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "عربي", "إنكليزي", "وطنية"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "فرنسي", "عربي", "إنكليزي", "وطنية"]
}

if "user_data" not in st.session_state: st.session_state["user_data"] = None
if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
if "oral_exam_history" not in st.session_state: st.session_state["oral_exam_history"] = []
if "last_active" not in st.session_state: st.session_state["last_active"] = time.time()
if "current_view" not in st.session_state: st.session_state["current_view"] = "home" 

if st.session_state["user_data"] is not None:
    if time.time() - st.session_state["last_active"] > 3600:
        st.session_state["user_data"] = None
        st.warning("تم تسجيل الخروج تلقائياً لأسباب أمنية (Timeout). يرجى تسجيل الدخول مجدداً.")
    st.session_state["last_active"] = time.time()

# ==========================================
# 6. شاشة الدخول والتسجيل
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
                if u == "Hosam" and verify_password(OWNER_PASS_HASH, p):
                    st.session_state["user_data"] = {"user": u, "role": "Owner", "grade": "الكل", "is_new": False, "is_premium": True}
                    st.rerun()
                else:
                    users = load_data(USERS_DB)
                    if not users.empty:
                        match = users[users["user"] == u]
                        if not match.empty:
                            stored_pass = match.iloc[0]["pass"]
                            if verify_password(stored_pass, p):
                                st.session_state["user_data"] = match.iloc[0].to_dict()
                                st.rerun()
                            else: st.error("⚠️ عذراً، كلمة المرور غير صحيحة")
                        else: st.error("⚠️ عذراً، اسم المستخدم غير موجود")
                    else: st.warning("لا يوجد مستخدمين مسجلين بعد.")
    
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
            else:
                users = load_data(USERS_DB)
                if not users.empty and nu in users['user'].values: st.error("⚠️ الاسم موجود مسبقاً.")
                else:
                    secure_pass = hash_password_secure(np)
                    new_user = pd.DataFrame([{"user": nu, "pass": secure_pass, "role": "طالب", "grade": ng, "fb_link": fb, "is_new": False, "is_premium": False, "invited_by": invite}])
                    save_data(pd.concat([users, new_user], ignore_index=True), USERS_DB)
                    st.success("🎉 تم إنشاء الحساب! سجل دخولك الآن.")

# ==========================================
# 7. شاشات المستخدمين (بعد تسجيل الدخول)
# ==========================================
else:
    user = st.session_state["user_data"]
    
    if user["role"] == "أستاذ" and user.get("is_new", True):
        st.markdown(f'<div class="modern-box"><div class="welcome-title">أهلاً وسهلاً بك يا أستاذنا الفاضل 👨‍🏫</div></div>', unsafe_allow_html=True)
        st.info("لتكتمل إعدادات حسابك، يرجى اختيار الصف والمادة التي تدرسها لترتبط ملفاتك بها مباشرة.")
        col_g, col_s = st.columns(2)
        sel_grade = col_g.selectbox("الصف الذي تدرسه:", list(subs_map.keys()) + ["كل الصفوف"])
        if sel_grade == "كل الصفوف":
            all_subs = list(set([item for sublist in subs_map.values() for item in sublist]))
            sel_sub = col_s.selectbox("مادتك الاختصاصية:", all_subs)
        else: sel_sub = col_s.selectbox("مادتك الاختصاصية:", subs_map[sel_grade])
        pic = st.file_uploader("ارفع صورتك الشخصية (اختياري)", type=['png', 'jpg', 'jpeg'])
        if st.button("حفظ الإعدادات والبدء 🚀"):
            if pic: Image.open(pic).save(f"profiles/{user['user']}.png")
            ts_db = load_data(TEACHER_SUBJECTS_DB)
            save_data(pd.concat([ts_db, pd.DataFrame([{"teacher_name": user["user"], "grade": sel_grade, "subject": sel_sub}])], ignore_index=True), TEACHER_SUBJECTS_DB)
            users_df = load_data(USERS_DB)
            users_df.loc[users_df['user'] == user['user'], 'is_new'] = False
            save_data(users_df, USERS_DB)
            st.session_state["user_data"]["is_new"] = False
            st.success("تم إعداد حسابك بنجاح!")
            st.rerun()
        st.stop() 
    
    teacher_grade, teacher_sub = "", ""
    if user["role"] == "أستاذ":
        ts_db = load_data(TEACHER_SUBJECTS_DB)
        t_match = ts_db[ts_db["teacher_name"] == user["user"]]
        if not t_match.empty:
            teacher_grade, teacher_sub = t_match.iloc[0]["grade"], t_match.iloc[0]["subject"]

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
        if st.button("🔴 تسجيل الخروج"):
            st.session_state["user_data"] = None
            st.rerun()

    # ----------------------------------------
    # واجهة الإدارة (Owner)
    # ----------------------------------------
    if user["role"] == "Owner":
        st.header(f"👑 لوحة تحكم الإدارة الشاملة - {time_greeting}")
        t_users, t_teachers, t_files, t_codes, t_notify, t_anti_cheat = st.tabs(["👥 الطلاب", "👨‍🏫 الأساتذة", "📁 الملفات", "💳 الاشتراكات", "📩 رسائل الأساتذة", "🕵️ كشف الغش"])
        
        with t_users:
            u_df = load_data(USERS_DB)
            if not u_df.empty:
                st.data_editor(u_df[u_df['role'] == 'طالب'], num_rows="dynamic", use_container_width=True)

        with t_teachers:
            st.markdown("### ➕ إضافة أستاذ جديد")
            c1, c2 = st.columns(2)
            t_name, t_pass = c1.text_input("اسم الأستاذ"), c2.text_input("كلمة مرور الأستاذ", type="password")
            if st.button("إنشاء حساب الأستاذ") and t_name and t_pass:
                users = load_data(USERS_DB)
                if t_name in users['user'].values: st.error("الاسم موجود.")
                else:
                    secure_t_pass = hash_password_secure(t_pass)
                    save_data(pd.concat([users, pd.DataFrame([{"user": t_name, "pass": secure_t_pass, "role": "أستاذ", "grade": "الكل", "fb_link": "معلم", "is_new": True, "is_premium": True, "invited_by": ""}])], ignore_index=True), USERS_DB)
                    st.success("تم التفعيل!")
                    st.rerun()

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
                st.success("تم الحذف!")
                st.rerun()

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
                st.success(f"تم توليد {num_codes} كود فريد وجديد بنجاح!")

        with t_notify:
            n_df = load_data(NOTIFY_DB)
            st.dataframe(n_df, use_container_width=True)
            if not n_df.empty and st.button("مسح جميع التنويهات"): 
                save_data(pd.DataFrame(columns=["sender", "message", "date"]), NOTIFY_DB)
                st.rerun()
                
        with t_anti_cheat:
            st.info("أدخل إجابتين لطالبين مختلفين. الذكاء الاصطناعي سيقوم بتحليل التشابه بدقة عالية.")
            text1 = st.text_area("إجابة الطالب الأول:")
            text2 = st.text_area("إجابة الطالب الثاني:")
            if st.button("فحص الغش 🕵️"):
                with st.spinner("جاري التحليل المعمق..."):
                    result = check_cheating_smart(text1, text2)
                    st.markdown(f'<div class="modern-box">{result}</div>', unsafe_allow_html=True)

    # ----------------------------------------
    # واجهة الطالب والأستاذ المشتركة 
    # ----------------------------------------
    elif user["role"] in ["طالب", "أستاذ"]:
        if user["role"] == "أستاذ":
            st.markdown(f'<div class="modern-box"><div class="welcome-title">👨‍🏫 أهلاً بك أستاذ {user["user"]}</div><div class="programmer-tag">{teacher_sub} - {teacher_grade}</div></div>', unsafe_allow_html=True)
            view_grade = st.selectbox("اختر الصف:", ["التاسع", "البكالوريا العلمي", "البكالوريا الأدبي"]) if teacher_grade == "كل الصفوف" else teacher_grade
            sub = teacher_sub
        else:
            st.markdown(f'<div class="modern-box"><div class="welcome-title">{time_greeting} يا بطل!</div><div class="programmer-tag">الصف: {user["grade"]}</div></div>', unsafe_allow_html=True)
            view_grade, sub = user["grade"], st.selectbox("اختر المادة:", subs_map[user["grade"]])
            
            b_df = load_data(BROADCAST_DB)
            if not b_df.empty:
                for _, b in b_df[(b_df['grade'] == view_grade) & (b_df['subject'] == sub)].tail(3).iterrows():
                    st.markdown(f"<div class='broadcast-box'><b>🔔 إشعار من {b['sender']}:</b> {b['message']}</div>", unsafe_allow_html=True)

        if st.session_state["current_view"] != "home":
            st.markdown('<div class="back-btn">', unsafe_allow_html=True)
            if st.button("🔙 العودة للقائمة الرئيسية", use_container_width=True):
                st.session_state["current_view"] = "home"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------------------------
        # الصفحة الرئيسية 
        # -----------------------------------------
        if st.session_state["current_view"] == "home":
            if user["role"] == "أستاذ":
                col1, col2 = st.columns(2)
                if col1.button("📢\nإرسال إشعار"): st.session_state["current_view"] = "notify"; st.rerun()
                if col2.button("📤\nرفع الملفات"): st.session_state["current_view"] = "upload"; st.rerun()
                
                col3, col4 = st.columns(2)
                if col3.button("📚\nالمكتبة"): st.session_state["current_view"] = "library"; st.rerun()
                if col4.button("🤖\nالمعلم الذكي"): st.session_state["current_view"] = "ai_teacher"; st.rerun()
                
                col5, col6 = st.columns(2)
                if col5.button("📸\nعدسة الذكاء"): st.session_state["current_view"] = "lens"; st.rerun()
                if col6.button("📝\nالامتحانات"): st.session_state["current_view"] = "exams"; st.rerun()
                
                col7, col8 = st.columns(2)
                if col7.button("📖\nأسئلة الدورات"): st.session_state["current_view"] = "past_papers"; st.rerun()
            
            else: 
                col1, col2 = st.columns(2)
                if col1.button("📚\nالمكتبة"): st.session_state["current_view"] = "library"; st.rerun()
                if col2.button("🤖\nالمعلم الذكي"): st.session_state["current_view"] = "ai_teacher"; st.rerun()
                
                col3, col4 = st.columns(2)
                if col3.button("📸\nعدسة الذكاء"): st.session_state["current_view"] = "lens"; st.rerun()
                if col4.button("📝\nالامتحانات"): st.session_state["current_view"] = "exams"; st.rerun()
                
                col5, col6 = st.columns(2)
                if col5.button("📅\nخطة الدراسة"): st.session_state["current_view"] = "plan"; st.rerun()
                if col6.button("📖\nأسئلة الدورات"): st.session_state["current_view"] = "past_papers"; st.rerun()

        # -----------------------------------------
        # تفاصيل الأقسام
        # -----------------------------------------
        elif st.session_state["current_view"] == "notify" and user["role"] == "أستاذ":
            st.subheader("📢 إرسال إشعار للطلاب")
            b_msg = st.text_area("اكتب الإشعار هنا لطلابك:")
            if st.button("🚀 إرسال فوراً") and b_msg:
                save_data(pd.concat([load_data(BROADCAST_DB), pd.DataFrame([{"sender": user["user"], "grade": view_grade, "subject": sub, "message": b_msg, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}])], ignore_index=True), BROADCAST_DB)
                st.success("تم نشر الإشعار بنجاح!")

        elif st.session_state["current_view"] == "upload" and user["role"] == "أستاذ":
            st.subheader("📤 رفع الملفات وتحليلها")
            with st.form("upload_form", clear_on_submit=True):
                uploaded_file = st.file_uploader("اختر ملف (PDF حصراً)", type="pdf")
                file_name_input = st.text_input("اسم الملف (مثال: نوطة الوحدة الأولى)")
                ch_num = st.number_input("رقم البحث", min_value=1, value=1)
                
                type_f = st.radio("تصنيف الملف:", ["بحث (درس/نوطة)", "نموذج امتحاني", "سلم تصحيح", "أسئلة دورات"], horizontal=True)
                
                if st.form_submit_button("🚀 رفع الملف للمنصة"):
                    if uploaded_file:
                        if uploaded_file.type != "application/pdf" or not uploaded_file.name.lower().endswith('.pdf'):
                            st.error("⚠️ غير مسموح برفع ملفات غير الـ PDF لأسباب أمنية.")
                        else:
                            internal_type = "بحث" if "بحث" in type_f else "نموذج" if "نموذج" in type_f else "دورات" if "دورات" in type_f else "سلم"
                            f_name = f"{internal_type}_{sub}_{file_name_input.replace(' ', '_') if file_name_input else uploaded_file.name.replace(' ', '_')}"
                            if not f_name.endswith('.pdf'): f_name += '.pdf'
                            
                            folder = "lessons" if internal_type in ["بحث", "دورات"] else "exams"
                            file_save_path = os.path.join(folder, f_name)
                            
                            with open(file_save_path, "wb") as f: f.write(uploaded_file.getbuffer())
                            
                            save_data(pd.concat([load_data(FILES_DB), pd.DataFrame([{"name": f_name, "grade": view_grade, "sub": sub, "type": internal_type, "date": datetime.now().strftime("%Y-%m-%d"), "uploader": user["user"], "chapter_num": ch_num}])], ignore_index=True), FILES_DB)
                            
                            if internal_type in ["بحث", "دورات"]:
                                with st.spinner("جاري قراءة الملف وتجهيز الذكاء الاصطناعي للإجابة منه لاحقاً... 🤖"):
                                    get_and_save_embeddings(file_save_path)
                            
                            st.success("تم الرفع والتجهيز بنجاح! 🎉")

        elif st.session_state["current_view"] == "library":
            st.subheader("📚 مكتبة الملفات والنوط")
            f_db = load_data(FILES_DB)
            my_f = f_db[(f_db["grade"] == view_grade) & (f_db["sub"] == sub)] if not f_db.empty else pd.DataFrame()
            if my_f.empty: st.info("المكتبة فارغة حالياً.")
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
                                with open(path, "rb") as f: st.download_button(f"📥 {r['name'].split('_')[-1]} ({r['type']})", f, file_name=r['name'], key=r['name'])
                        with c_f2: st.markdown(f"<div class='teacher-badge'>أ. {r.get('uploader', 'غير معروف')}</div>", unsafe_allow_html=True)

        elif st.session_state["current_view"] == "ai_teacher":
            st.subheader("🤖 المعلم الذكي (مانع الهلوسة)")
            f_db = load_data(FILES_DB)
            my_f = f_db[(f_db["grade"] == view_grade) & (f_db["sub"] == sub)] if not f_db.empty else pd.DataFrame()
            available_files = my_f[my_f["type"] == "بحث"] if not my_f.empty else pd.DataFrame()
            best_context = ""
            file_path = ""
            
            if not available_files.empty:
                selected_file = st.selectbox("📚 اختر النوطة التي تدرسها لنسأل منها:", available_files['name'].tolist(), format_func=lambda x: x.split('_')[-1])
                file_path = os.path.join("lessons", selected_file)
            else: st.warning("⚠️ لا يوجد نوط مرفوعة. سيجيب من معلوماته العامة.")

            style = st.radio("طريقة الشرح:", ["علمي صارم (من النوطة)", "بالمشرمحي"], horizontal=True)
            for msg in st.session_state["chat_history"]: st.chat_message(msg["role"]).write(msg["content"])
            
            if q := st.chat_input("اسأل معلمك الذكي..."):
                st.session_state["chat_history"].append({"role": "user", "content": q})
                st.chat_message("user").write(q)
                
                with st.spinner("يبحث في النوطة..."):
                    strict = True if "صارم" in style else False
                    pr = f"أجب لمادة {sub} صف {view_grade}: {q}\n" if not strict else q
                    if "بالمشرمحي" in style: pr += " اشرحها عامية سورية بأمثلة واقعية"
                    
                    if file_path and os.path.exists(file_path):
                        # نرسل الان أفضل 3 مقاطع بفضل التحديث
                        best_context = get_best_context_smart(q, file_path, top_k=3)
                        
                    ans = get_ai_response(pr, strict_mode=strict, context_text=best_context)
                st.session_state["chat_history"].append({"role": "assistant", "content": ans})
                st.chat_message("assistant").write(ans)

        elif st.session_state["current_view"] == "lens":
            st.subheader("📸 عدسة الذكاء (التصحيح الآلي)")
            v_mode = st.radio("الخدمة المطلوبة:", ["شرح مسألة من الصورة", "تصحيح حلي بناءً على السلالم"])
            if img := st.file_uploader("التقط أو ارفع صورة:", type=["jpg", "png", "jpeg"]):
                if st.button("🚀 بدء التحليل"):
                    with st.spinner("يتم فحص الصورة بدقة..."):
                        st.info(get_ai_response(f"أنت معلم مادة {sub}. " + ("اشرح الحل المرفق" if v_mode=="شرح مسألة" else "صحح الحل بناء على السلالم السورية وأعط درجة."), image=Image.open(img), strict_mode=True))

        elif st.session_state["current_view"] == "exams":
            st.subheader("📝 قسم الامتحانات والتسميع")
            if st.button("🎯 توليد أسئلة أتمتة شاملة"): 
                st.markdown(f'<div class="modern-box">{get_ai_response(f"ولد نموذج وزاري سوري لمادة {sub} معتمداً حصراً على أسلوب النماذج المرفوعة.", strict_mode=True)}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("🗣️ **التسميع الشفهي الذكي (تحدث ليتم التقييم)**")
            st.info("اضغط على المايكروفون للإجابة شفهياً. سيقوم النظام بتحليل نطقك ومعلوماتك.")
            
            audio_val = st.audio_input("🎤 سجل إجابتك هنا:")
            if audio_val:
                st.audio(audio_val)
                with st.spinner("يستمع لإجابتك ويقيّمها..."):
                    audio_data = {"mime_type": "audio/wav", "data": audio_val.getvalue()}
                    o_ans = get_ai_response(f"استمع إلى إجابة الطالب بمادة {sub}. اكتب ما قاله حرفياً، ثم صحح الإجابة علمياً واطرح سؤالاً جديداً.", audio=audio_data, strict_mode=True)
                    st.success(o_ans)

        elif st.session_state["current_view"] == "plan" and user["role"] == "طالب":
            st.subheader("📅 المولد السحري لخطة الدراسة")
            c_plan1, c_plan2 = st.columns(2)
            days_left = c_plan1.number_input("كم يوم متبقي للامتحان؟", 1, value=20)
            hours_daily = c_plan2.slider("كم ساعة تستطيع الدراسة باليوم؟", 1, 15, 6)
            if st.button("توليد الخطة 🪄"):
                with st.spinner("جاري التخطيط لمستقبلك..."):
                    plan_prompt = f"أنا طالب سوري في {view_grade}. متبقي {days_left} يوماً للامتحان، سأدرس {hours_daily} ساعات يومياً مادة {sub}. قم بتوليد جدول دراسي يومي واقعي مع فترات مراجعة."
                    st.markdown(f'<div class="modern-box">{get_ai_response(plan_prompt)}</div>', unsafe_allow_html=True)

        elif st.session_state["current_view"] == "past_papers":
            st.subheader("📖 مستكشف أسئلة الدورات السابقة")
            st.info("الذكاء الاصطناعي سيستخرج لك الأسئلة التي وردت في الدورات السابقة للبحث الذي تختاره حصراً بدقة عالية جداً بفضل الـ File API.")
            
            f_db = load_data(FILES_DB)
            my_f = f_db[(f_db["grade"] == view_grade) & (f_db["sub"] == sub)] if not f_db.empty else pd.DataFrame()
            past_papers_files = my_f[my_f["type"] == "دورات"] if not my_f.empty else pd.DataFrame()
            
            if past_papers_files.empty:
                st.warning("⚠️ لم يقم الأساتذة برفع أي ملف 'أسئلة دورات' لهذه المادة حتى الآن.")
            else:
                selected_paper = st.selectbox("اختر ملف الدورات المرفوع:", past_papers_files['name'].tolist(), format_func=lambda x: x.split('_')[-1])
                topic_query = st.text_input("عن أي بحث أو موضوع تبحث؟ (مثال: النواس المرن، المغناطيسية، الطفرات):")
                
                if st.button("🔍 استخراج أسئلة الدورات لهذا البحث"):
                    if topic_query:
                        file_path = os.path.join("lessons", selected_paper)
                        if os.path.exists(file_path):
                            with st.spinner("يقرأ ملف الدورات ويستخرج الأسئلة المطلوبة..."):
                                try:
                                    # رفع الملف مؤقتاً لخوادم جوجل لمعالجته كقطعة واحدة
                                    uploaded_gemini_file = genai.upload_file(file_path)
                                    
                                    prompt = f"""أنت خبير في المنهاج السوري. اقرأ ملف أسئلة الدورات السورية المرفق هذا، واستخرج **فقط** الأسئلة التي تخص موضوع أو بحث '{topic_query}'.
                                    - اذكر صيغة السؤال كما ورد في الدورة تماماً.
                                    - اذكر السنة أو الدورة إذا كانت مكتوبة بجانب السؤال.
                                    - لا تقم بالإجابة على الأسئلة، فقط استخرجها ورتبها في قائمة."""
                                    
                                    res = get_ai_response(prompt, strict_mode=False, file_uri=uploaded_gemini_file)
                                    st.markdown(f'<div class="modern-box">{res}</div>', unsafe_allow_html=True)
                                    
                                    # حذف الملف من خوادم جوجل بعد الانتهاء للحفاظ على المساحة والخصوصية
                                    genai.delete_file(uploaded_gemini_file.name)
                                except Exception as e:
                                    st.error(f"حدث خطأ أثناء معالجة الملف: {str(e)}")
                        else:
                            st.error("عذراً، ملف الدورات غير موجود في المجلد.")
                    else:
                        st.warning("يرجى كتابة اسم البحث أو الموضوع أولاً.")
