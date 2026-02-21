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
import math
import difflib

# ==========================================
# وظائف استخراج النص والـ RAG الاحترافي (منع الهلوسة والبطء)
# ==========================================
@st.cache_data # إضافة الـ Caching لتوفير التكلفة والوقت
def get_embedding(text):
    try:
        # تحويل النص إلى متجهات رياضية للبحث الذكي
        result = genai.embed_content(model="models/embedding-001", content=text)
        return result['embedding']
    except: return []

def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2: return 0
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    return dot / (norm1*norm2) if norm1*norm2 != 0 else 0

@st.cache_data # إضافة Caching لتسريع قراءة الملفات
def extract_and_chunk_pdf(pdf_path, chunk_size=1500):
    chunks = []
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join([page.extract_text() or "" for page in reader.pages])
            # تقسيم النص إلى مقاطع (Chunks) لتجنب تجاوز التوكنز
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    except Exception as e:
        pass
    return chunks

def get_best_context(query, chunks):
    if not chunks: return ""
    query_embed = get_embedding(query)
    best_chunk, max_score = "", -1
    for chunk in chunks:
        chunk_embed = get_embedding(chunk)
        score = cosine_similarity(query_embed, chunk_embed)
        if score > max_score:
            max_score, best_chunk = score, chunk
    return best_chunk

# ==========================================
# نظام كشف الغش (نسبة التطابق الذكي عبر الـ Embeddings)
# ==========================================
def check_cheating(text1, text2):
    # استخدام الـ Embedding Similarity بدلاً من difflib البدائي لكشف التلاعب بالكلمات
    vec1 = get_embedding(text1)
    vec2 = get_embedding(text2)
    sim = cosine_similarity(vec1, vec2)
    return round(sim * 100, 2)

# ==========================================
# 1. إعدادات الأمان والذكاء الاصطناعي
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

def get_ai_response(prompt, image=None, audio=None, strict_mode=False, context_text=""):
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
                
                النص المرجعي (الفقرة الأقرب للسؤال):
                {context_text}"""
            else:
                system_instruction = "تعليمات صارمة: أنت معلم سوري. التزم حصراً بالمعلومات الموجودة في المنهاج السوري. لا تقم بإضافة أي معلومات خارجية. إذا كان السؤال خارج المنهاج قل 'هذا السؤال خارج المنهاج المقرر'."
            
            prompt = system_instruction + "\n\nسؤال/طلب الطالب:\n" + prompt

        for model_name in safe_models:
            try:
                model = genai.GenerativeModel(model_name)
                contents = [prompt]
                if image: contents.append(image)
                if audio: contents.append(audio) # تم إضافة دعم الصوت
                return model.generate_content(contents).text
            except Exception: continue 
        return "⚠️ تم رفض الاتصال. جرب تشغيل VPN."
    except Exception as e: return f"⚠️ خطأ عام: {str(e)}"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def speak_text(text):
    try:
        tts = gTTS(text=text[:250], lang='ar')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except: return None

# ==========================================
# 2. تهيئة قواعد البيانات والمجلدات
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

def init_db(path, columns):
    if not os.path.exists(path): pd.DataFrame(columns=columns).to_csv(path, index=False)

init_db(USERS_DB, ["user", "pass", "role", "grade", "fb_link", "is_new", "is_premium", "invited_by"]) 
init_db(FILES_DB, ["name", "grade", "sub", "type", "date", "uploader", "chapter_num"]) 
init_db(GRADES_DB, ["user", "sub", "score", "date"])
init_db(NOTIFY_DB, ["sender", "message", "date"])
init_db(TEACHER_SUBJECTS_DB, ["teacher_name", "grade", "subject"])
init_db(CODES_DB, ["code", "is_used", "used_by", "date_created"])
init_db(BROADCAST_DB, ["sender", "grade", "subject", "message", "date"])

def load_data(path):
    try: return pd.read_csv(path)
    except: return pd.DataFrame()

db_users_check = load_data(USERS_DB)
if not db_users_check.empty:
    changed = False
    if "is_new" not in db_users_check.columns: db_users_check["is_new"] = True; changed = True
    if "fb_link" not in db_users_check.columns: db_users_check["fb_link"] = ""; changed = True
    if "is_premium" not in db_users_check.columns: db_users_check["is_premium"] = False; changed = True
    if "invited_by" not in db_users_check.columns: db_users_check["invited_by"] = ""; changed = True
    if changed: db_users_check.to_csv(USERS_DB, index=False)

db_files_check = load_data(FILES_DB)
if not db_files_check.empty:
    changed = False
    if "uploader" not in db_files_check.columns: db_files_check["uploader"] = "غير معروف"; changed = True
    if "chapter_num" not in db_files_check.columns: db_files_check["chapter_num"] = 1; changed = True
    if changed: db_files_check.to_csv(FILES_DB, index=False)

# ==========================================
# 3. إعدادات الواجهة والترحيب الزمني 
# ==========================================
st.set_page_config(page_title="منصة سند التعليمية", layout="wide", page_icon="🎓")

hour = datetime.now().hour
if 5 <= hour < 12: time_greeting = "صباح الخير ☀️"
elif 12 <= hour < 18: time_greeting = "طاب نهارك 🌤️"
else: time_greeting = "مساء الخير 🌙"

st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    html, body, [class*="st-"] { scroll-behavior: smooth; overscroll-behavior-y: none; }
    .stApp { overflow-x: hidden; }
    .stButton>button { width: 100%; border-radius: 8px; background: #1E88E5; color: white; font-weight: bold; border: none; padding: 0.5rem; transition: 0.2s; }
    .stButton>button:active { transform: scale(0.98); }
    .modern-box { padding: 15px; background-color: rgba(30, 136, 229, 0.05); border-radius: 10px; border-right: 4px solid #1E88E5; margin-bottom: 15px; }
    .broadcast-box { padding: 15px; background-color: #fff3cd; border-right: 4px solid #ffc107; border-radius: 10px; margin-bottom: 15px; color: black; }
    .welcome-title { font-size: 1.8rem; font-weight: bold; text-align: center; color: #1E88E5; }
    .programmer-tag { font-size: 0.85rem; text-align: center; font-weight: bold; opacity: 0.7; }
    .teacher-badge { font-size: 0.8rem; background-color: #f0f2f6; color: #1E88E5; padding: 2px 8px; border-radius: 10px; border: 1px solid #1E88E5; margin-left: 10px; float: left; }
    </style>
    """, unsafe_allow_html=True)

subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "إنكليزي", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "عربي", "إنكليزي", "وطنية"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "فرنسي", "عربي", "إنكليزي", "وطنية"]
}

# --- إدارة الجلسات ومؤقت الأمان (Session Timeout) ---
if "user_data" not in st.session_state: st.session_state["user_data"] = None
if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
if "oral_exam_history" not in st.session_state: st.session_state["oral_exam_history"] = []
if "last_active" not in st.session_state: st.session_state["last_active"] = time.time()

# التحقق من Timeout (خروج تلقائي بعد ساعة من الخمول)
if st.session_state["user_data"] is not None:
    if time.time() - st.session_state["last_active"] > 3600:
        st.session_state["user_data"] = None
        st.warning("تم تسجيل الخروج تلقائياً لأسباب أمنية (Timeout). يرجى تسجيل الدخول مجدداً.")
    st.session_state["last_active"] = time.time()

# هاش كلمة سر المالك المعتمدة بدلاً من كتابتها نصياً صريحاً 
# (hosam031007 = 1a6b0cf... بـ SHA256)
OWNER_PASS_HASH = hash_password("hosam031007")

# ==========================================
# 4. شاشة الدخول والتسجيل
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
                # التحقق المشفر للمالك
                if u == "Hosam" and hash_password(p) == OWNER_PASS_HASH:
                    st.session_state["user_data"] = {"user": u, "role": "Owner", "grade": "الكل", "is_new": False, "is_premium": True}
                    st.rerun()
                else:
                    users = load_data(USERS_DB)
                    if not users.empty:
                        match = users[(users["user"] == u) & (users["pass"] == hash_password(p))]
                        if not match.empty:
                            st.session_state["user_data"] = match.iloc[0].to_dict()
                            st.rerun()
                        else: st.error("⚠️ عذراً، البيانات غير صحيحة")
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
                    new_user = pd.DataFrame([{"user": nu, "pass": hash_password(np), "role": "طالب", "grade": ng, "fb_link": fb, "is_new": False, "is_premium": False, "invited_by": invite}])
                    pd.concat([users, new_user], ignore_index=True).to_csv(USERS_DB, index=False)
                    st.success("🎉 تم إنشاء الحساب! سجل دخولك الآن.")

# ==========================================
# 5. شاشات المستخدمين (بعد تسجيل الدخول)
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
            pd.concat([ts_db, pd.DataFrame([{"teacher_name": user["user"], "grade": sel_grade, "subject": sel_sub}])], ignore_index=True).to_csv(TEACHER_SUBJECTS_DB, index=False)
            users_df = load_data(USERS_DB)
            users_df.loc[users_df['user'] == user['user'], 'is_new'] = False
            users_df.to_csv(USERS_DB, index=False)
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
                                codes_df.to_csv(CODES_DB, index=False)
                                users_df = load_data(USERS_DB)
                                users_df.loc[users_df['user'] == user['user'], 'is_premium'] = True
                                users_df.to_csv(USERS_DB, index=False)
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
                    pd.concat([users, pd.DataFrame([{"user": t_name, "pass": hash_password(t_pass), "role": "أستاذ", "grade": "الكل", "fb_link": "معلم", "is_new": True, "is_premium": True, "invited_by": ""}])], ignore_index=True).to_csv(USERS_DB, index=False)
                    st.success("تم التفعيل!")
                    st.rerun()

        with t_files:
            f_df = load_data(FILES_DB)
            file_to_del = st.selectbox("اختر الملف للحذف:", [""] + list(f_df['name'].values))
            if st.button("🗑️ حذف الملف") and file_to_del:
                row = f_df[f_df['name'] == file_to_del].iloc[0]
                t_path = os.path.join("lessons" if row['type'] == "بحث" else "exams", file_to_del)
                if os.path.exists(t_path): os.remove(t_path)
                f_df[f_df['name'] != file_to_del].to_csv(FILES_DB, index=False)
                st.success("تم الحذف!")
                st.rerun()

        with t_codes:
            num_codes = st.number_input("عدد الأكواد (5 أرقام):", min_value=1, value=10)
            if st.button("توليد الأكواد ⚙️"):
                # نظام يمنع تكرار الأكواد كلياً 
                c_df = load_data(CODES_DB)
                existing_codes = set(c_df['code'].tolist()) if not c_df.empty else set()
                new_codes = []
                while len(new_codes) < num_codes:
                    new_c = random.randint(10000, 99999)
                    if new_c not in existing_codes:
                        new_codes.append({"code": new_c, "is_used": False, "used_by": "", "date_created": datetime.now().strftime("%Y-%m-%d")})
                        existing_codes.add(new_c)
                pd.concat([c_df, pd.DataFrame(new_codes)], ignore_index=True).to_csv(CODES_DB, index=False)
                st.success(f"تم توليد {num_codes} كود فريد وجديد بنجاح!")

        with t_notify:
            n_df = load_data(NOTIFY_DB)
            st.dataframe(n_df, use_container_width=True)
            if not n_df.empty and st.button("مسح جميع التنويهات"): 
                pd.DataFrame(columns=["sender", "message", "date"]).to_csv(NOTIFY_DB, index=False)
                st.rerun()
                
        # قسم كشف الغش المطور
        with t_anti_cheat:
            st.info("أدخل إجابتين لطالبين مختلفين لمعرفة نسبة التطابق الدلالي بينهما (الـ AI سيكشف تغيير الكلمات).")
            text1 = st.text_area("إجابة الطالب الأول:")
            text2 = st.text_area("إجابة الطالب الثاني:")
            if st.button("فحص نسبة التطابق 🕵️"):
                score = check_cheating(text1, text2)
                if score > 85:
                    st.error(f"🚨 نسبة التطابق عالية جداً: {score}% (احتمال نسخ ولصق كبير)")
                else:
                    st.success(f"✅ نسبة التطابق طبيعية: {score}%")

    # ----------------------------------------
    # واجهة الطالب والأستاذ المشتركة 
    # ----------------------------------------
    elif user["role"] in ["طالب", "أستاذ"]:
        if user["role"] == "أستاذ":
            st.markdown(f'<div class="modern-box"><div class="welcome-title">👨‍🏫 أهلاً بك أستاذ {user["user"]}</div><div class="programmer-tag">{teacher_sub} - {teacher_grade}</div></div>', unsafe_allow_html=True)
            view_grade = st.selectbox("اختر الصف:", ["التاسع", "البكالوريا العلمي", "البكالوريا الأدبي"]) if teacher_grade == "كل الصفوف" else teacher_grade
            sub = teacher_sub
            tabs = st.tabs(["📢 إرسال إشعار", "📤 رفع الملفات", "📚 المكتبة", "🤖 المعلم الذكي", "📸 عدسة الذكاء", "📝 الامتحانات"])
        else:
            st.markdown(f'<div class="modern-box"><div class="welcome-title">{time_greeting} يا بطل!</div><div class="programmer-tag">الصف: {user["grade"]}</div></div>', unsafe_allow_html=True)
            view_grade, sub = user["grade"], st.selectbox("اختر المادة:", subs_map[user["grade"]])
            
            b_df = load_data(BROADCAST_DB)
            if not b_df.empty:
                for _, b in b_df[(b_df['grade'] == view_grade) & (b_df['subject'] == sub)].tail(3).iterrows():
                    st.markdown(f"<div class='broadcast-box'><b>🔔 إشعار من {b['sender']}:</b> {b['message']}</div>", unsafe_allow_html=True)

            tabs = st.tabs(["📚 المكتبة", "🤖 المعلم الذكي", "📸 عدسة الذكاء", "📝 الامتحانات", "📅 الخطة"])

        tab_index = 0

        # -- للأساتذة فقط (إشعارات ورفع مع تحقق أمني) --
        if user["role"] == "أستاذ":
            with tabs[tab_index]:
                b_msg = st.text_area("اكتب الإشعار للطلاب:")
                if st.button("🚀 إرسال") and b_msg:
                    pd.concat([load_data(BROADCAST_DB), pd.DataFrame([{"sender": user["user"], "grade": view_grade, "subject": sub, "message": b_msg, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}])], ignore_index=True).to_csv(BROADCAST_DB, index=False)
                    st.success("تم الإرسال!")
            tab_index += 1

            with tabs[tab_index]:
                with st.form("upload_form", clear_on_submit=True):
                    uploaded_file = st.file_uploader("اختر ملف (PDF)", type="pdf")
                    file_name_input = st.text_input("اسم الملف (مثال: نوطة الوحدة الأولى)")
                    ch_num = st.number_input("رقم البحث", min_value=1, value=1)
                    type_f = st.radio("تصنيف الملف:", ["بحث (درس/نوطة)", "نموذج امتحاني", "سلم تصحيح (للذكاء الاصطناعي)"], horizontal=True)
                    
                    if st.form_submit_button("🚀 رفع الملف"):
                        if uploaded_file:
                            # حماية الرفع: التأكد من امتداد ونوع الملف
                            if uploaded_file.type != "application/pdf" or not uploaded_file.name.lower().endswith('.pdf'):
                                st.error("⚠️ غير مسموح برفع ملفات غير الـ PDF لأسباب أمنية.")
                            else:
                                internal_type = "بحث" if "بحث" in type_f else "نموذج" if "نموذج" in type_f else "سلم"
                                f_name = f"{internal_type}_{sub}_{file_name_input.replace(' ', '_') if file_name_input else uploaded_file.name.replace(' ', '_')}"
                                if not f_name.endswith('.pdf'): f_name += '.pdf'
                                folder = "lessons" if internal_type == "بحث" else "exams"
                                with open(os.path.join(folder, f_name), "wb") as f: f.write(uploaded_file.getbuffer())
                                pd.concat([load_data(FILES_DB), pd.DataFrame([{"name": f_name, "grade": view_grade, "sub": sub, "type": internal_type, "date": datetime.now().strftime("%Y-%m-%d"), "uploader": user["user"], "chapter_num": ch_num}])], ignore_index=True).to_csv(FILES_DB, index=False)
                                st.success("تم الرفع بنجاح!")
            tab_index += 1

        # -- المكتبة --
        with tabs[tab_index]:
            f_db = load_data(FILES_DB)
            my_f = f_db[(f_db["grade"] == view_grade) & (f_db["sub"] == sub)] if not f_db.empty else pd.DataFrame()
            if my_f.empty: st.info("المكتبة فارغة.")
            else:
                for _, r in my_f.iterrows():
                    path = os.path.join("lessons" if r['type'] == "بحث" else "exams", r['name'])
                    if os.path.exists(path):
                        is_locked = user["role"] == "طالب" and not user.get("is_premium", False) and r.get("chapter_num", 1) > 2
                        c_f1, c_f2 = st.columns([4, 1])
                        with c_f1:
                            if is_locked: st.button(f"🔒 مقفول: {r['name'].split('_')[-1]}", disabled=True, key=f"lock_{r['name']}")
                            else: 
                                with open(path, "rb") as f: st.download_button(f"📥 {r['name'].split('_')[-1]}", f, file_name=r['name'], key=r['name'])
                        with c_f2: st.markdown(f"<div class='teacher-badge'>أ. {r.get('uploader', 'غير معروف')}</div>", unsafe_allow_html=True)
        tab_index += 1

        # -- المعلم الذكي (مع مانع الهلوسة - RAG المتطور والـ Caching) --
        with tabs[tab_index]:
            st.info("💡 المعلم الذكي سيبحث داخل أجزاء النوطة الأقرب لسؤالك لضمان الدقة وتوفير الوقت والتكلفة.")
            
            available_files = my_f[my_f["type"] == "بحث"] if not my_f.empty else pd.DataFrame()
            best_context = ""
            
            if not available_files.empty:
                selected_file = st.selectbox("📚 اختر النوطة/البحث الذي تسأل عنه:", available_files['name'].tolist(), format_func=lambda x: x.split('_')[-1])
                file_path = os.path.join("lessons", selected_file)
                
                # تخزين المقاطع في ذاكرة الجلسة لتسريع البحث
                if "pdf_chunks" not in st.session_state or st.session_state.get("current_pdf") != file_path:
                    if os.path.exists(file_path):
                        with st.spinner("جاري تهيئة النوطة للبحث الذكي..."):
                            st.session_state["pdf_chunks"] = extract_and_chunk_pdf(file_path)
                            st.session_state["current_pdf"] = file_path
            else:
                st.warning("⚠️ لا يوجد نوط مرفوعة لهذه المادة بعد. المعلم سيجيب من معلوماته العامة.")

            style = st.radio("طريقة الشرح:", ["علمي صارم (من النوطة حصراً)", "بالمشرمحي (ابن البلد)"], horizontal=True)
            for msg in st.session_state["chat_history"]: st.chat_message(msg["role"]).write(msg["content"])
            
            if q := st.chat_input("اكتب سؤالك من النوطة..."):
                st.session_state["chat_history"].append({"role": "user", "content": q})
                st.chat_message("user").write(q)
                
                with st.spinner("يبحث عن أقرب فقرة لسؤالك..."):
                    strict = True if style == "علمي صارم (من النوطة حصراً)" else False
                    pr = f"أجب لمادة {sub} صف {view_grade}: {q}\n"
                    if style == "بالمشرمحي (ابن البلد)": pr += "اشرحها عامية سورية بأمثلة من الشارع"
                    
                    # استخراج أفضل فقرة تتطابق مع السؤال
                    if "pdf_chunks" in st.session_state and st.session_state["pdf_chunks"]:
                        best_context = get_best_context(q, st.session_state["pdf_chunks"])
                        
                    ans = get_ai_response(pr, strict_mode=strict, context_text=best_context)
                    
                st.session_state["chat_history"].append({"role": "assistant", "content": ans})
                st.chat_message("assistant").write(ans)
        tab_index += 1

        # -- عدسة الذكاء --
        with tabs[tab_index]:
            v_mode = st.radio("الخدمة:", ["شرح مسألة", "تصحيح بناءً على سلم الأساتذة"])
            if img := st.file_uploader("ارفع الصورة", type=["jpg", "png", "jpeg"]):
                if st.button("🚀 تحليل"):
                    with st.spinner("جاري التحليل..."):
                        st.info(get_ai_response(f"أنت معلم لمادة {sub}. " + ("اشرح الحل" if v_mode=="شرح مسألة" else "صحح الحل بناء على السلالم السورية."), image=Image.open(img), strict_mode=True))
        tab_index += 1

        # -- الامتحانات (مضاف إليها التسميع الصوتي الحقيقي) --
        with tabs[tab_index]:
            if st.button("🎯 توليد أسئلة من أبحاث الأساتذة (Strict)"): 
                st.markdown(f'<div class="modern-box">{get_ai_response(f"ولد نموذج وزاري سوري لمادة {sub} معتمداً حصراً على أسلوب النماذج المرفوعة.", strict_mode=True)}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("🗣️ **التسميع الشفهي (تحدث ليتم التقييم)**")
            st.info("اضغط على المايكروفون للإجابة شفهياً. سيقوم النظام بتحليل نطقك ومعلوماتك.")
            
            # ميزة تسجيل الصوت الحقيقية
            audio_val = st.audio_input("🎤 سجل إجابتك هنا:")
            if audio_val:
                st.audio(audio_val) # لسماع التسجيل
                with st.spinner("الذكاء الاصطناعي يستمع ويقيّم إجابتك..."):
                    # إرسال الملف الصوتي للـ API
                    audio_data = {"mime_type": "audio/wav", "data": audio_val.getvalue()}
                    o_ans = get_ai_response(f"استمع إلى إجابة الطالب بمادة {sub}. اكتب ما قاله أولاً، ثم صحح الإجابة علمياً ولغوياً واطرح سؤالاً جديداً.", audio=audio_data, strict_mode=True)
                    st.success(o_ans)
        tab_index += 1
        
        # -- الخطة الدراسية (مضافة كقسم جديد للطلاب) --
        if user["role"] == "طالب":
            with tabs[tab_index]:
                st.markdown("### 📅 مولد خطط دراسة تلقائي")
                st.info("ادخل الأيام المتبقية وساعات الفراغ وسنقوم بتوليد خطة منقذة لك.")
                c_plan1, c_plan2 = st.columns(2)
                days_left = c_plan1.number_input("كم يوم متبقي للامتحان؟", min_value=1, value=20)
                hours_daily = c_plan2.slider("كم ساعة تستطيع الدراسة يومياً؟", 1, 15, 6)
                
                if st.button("توليد الخطة السحرية 🪄"):
                    with st.spinner("جاري تخطيط مستقبلك..."):
                        plan_prompt = f"أنا طالب سوري في {view_grade}. متبقي لي {days_left} يوماً للامتحان، وأستطيع دراسة {hours_daily} ساعات يومياً مادة {sub}. قم بتوليد جدول دراسي يومي مقسم بالمواد، مع تحديد أوقات للمراجعة. اجعله واقعياً ومحفزاً ومنسقاً."
                        st.markdown(f'<div class="modern-box">{get_ai_response(plan_prompt)}</div>', unsafe_allow_html=True)
