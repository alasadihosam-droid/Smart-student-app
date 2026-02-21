import streamlit as st
import pandas as pd
from PIL import Image
import os
import google.generativeai as genai
from datetime import datetime
from gtts import gTTS
import io
import hashlib
import random # لإضافة الأكواد العشوائية
import re

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

def get_ai_response(prompt, image=None, strict_mode=False):
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        safe_models = [m for m in available_models if "2.5" not in m]
        if not safe_models: return "⚠️ عذراً، جميع الموديلات المتاحة في حسابك غير مجانية."
        
        # تقييد الذكاء الاصطناعي للمنهاج السوري حصراً
        system_instruction = ""
        if strict_mode:
            system_instruction = "تعليمات صارمة: أنت معلم سوري. التزم حصراً بالمعلومات الموجودة في المنهاج السوري، سلالم التصحيح، والنماذج المرفوعة. لا تقم بإضافة أي معلومات خارجية من الإنترنت. إذا كان السؤال خارج المنهاج قل 'هذا السؤال خارج المنهاج المقرر'."
            prompt = system_instruction + "\n\n" + prompt

        for model_name in safe_models:
            try:
                model = genai.GenerativeModel(model_name)
                if image: return model.generate_content([prompt, image]).text
                else: return model.generate_content(prompt).text
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
CODES_DB = "db/codes.csv" # قاعدة بيانات أكواد التفعيل
BROADCAST_DB = "db/broadcasts.csv" # قاعدة التنبيهات من الأساتذة للطلاب

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

# تأمين التوافقية مع قواعد البيانات القديمة
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
    .stButton>button { 
        width: 100%; border-radius: 8px; background: #1E88E5; color: white; 
        font-weight: bold; border: none; padding: 0.5rem; transition: 0.2s;
    }
    .stButton>button:active { transform: scale(0.98); }
    .modern-box { 
        padding: 15px; background-color: rgba(30, 136, 229, 0.05); 
        border-radius: 10px; border-right: 4px solid #1E88E5; margin-bottom: 15px;
    }
    .broadcast-box {
        padding: 15px; background-color: #fff3cd; border-right: 4px solid #ffc107; 
        border-radius: 10px; margin-bottom: 15px; color: black;
    }
    .welcome-title { font-size: 1.8rem; font-weight: bold; text-align: center; color: #1E88E5; }
    .programmer-tag { font-size: 0.85rem; text-align: center; font-weight: bold; opacity: 0.7; }
    .teacher-badge {
        font-size: 0.8rem; background-color: #f0f2f6; color: #1E88E5; padding: 2px 8px; 
        border-radius: 10px; border: 1px solid #1E88E5; margin-left: 10px; float: left;
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

# ==========================================
# نظام تسجيل الدخول التلقائي (حفظ الجلسة)
# ==========================================
if st.session_state["user_data"] is None and "session_token" in st.query_params:
    token = st.query_params["session_token"]
    if token == "Hosam":
        st.session_state["user_data"] = {"user": "Hosam", "role": "Owner", "grade": "الكل", "is_new": False, "is_premium": True}
    else:
        users = load_data(USERS_DB)
        match = users[users["user"] == token]
        if not match.empty:
            st.session_state["user_data"] = match.iloc[0].to_dict()

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
                if u == "Hosam" and p == "hosam031007":
                    st.session_state["user_data"] = {"user": u, "role": "Owner", "grade": "الكل", "is_new": False, "is_premium": True}
                    st.query_params["session_token"] = u 
                    st.rerun()
                else:
                    users = load_data(USERS_DB)
                    if not users.empty:
                        match = users[(users["user"] == u) & (users["pass"] == hash_password(p))]
                        if not match.empty:
                            st.session_state["user_data"] = match.iloc[0].to_dict()
                            st.query_params["session_token"] = u 
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
    
    # --- معالجة الدخول الأول للأستاذ ---
    if user["role"] == "أستاذ" and user.get("is_new", True):
        st.markdown(f'<div class="modern-box"><div class="welcome-title">أهلاً وسهلاً بك يا أستاذنا الفاضل 👨‍🏫</div></div>', unsafe_allow_html=True)
        st.info("لتكتمل إعدادات حسابك، يرجى اختيار الصف والمادة التي تدرسها لترتبط ملفاتك بها مباشرة.")
        
        col_g, col_s = st.columns(2)
        sel_grade = col_g.selectbox("الصف الذي تدرسه:", list(subs_map.keys()) + ["كل الصفوف"])
        
        # تحديد المادة بناءً على الاختيار
        if sel_grade == "كل الصفوف":
            # جمع كل المواد بدون تكرار
            all_subs = list(set([item for sublist in subs_map.values() for item in sublist]))
            sel_sub = col_s.selectbox("مادتك الاختصاصية:", all_subs)
        else:
            sel_sub = col_s.selectbox("مادتك الاختصاصية:", subs_map[sel_grade])
        
        pic = st.file_uploader("ارفع صورتك الشخصية (اختياري)", type=['png', 'jpg', 'jpeg'])
        
        if st.button("حفظ الإعدادات والبدء 🚀"):
            if pic: Image.open(pic).save(f"profiles/{user['user']}.png")
            ts_db = load_data(TEACHER_SUBJECTS_DB)
            new_ts = pd.DataFrame([{"teacher_name": user["user"], "grade": sel_grade, "subject": sel_sub}])
            pd.concat([ts_db, new_ts], ignore_index=True).to_csv(TEACHER_SUBJECTS_DB, index=False)

            users_df = load_data(USERS_DB)
            users_df.loc[users_df['user'] == user['user'], 'is_new'] = False
            users_df.to_csv(USERS_DB, index=False)
            
            st.session_state["user_data"]["is_new"] = False
            st.success("تم إعداد حسابك بنجاح!")
            st.rerun()
        st.stop() 
    
    # --- استرجاع مادة الأستاذ ---
    teacher_grade, teacher_sub = "", ""
    if user["role"] == "أستاذ":
        ts_db = load_data(TEACHER_SUBJECTS_DB)
        t_match = ts_db[ts_db["teacher_name"] == user["user"]]
        if not t_match.empty:
            teacher_grade = t_match.iloc[0]["grade"]
            teacher_sub = t_match.iloc[0]["subject"]

    # --- القائمة الجانبية (Sidebar) ---
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
            if user.get('is_premium', False):
                st.success("حساب مدفوع (Premium) 🌟")
            else:
                st.info("حساب مجاني 🆓")
                st.markdown("**(أول بحثين مجاناً)**")
                with st.form("premium_form"):
                    code_input = st.text_input("أدخل كود التفعيل (5 أرقام):")
                    if st.form_submit_button("تفعيل الاشتراك 🚀"):
                        codes_df = load_data(CODES_DB)
                        if not codes_df.empty:
                            match_code = codes_df[(codes_df['code'] == int(code_input)) & (codes_df['is_used'] == False)]
                            if not match_code.empty:
                                # تفعيل الكود
                                codes_df.loc[codes_df['code'] == int(code_input), 'is_used'] = True
                                codes_df.loc[codes_df['code'] == int(code_input), 'used_by'] = user['user']
                                codes_df.to_csv(CODES_DB, index=False)
                                # ترقية الطالب
                                users_df = load_data(USERS_DB)
                                users_df.loc[users_df['user'] == user['user'], 'is_premium'] = True
                                users_df.to_csv(USERS_DB, index=False)
                                st.session_state["user_data"]["is_premium"] = True
                                st.success("تم تفعيل حسابك بنجاح لسنة كاملة! 🎉")
                                st.rerun()
                            else: st.error("الكود غير صحيح أو مستخدم مسبقاً.")
                        else: st.error("لا توجد أكواد في النظام.")
                
        st.divider()
        st.markdown("### 🤝 دعوة للمنصة")
        st.text_input("شارك الرابط مع أصدقائك:", value="https://sanad.streamlit.app", disabled=True)
            
        st.divider()
        if st.button("🔴 تسجيل الخروج"):
            st.session_state["user_data"] = None
            if "session_token" in st.query_params: del st.query_params["session_token"]
            st.rerun()

    # ----------------------------------------
    # واجهة الإدارة (Owner Dashboard)
    # ----------------------------------------
    if user["role"] == "Owner":
        st.header(f"👑 لوحة تحكم الإدارة الشاملة - {time_greeting}")
        t_users, t_teachers, t_files, t_codes, t_notify, t_settings = st.tabs(["👥 الطلاب", "👨‍🏫 الأساتذة", "📁 الملفات", "💳 الاشتراكات", "📩 رسائل الأساتذة", "⚙️ إعداداتي"])
        
        with t_users:
            u_df = load_data(USERS_DB)
            students = u_df[u_df['role'] == 'طالب']
            st.data_editor(students, num_rows="dynamic", use_container_width=True)

        with t_teachers:
            st.markdown("### ➕ إضافة أستاذ جديد")
            col1, col2 = st.columns(2)
            t_name = col1.text_input("اسم الأستاذ")
            t_pass = col2.text_input("كلمة مرور الأستاذ", type="password")
            if st.button("إنشاء حساب الأستاذ"):
                if t_name and t_pass:
                    users = load_data(USERS_DB)
                    if not users.empty and t_name in users['user'].values: st.error("الاسم موجود.")
                    else:
                        new_t = pd.DataFrame([{"user": t_name, "pass": hash_password(t_pass), "role": "أستاذ", "grade": "الكل", "fb_link": "معلم", "is_new": True, "is_premium": True, "invited_by": ""}])
                        pd.concat([users, new_t], ignore_index=True).to_csv(USERS_DB, index=False)
                        st.success("تم تفعيل حساب الأستاذ بنجاح!")
                        st.rerun()
            st.markdown("---")
            st.markdown("### سجل الأساتذة (Affiliate)")
            teachers_df = u_df[u_df['role'] == 'أستاذ']
            st.dataframe(teachers_df, use_container_width=True)
            
            # إحصائيات دعوات الأساتذة
            st.markdown("### 📊 إحصائيات دعوات الأساتذة للطلاب")
            invite_counts = students['invited_by'].value_form().reset_index()
            invite_counts.columns = ['اسم الأستاذ', 'عدد الطلاب المدعوين']
            invite_counts = invite_counts[invite_counts['اسم الأستاذ'] != ""]
            st.dataframe(invite_counts, use_container_width=True)

        with t_files:
            f_df = load_data(FILES_DB)
            file_to_del = st.selectbox("اختر الملف للحذف نهائياً:", [""] + list(f_df['name'].values))
            if st.button("🗑️ حذف الملف") and file_to_del:
                row = f_df[f_df['name'] == file_to_del].iloc[0]
                target_path = os.path.join("lessons" if row['type'] == "بحث" else "exams", file_to_del)
                if os.path.exists(target_path): os.remove(target_path)
                f_df[f_df['name'] != file_to_del].to_csv(FILES_DB, index=False)
                st.success("تم الحذف!")
                st.rerun()

        with t_codes:
            st.markdown("### 💳 توليد أكواد الاشتراك (5 أرقام)")
            num_codes = st.number_input("عدد الأكواد المطلوبة:", min_value=1, max_value=500, value=10)
            if st.button("توليد الأكواد ⚙️"):
                new_codes = []
                for _ in range(num_codes):
                    new_codes.append({"code": random.randint(10000, 99999), "is_used": False, "used_by": "", "date_created": datetime.now().strftime("%Y-%m-%d")})
                c_db = load_data(CODES_DB)
                pd.concat([c_db, pd.DataFrame(new_codes)], ignore_index=True).to_csv(CODES_DB, index=False)
                st.success(f"تم توليد {num_codes} كود بنجاح!")
            
            st.markdown("### 📋 سجل الأكواد")
            c_df = load_data(CODES_DB)
            if not c_df.empty:
                st.dataframe(c_df, use_container_width=True)

        with t_notify:
            n_df = load_data(NOTIFY_DB)
            if not n_df.empty:
                st.dataframe(n_df, use_container_width=True)
                if st.button("مسح جميع التنويهات"): 
                    pd.DataFrame(columns=["sender", "message", "date"]).to_csv(NOTIFY_DB, index=False)
                    st.rerun()
            else: st.info("لا يوجد رسائل أو تنويهات جديدة من الأساتذة.")
                
        with t_settings:
            pic = st.file_uploader("ارفع صورتك الشخصية (JPG/PNG)", type=['png', 'jpg', 'jpeg'])
            if pic and st.button("💾 حفظ الصورة"):
                Image.open(pic).save(f"profiles/{user['user']}.png")
                st.success("تم التحديث!")
                st.rerun()

    # ----------------------------------------
    # واجهة الطالب والأستاذ المشتركة 
    # ----------------------------------------
    elif user["role"] in ["طالب", "أستاذ"]:
        if user["role"] == "أستاذ":
            st.markdown(f'<div class="modern-box"><div class="welcome-title">👨‍🏫 أهلاً بك أستاذ {user["user"]}</div><div class="programmer-tag">{teacher_sub} - {teacher_grade}</div></div>', unsafe_allow_html=True)
            
            # السماح للأستاذ بالتنقل بين الصفوف إذا كان "كل الصفوف"
            if teacher_grade == "كل الصفوف":
                view_grade = st.selectbox("اختر الصف للعمل عليه الآن:", ["التاسع", "البكالوريا العلمي", "البكالوريا الأدبي"])
            else:
                view_grade = teacher_grade
            sub = teacher_sub
            
            tabs = st.tabs(["📢 إرسال إشعار", "📤 رفع الملفات", "📚 المكتبة", "🤖 المعلم الذكي", "📸 عدسة الذكاء", "📝 الامتحانات", "💬 مراسلة الإدارة"])
        else:
            st.markdown(f'<div class="modern-box"><div class="welcome-title">{time_greeting} يا بطل!</div><div class="programmer-tag">الصف: {user["grade"]}</div></div>', unsafe_allow_html=True)
            view_grade = user["grade"]
            sub = st.selectbox("اختر المادة التي ترغب بدراستها:", subs_map[view_grade])
            
            # عرض الإشعارات للطلاب
            b_df = load_data(BROADCAST_DB)
            if not b_df.empty:
                my_broadcasts = b_df[(b_df['grade'] == view_grade) & (b_df['subject'] == sub)]
                for _, b in my_broadcasts.tail(3).iterrows():
                    st.markdown(f"<div class='broadcast-box'><b>🔔 تنبيه من الأستاذ {b['sender']}:</b> {b['message']} <br><small>{b['date']}</small></div>", unsafe_allow_html=True)

            tabs = st.tabs(["📚 المكتبة", "🤖 المعلم الذكي", "📸 عدسة الذكاء", "📝 الامتحانات", "📅 المنقذ", "📊 مستواي"])

        tab_index = 0

        # -- تاب التنبيهات والرفع (للأستاذ فقط) --
        if user["role"] == "أستاذ":
            with tabs[tab_index]:
                st.info("أرسل تنبيهاً أو رسالة سريعة لطلابك (ستظهر لهم أعلى الشاشة).")
                b_msg = st.text_area("اكتب الإشعار هنا:")
                if st.button("🚀 إرسال الإشعار للطلاب"):
                    b_db = load_data(BROADCAST_DB)
                    new_b = pd.DataFrame([{"sender": user["user"], "grade": view_grade, "subject": sub, "message": b_msg, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}])
                    pd.concat([b_db, new_b], ignore_index=True).to_csv(BROADCAST_DB, index=False)
                    st.success("تم نشر الإشعار بنجاح!")
            tab_index += 1

            with tabs[tab_index]:
                st.info("ارفع (نوط، دروس، سلالم تصحيح، ونماذج). سيقوم الذكاء الاصطناعي بتوليد الأسئلة منها حصراً.")
                with st.form("upload_form", clear_on_submit=True):
                    uploaded_file = st.file_uploader("اختر ملف (PDF)", type="pdf")
                    file_name_input = st.text_input("اسم الملف (مثال: نوطة الوحدة الأولى)")
                    ch_num = st.number_input("رقم البحث (البحث 1 و 2 مجاني للطلاب)", min_value=1, value=1)
                    type_f = st.radio("تصنيف الملف:", ["بحث (درس/نوطة)", "نموذج امتحاني", "سلم تصحيح (للذكاء الاصطناعي)"], horizontal=True)
                    
                    if st.form_submit_button("🚀 رفع الملف للمنصة"):
                        if uploaded_file:
                            with st.spinner("جاري الرفع..."):
                                internal_type = "بحث" if "بحث" in type_f else "نموذج" if "نموذج" in type_f else "سلم"
                                final_name = file_name_input.replace(' ', '_') if file_name_input else uploaded_file.name.replace(' ', '_')
                                if not final_name.endswith('.pdf'): final_name += '.pdf'
                                f_name = f"{internal_type}_{sub}_{final_name}"
                                folder = "lessons" if internal_type == "بحث" else "exams"
                                with open(os.path.join(folder, f_name), "wb") as f: f.write(uploaded_file.getbuffer())
                                pd.concat([load_data(FILES_DB), pd.DataFrame([{"name": f_name, "grade": view_grade, "sub": sub, "type": internal_type, "date": datetime.now().strftime("%Y-%m-%d"), "uploader": user["user"], "chapter_num": ch_num}])], ignore_index=True).to_csv(FILES_DB, index=False)
                            st.success("تم الرفع بنجاح!")
                        else: st.warning("يرجى اختيار ملف.")
            tab_index += 1

        # -- المكتبة (الطلاب والأساتذة) --
        with tabs[tab_index]:
            f_db = load_data(FILES_DB)
            if not f_db.empty:
                my_f = f_db[(f_db["grade"] == view_grade) & (f_db["sub"] == sub)]
                if my_f.empty: st.info("لا توجد ملفات.")
                else:
                    for _, r in my_f.iterrows():
                        path = os.path.join("lessons" if r['type'] == "بحث" else "exams", r['name'])
                        if os.path.exists(path):
                            uploader_name = r.get("uploader", "غير معروف")
                            ch_n = r.get("chapter_num", 1)
                            
                            # نظام الـ Freemium (البحث 1 و 2 مجاني، الباقي مقفول إلا للمدفعوع)
                            is_locked = False
                            if user["role"] == "طالب" and not user.get("is_premium", False) and ch_n > 2:
                                is_locked = True
                                
                            col_f1, col_f2 = st.columns([4, 1])
                            with col_f1:
                                if is_locked:
                                    st.button(f"🔒 مقفول: {r['name'].split('_')[-1]} (اشترك لفتح البحث)", disabled=True, key=f"lock_{r['name']}")
                                else:
                                    with open(path, "rb") as f: 
                                        st.download_button(f"📥 {r['name'].split('_')[-1]} (بحث {ch_n})", f, file_name=r['name'], key=r['name'])
                            with col_f2:
                                t_profile_path = f"profiles/{uploader_name}.png"
                                if os.path.exists(t_profile_path): st.image(t_profile_path, width=30)
                                st.markdown(f"<div class='teacher-badge'>أ. {uploader_name}</div>", unsafe_allow_html=True)
            else: st.info("المكتبة فارغة.")
        tab_index += 1

        # -- المعلم الذكي (Strict Mode) --
        with tabs[tab_index]:
            style = st.radio("طريقة الشرح:", ["علمي صارم (من المنهاج حصراً)", "بالمشرمحي (ابن البلد)"], horizontal=True)
            for msg in st.session_state["chat_history"]: st.chat_message(msg["role"]).write(msg["content"])
            if q := st.chat_input("اكتب سؤالك..."):
                st.session_state["chat_history"].append({"role": "user", "content": q})
                st.chat_message("user").write(q)
                with st.spinner("يراجع المنهاج وسلالم التصحيح المرفوعة..."):
                    strict = True if style == "علمي صارم (من المنهاج حصراً)" else False
                    pr = f"أجب لمادة {sub} صف {view_grade}: {q}\n"
                    if style == "بالمشرمحي (ابن البلد)": pr += "اشرحها عامية سورية بأمثلة من الشارع"
                    ans = get_ai_response(pr, strict_mode=strict)
                st.session_state["chat_history"].append({"role": "assistant", "content": ans})
                st.chat_message("assistant").write(ans)
        tab_index += 1

        # -- عدسة الذكاء --
        with tabs[tab_index]:
            v_mode = st.radio("الخدمة:", ["شرح مسألة", "تصحيح بناءً على سلم الأساتذة"])
            if img := st.file_uploader("ارفع الصورة", type=["jpg", "png", "jpeg"]):
                if st.button("🚀 تحليل"):
                    with st.spinner("جاري التحليل..."):
                        res = get_ai_response(f"أنت معلم لمادة {sub} لصف {view_grade}. " + ("اشرح الدرس وطريقة الحل" if v_mode=="شرح مسألة" else "صحح الحل بالصورة بناء على سلالم التصحيح السورية المرفوعة واعط درجة من 100."), Image.open(img), strict_mode=True)
                        st.info(res)
        tab_index += 1

        # -- الامتحانات --
        with tabs[tab_index]:
            if st.button("🎯 توليد أسئلة من أبحاث الأساتذة (Strict)"): 
                st.markdown(f'<div class="modern-box">{get_ai_response(f"ولد نموذج وزاري سوري لمادة {sub} صف {view_grade} معتمداً حصراً على أسلوب النماذج المرفوعة من الأساتذة.", strict_mode=True)}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("🗣️ **التسميع الشفهي**")
            for m in st.session_state["oral_exam_history"]: st.chat_message(m["role"]).write(m["content"])
            if oral_q := st.chat_input("إجابتك للتسميع..."):
                st.session_state["oral_exam_history"].append({"role": "user", "content": oral_q})
                st.chat_message("user").write(oral_q)
                with st.spinner("يقيّم..."):
                    o_ans = get_ai_response(f"صحح إجابة الطالب: '{oral_q}' بمادة {sub}، واطرح سؤال شفهي جديد.", strict_mode=True)
                st.session_state["oral_exam_history"].append({"role": "assistant", "content": o_ans})
                st.chat_message("assistant").write(o_ans)
        tab_index += 1

        # -- التاب الأخير (المنقذ للطالب / رسائل الإدارة للأستاذ) --
        with tabs[tab_index]:
            if user["role"] == "طالب":
                ca, cb = st.columns(2)
                if st.button("توليد الخطة"): st.markdown(f'<div class="modern-box">{get_ai_response(f"خطة دراسة {sub} لصف {view_grade} في {ca.number_input("أيام؟",1,value=7)} أيام بـ {cb.slider("ساعات؟",1,15,5)} ساعات يوميا.")}</div>', unsafe_allow_html=True)
            else:
                st.markdown("### 💬 إرسال تنويه لمالك المنصة")
                msg = st.text_area("اكتب رسالتك، استفسارك، أو تقريرك هنا:")
                if st.button("إرسال للإدارة 📩") and msg:
                    n_db = load_data(NOTIFY_DB)
                    new_n = pd.DataFrame([{"sender": user["user"], "message": msg, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}])
                    pd.concat([n_db, new_n], ignore_index=True).to_csv(NOTIFY_DB, index=False)
                    st.success("تم إرسال رسالتك للأستاذ حسام بنجاح!")
