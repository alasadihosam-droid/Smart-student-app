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
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("⚠️ لم يتم العثور على مفتاح API في Secrets.")
    st.stop()

genai.configure(api_key=API_KEY)

@st.cache_resource
def load_ai_model():
    return genai.GenerativeModel("gemini-1.5-flash")

def get_ai_response(prompt, image=None):
    try:
        model = load_ai_model()
        if image:
            response = model.generate_content([prompt, image])
        else:
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ عذراً، هناك مشكلة في الاتصال. (Error: {str(e)})"

# دالة التشفير لحماية كلمات المرور
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# دالة المعلم الناطق
def speak_text(text):
    try:
        tts = gTTS(text=text[:250], lang='ar')
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
    return pd.read_csv(path)

# --- 3. إدارة الجلسة ---
if "user_data" not in st.session_state:
    st.session_state["user_data"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- 4. الوقت والثيم الذكي ---
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
                else: st.error("عذراً، البيانات غير صحيحة")
    
    with t_sign:
        nu = st.text_input("الاسم الكامل")
        np = st.text_input("كلمة السر", type="password")
        nr = st.selectbox("أنا:", ["طالب", "أستاذ"])
        ng = st.selectbox("الصف:", list(subs_map.keys())) if nr == "طالب" else "الكل"
        if st.button("تأكيد إنشاء الحساب"):
            if nu and np:
                users = load_data(USERS_DB)
                if nu in users['user'].values: st.error("الاسم موجود مسبقاً")
                else:
                    new_user = pd.DataFrame([{"user": nu, "pass": hash_password(np), "role": nr, "grade": ng}])
                    pd.concat([users, new_user]).to_csv(USERS_DB, index=False)
                    st.success("تم بنجاح! سجل دخولك الآن")

else:
    user = st.session_state["user_data"]
    st.sidebar.markdown(f"### 👋 أهلاً {user['user']}")
    if st.sidebar.button("🔴 تسجيل الخروج"):
        st.session_state["user_data"] = None
        st.session_state["chat_history"] = []
        st.rerun()

    # --- واجهة المالك (حسام) ---
    if user["role"] == "Owner":
        st.header("👑 لوحة التحكم العليا")
        t_users, t_files, t_all_grades = st.tabs(["👥 الأعضاء", "📁 الملفات", "📊 درجات الطلاب"])
        with t_users:
            u_df = load_data(USERS_DB)
            edited_u = st.data_editor(u_df, num_rows="dynamic")
            if st.button("حفظ تعديلات المستخدمين"): edited_u.to_csv(USERS_DB, index=False)
        with t_files:
            f_df = load_data(FILES_DB)
            edited_f = st.data_editor(f_df, num_rows="dynamic")
            if st.button("حفظ تعديلات الملفات"): edited_f.to_csv(FILES_DB, index=False)
        with t_all_grades:
            st.dataframe(load_data(GRADES_DB), use_container_width=True)

    # --- واجهة الأستاذ ---
    elif user["role"] == "أستاذ":
        st.header("👨‍🏫 مركز رفع الدروس")
        col1, col2 = st.columns(2)
        with col1: tg = st.selectbox("استهداف الصف:", list(subs_map.keys()))
        with col2: ts = st.selectbox("المادة:", subs_map[tg])
        type_f = st.radio("نوع الملف:", ["بحث", "نموذج امتحاني"])
        up = st.file_uploader("اختر الملف (PDF)", type=['pdf'])
        if up and st.button("🚀 رفع الملف"):
            f_name = f"{type_f}_{ts}_{up.name.replace(' ','_')}"
            folder = "lessons" if type_f == "بحث" else "exams"
            with open(os.path.join(folder, f_name), "wb") as f: f.write(up.getbuffer())
            f_db = load_data(FILES_DB)
            pd.concat([f_db, pd.DataFrame([{"name": f_name, "grade": tg, "sub": ts, "type": type_f, "date": datetime.now().strftime("%Y-%m-%d")}])]).to_csv(FILES_DB, index=False)
            st.success("تم الرفع بنجاح!")

    # --- واجهة الطالب ---
    elif user["role"] == "طالب":
        st.markdown(f'<div class="greeting-box"><h3>{greeting} يا بطل</h3><p>صفتك: {user["grade"]}</p></div>', unsafe_allow_html=True)
        sub = st.selectbox("اختر المادة للدراسة:", subs_map[user['grade']])
        
        t_study, t_ai, t_plan, t_progress = st.tabs(["📚 المكتبة", "🤖 المعلم الذكي", "📅 المنقذ", "📊 مستواي"])
        
        with t_study:
            search_q = st.text_input("🔍 ابحث عن درس معين...")
            f_db = load_data(FILES_DB)
            my_f = f_db[(f_db["grade"] == user["grade"]) & (f_db["sub"] == sub)]
            if search_q:
                my_f = my_f[my_f['name'].str.contains(search_q, case=False)]
            
            if my_f.empty: st.info("لا توجد ملفات.")
            for _, r in my_f.iterrows():
                folder = "lessons" if r['type'] == "بحث" else "exams"
                path = os.path.join(folder, r['name'])
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button(f"📥 تحميل {r['name'].split('_')[-1]}", f, file_name=r['name'])

        with t_ai:
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]): st.write(msg["content"])
            
            q = st.chat_input("اسألني أي سؤال...")
            if q:
                st.session_state["chat_history"].append({"role": "user", "content": q})
                with st.chat_message("user"): st.write(q)
                ans = get_ai_response(f"أنت معلم خبير، أجب عن {sub} لصف {user['grade']}: {q}")
                st.session_state["chat_history"].append({"role": "assistant", "content": ans})
                with st.chat_message("assistant"):
                    st.write(ans)
                    audio = speak_text(ans)
                    if audio: st.audio(audio)

            st.divider()
            st.subheader("📸 مصحح الأوراق الآلي")
            img = st.file_uploader("ارفع صورة حلك", type=["jpg", "png", "jpeg"])
            if img and st.button("تصحيح الحل"):
                res = get_ai_response(f"صحح ورقة الطالب في {sub} لصف {user['grade']} واعط علامة من 100.", Image.open(img))
                st.write(res)
                # حفظ الدرجة تلقائياً إذا وجدت في النص
                try:
                    score = [int(s) for s in res.split() if s.isdigit() and int(s) <= 100][0]
                    g_db = load_data(GRADES_DB)
                    new_g = pd.DataFrame([{"user": user['user'], "sub": sub, "score": score, "date": datetime.now().strftime("%Y-%m-%d")}])
                    pd.concat([g_db, new_g]).to_csv(GRADES_DB, index=False)
                    st.toast(f"تم تسجيل درجتك: {score}/100")
                except: pass

        with t_plan:
            d = st.number_input("الأيام المتبقية:", 1, 100, 7)
            h = st.slider("الساعات اليومية:", 1, 15, 6)
            if st.button("توليد خطة"):
                plan = get_ai_response(f"خطة دراسة {sub} لصف {user['grade']} في {d} أيام، {h} ساعات يومياً.")
                st.markdown(f'<div class="plan-box">{plan}</div>', unsafe_allow_html=True)
                st.download_button("📥 تحميل الخطة كملف نصي", plan, file_name="my_plan.txt")

        with t_progress:
            st.subheader(f"📈 تطور مستواك في مادة {sub}")
            g_db = load_data(GRADES_DB)
            my_scores = g_db[(g_db["user"] == user["user"]) & (g_db["sub"] == sub)]
            if not my_scores.empty:
                st.line_chart(my_scores.set_index("date")["score"])
                st.write(f"متوسط درجاتك: {my_scores['score'].mean():.1f}%")
            else: st.info("لا توجد درجات مسجلة بعد. استخدم المصحح الآلي لتقييم حلك!")
