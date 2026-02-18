import streamlit as st
import pandas as pd
from PIL import Image
import os
import google.generativeai as genai
from datetime import datetime
from gtts import gTTS
import io

# --- 1. إعدادات الأمان والذكاء الاصطناعي (باستخدام مفتاحك الجديد) ---
# الأفضل وضع المفتاح في Secrets كما شرحت لك، ولكن سأضعه هنا كاحتياط أيضاً
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    API_KEY = "AIzaSyCn33VD-Dc241aVPEkh7HuSQRw0K1fHGB4"

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
        return f"⚠️ عذراً، الذكاء الاصطناعي يواجه ضغطاً. (Error: {str(e)})"

# دالة المعلم الناطق
def speak_text(text):
    try:
        # ننطق أول 250 حرف فقط لضمان السرعة وعدم التعليق
        tts = gTTS(text=text[:250], lang='ar')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp
    except:
        return None

# --- 2. نظام المجلدات وقواعد البيانات ---
for folder in ['lessons', 'exams', 'keys', 'db']:
    os.makedirs(folder, exist_ok=True)

USERS_DB = "db/users.csv"
FILES_DB = "db/files.csv"
GRADES_DB = "db/grades.csv"

def load_data(path, columns):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)

# --- 3. إدارة الجلسة (منع تسجيل الخروج) ---
if "user_data" not in st.session_state:
    st.session_state["user_data"] = None

# --- 4. الوقت والثيم الذكي ---
hour = datetime.now().hour
if 5 <= hour < 18:
    greeting, bg, txt, card = "☀️ صباح الخير", "#F0F2F6", "#000000", "#FFFFFF"
else:
    greeting, bg, txt, card = "🌙 ليلة سعيدة", "#0E1117", "#FFFFFF", "#262730"

st.set_page_config(page_title="منصة حسام الذكية", layout="wide")

# تصميم الواجهة (CSS)
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
        padding: 15px; border-radius: 8px; color: black; margin-top: 10px;
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
    st.markdown(f'<div class="greeting-box"><h1>{greeting}</h1><p>أهلاً بك في منصة حسام التعليمية</p></div>', unsafe_allow_html=True)
    t_log, t_sign = st.tabs(["🔐 تسجيل الدخول", "📝 إنشاء حساب"])
    
    with t_log:
        u = st.text_input("اسم المستخدم", key="login_u")
        p = st.text_input("كلمة المرور", type="password", key="login_p")
        if st.button("دخول المنصة"):
            if u == "Hosam" and p == "Anahosam031007":
                st.session_state["user_data"] = {"user": u, "role": "Owner", "grade": "الكل"}
                st.rerun()
            else:
                users = load_data(USERS_DB, ["user", "pass", "role", "grade"])
                match = users[(users["user"] == u) & (users["pass"] == p)]
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
            users = load_data(USERS_DB, ["user", "pass", "role", "grade"])
            if nu in users['user'].values: st.error("الاسم موجود مسبقاً")
            else:
                pd.concat([users, pd.DataFrame([{"user": nu, "pass": np, "role": nr, "grade": ng}])]).to_csv(USERS_DB, index=False)
                st.success("تم بنجاح! سجل دخولك الآن")

else:
    user = st.session_state["user_data"]
    st.sidebar.markdown(f"### 👋 أهلاً {user['user']}\n**{greeting}**")
    if st.sidebar.button("🔴 تسجيل الخروج"):
        st.session_state["user_data"] = None
        st.rerun()

    # --- واجهة المالك (حسام) ---
    if user["role"] == "Owner":
        st.header("👑 لوحة التحكم العليا")
        t_users, t_files = st.tabs(["👥 الأعضاء", "📁 الملفات"])
        with t_users: st.dataframe(load_data(USERS_DB, ["user", "pass", "role", "grade"]), use_container_width=True)
        with t_files: st.dataframe(load_data(FILES_DB, ["name", "grade", "sub", "type", "date"]), use_container_width=True)

    # --- واجهة الأستاذ ---
    elif user["role"] == "أستاذ":
        st.header("👨‍🏫 مركز رفع الدروس")
        col1, col2 = st.columns(2)
        with col1: tg = st.selectbox("استهداف الصف:", list(subs_map.keys()))
        with col2: ts = st.selectbox("المادة:", subs_map[tg])
        
        type_f = st.radio("نوع الملف المرفوع:", ["بحث", "نموذج امتحاني"])
        up = st.file_uploader("اختر الملف (PDF)", type=['pdf'], key="teacher_upload")
        
        if up and st.button("🚀 تأكيد الحفظ على السيرفر"):
            with st.spinner("جاري المعالجة..."):
                folder = "lessons" if type_f == "بحث" else "exams"
                f_name = f"{type_f}_{ts}_{up.name.replace(' ','_')}"
                with open(os.path.join(folder, f_name), "wb") as f:
                    f.write(up.getbuffer())
                
                f_db = load_data(FILES_DB, ["name", "grade", "sub", "type", "date"])
                pd.concat([f_db, pd.DataFrame([{"name": f_name, "grade": tg, "sub": ts, "type": type_f, "date": datetime.now().strftime("%Y-%m-%d")}])]).to_csv(FILES_DB, index=False)
                st.success(f"✅ تم رفع {f_name} بنجاح!")

    # --- واجهة الطالب (الخدمات الذكية) ---
    elif user["role"] == "طالب":
        st.markdown(f'<div class="greeting-box"><h3>{greeting} يا بطل</h3><p>صفتك: {user["grade"]}</p></div>', unsafe_allow_html=True)
        sub = st.selectbox("اختر المادة للدراسة:", subs_map[user['grade']])
        
        t_study, t_ai, t_plan = st.tabs(["📚 المكتبة", "🤖 المعلم الذكي", "📅 المنقذ (جدول)"])
        
        with t_study:
            f_db = load_data(FILES_DB, ["name", "grade", "sub", "type", "date"])
            my_f = f_db[(f_db["grade"] == user["grade"]) & (f_db["sub"] == sub)]
            if my_f.empty: st.info("لا توجد ملفات مرفوعة حالياً لهذه المادة.")
            for _, r in my_f.iterrows():
                folder = "lessons" if r['type'] == "بحث" else "exams"
                path = os.path.join(folder, r['name'])
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button(f"📥 تحميل {r['type']}: {r['name'].split('_')[-1]}", f, file_name=r['name'])

        with t_ai:
            st.subheader("💬 المعلم الذكي (صوت وكتابة)")
            q = st.chat_input("اسألني أي سؤال في المنهاج...")
            if q:
                with st.spinner("جاري التفكير..."):
                    ans = get_ai_response(f"أنت معلم سوري خبير، أجب بدقة واختصار عن {sub} لصف {user['grade']}: {q}")
                    st.chat_message("assistant").write(ans)
                    # ميزة الصوت
                    audio_data = speak_text(ans)
                    if audio_data:
                        st.audio(audio_data, format='audio/mp3')
                        st.caption("🔊 اضغط لتسمع شرح المعلم")
            
            st.divider()
            st.subheader("📸 مصحح الأوراق الآلي")
            img = st.file_uploader("ارفع صورة حلك (واضحة)", type=["jpg", "png", "jpeg"])
            if img and st.button("بدء التصحيح الذكي"):
                with st.spinner("جاري تحليل الحل..."):
                    res = get_ai_response(f"صحح ورقة الطالب في {sub} لصف {user['grade']} واعط علامة من 100 مع ملاحظات.", Image.open(img))
                    st.success("اكتمل التحليل!")
                    st.write(res)

        with t_plan:
            st.subheader("🗓️ صانع الجداول الذكي")
            days = st.number_input("كم يوم باقي للفحص؟", 1, 100, 7)
            hours = st.slider("ساعات الدراسة اليومية:", 1, 15, 6)
            if st.button("🚀 صمم لي خطة الإنقاذ"):
                with st.spinner("جاري التصميم..."):
                    plan_prompt = f"صمم جدول دراسي مكثف لمادة {sub} لصف {user['grade']} لمدة {days} أيام، بمعدل {hours} ساعات يومياً. وزع المنهاج بشكل منطقي."
                    plan_res = get_ai_response(plan_prompt)
                    st.markdown(f'<div class="plan-box">{plan_res}</div>', unsafe_allow_html=True)
                    st.balloons()
