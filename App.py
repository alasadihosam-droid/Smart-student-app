import streamlit as st
import pandas as pd
from PIL import Image
import os
import google.generativeai as genai
from datetime import datetime
from gtts import gTTS # للمعلم الناطق
import io

# --- 1. إعدادات الذكاء الاصطناعي ---
API_KEY = "AIzaSyBkrJ1cCsCQtoYGK361daqbaxdlyQWFPKw"
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
        return f"⚠️ مشكلة في الذكاء الاصطناعي: {str(e)}"

# دالة تحويل النص لصوت (المعلم الناطق)
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='ar')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp
    except:
        return None

# --- 2. نظام المجلدات ---
for folder in ['lessons', 'exams', 'keys', 'db']:
    os.makedirs(folder, exist_ok=True)

USERS_DB, FILES_DB, GRADES_DB = "db/users.csv", "db/files.csv", "db/grades.csv"

def load_data(path, columns):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)

# --- 3. إدارة الجلسة ---
if "user_data" not in st.session_state:
    params = st.query_params
    if "user" in params:
        users = load_data(USERS_DB, ["user", "pass", "role", "grade"])
        match = users[users["user"] == params["user"]]
        if not match.empty:
            st.session_state["user_data"] = match.iloc[0].to_dict()
        else: st.session_state["user_data"] = None
    else: st.session_state["user_data"] = None

# --- 4. الثيم والوقت ---
hour = datetime.now().hour
greeting, bg, txt, card = ("☀️ صباح الخير", "#F0F2F6", "#000000", "#FFFFFF") if 5 <= hour < 18 else ("🌙 ليلة سعيدة", "#0E1117", "#FFFFFF", "#262730")

st.set_page_config(page_title="منصة حسام الذكية", layout="wide")

st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg}; color: {txt}; }}
    .stButton>button {{ width: 100%; border-radius: 12px; height: 3.5em; background: linear-gradient(45deg, #D32F2F, #B71C1C); color: white; font-weight: bold; border: none; }}
    .greeting-box {{ padding: 20px; background-color: {card}; border-radius: 15px; border: 1px solid #D32F2F; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
    .plan-box {{ background-color: #fdf2f2; border-right: 5px solid #D32F2F; padding: 15px; border-radius: 8px; color: black; }}
    </style>
    """, unsafe_allow_html=True)

subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "إنكليزي", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "عربي"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "فرنسي", "عربي"]
}

# --- 5. منطق الدخول ---
if st.session_state["user_data"] is None:
    st.markdown(f'<div class="greeting-box"><h1>{greeting}</h1><p>أهلاً بك في منصة الطالب السوري</p></div>', unsafe_allow_html=True)
    t_log, t_sign = st.tabs(["🔐 دخول", "📝 جديد"])
    with t_log:
        u, p = st.text_input("المستخدم"), st.text_input("كلمة المرور", type="password")
        if st.button("دخول"):
            users = load_data(USERS_DB, ["user", "pass", "role", "grade"])
            if u == "Hosam" and p == "Anahosam031007":
                st.session_state["user_data"] = {"user": u, "role": "Owner", "grade": "الكل"}
                st.query_params["user"] = u; st.rerun()
            else:
                match = users[(users["user"] == u) & (users["pass"] == p)]
                if not match.empty:
                    st.session_state["user_data"] = match.iloc[0].to_dict()
                    st.query_params["user"] = u; st.rerun()
                else: st.error("بيانات خاطئة")
    with t_sign:
        nu, np = st.text_input("الاسم"), st.text_input("الباسورد", type="password")
        nr = st.selectbox("الرتبة", ["طالب", "أستاذ"])
        ng = st.selectbox("الصف", list(subs_map.keys())) if nr == "طالب" else "الكل"
        if st.button("تأكيد"):
            users = load_data(USERS_DB, ["user", "pass", "role", "grade"])
            pd.concat([users, pd.DataFrame([{"user": nu, "pass": np, "role": nr, "grade": ng}])]).to_csv(USERS_DB, index=False)
            st.success("تم بنجاح!")

# --- 6. الواجهة الرئيسية ---
else:
    user = st.session_state["user_data"]
    st.sidebar.markdown(f"### 👋 {user['user']}\n{greeting}")
    if st.sidebar.button("🔴 خروج"):
        st.session_state["user_data"] = None; st.query_params.clear(); st.rerun()

    if user["role"] == "Owner":
        st.header("👑 لوحة الملك حسام")
        st.dataframe(load_data(USERS_DB, ["user", "pass", "role", "grade"]), use_container_width=True)

    elif user["role"] == "أستاذ":
        st.header("👨‍🏫 مركز الرفع")
        tg, ts = st.selectbox("الصف:", list(subs_map.keys())), st.selectbox("المادة:", subs_map["التاسع"]) # تبسيط
        type_f = st.radio("النوع:", ["بحث", "نموذج"])
        up = st.file_uploader("اختر ملف PDF", type=['pdf'])
        if up and st.button("✅ حفظ"):
            folder = "lessons" if type_f == "بحث" else "exams"
            f_name = f"{type_f}_{ts}_{up.name.replace(' ','_')}"
            with open(os.path.join(folder, f_name), "wb") as f: f.write(up.getbuffer())
            f_db = load_data(FILES_DB, ["name", "grade", "sub", "type", "date"])
            pd.concat([f_db, pd.DataFrame([{"name": f_name, "grade": tg, "sub": ts, "type": type_f, "date": datetime.now().strftime("%Y-%m-%d")}])]).to_csv(FILES_DB, index=False)
            st.success("تم الرفع!")

    elif user["role"] == "طالب":
        st.markdown(f'<div class="greeting-box"><h3>{greeting} يا بطل</h3></div>', unsafe_allow_html=True)
        sub = st.selectbox("اختر مادة التركيز:", subs_map[user['grade']])
        t_study, t_ai, t_plan = st.tabs(["📚 ملفاتي", "🤖 المعلم الذكي", "📅 المنقذ (جدول دراسي)"])
        
        with t_study:
            f_db = load_data(FILES_DB, ["name", "grade", "sub", "type", "date"])
            my_f = f_db[(f_db["grade"] == user["grade"]) & (f_db["sub"] == sub)]
            for _, r in my_f.iterrows():
                path = os.path.join("lessons" if r['type'] == "بحث" else "exams", r['name'])
                if os.path.exists(path):
                    with open(path, "rb") as f: st.download_button(f"📥 {r['type']}: {r['name']}", f, file_name=r['name'])

        with t_ai:
            q = st.chat_input("اسأل أي سؤال في المنهاج...")
            if q:
                ans = get_ai_response(f"أنت مدرس سوري، أجب باختصار عن: {q}")
                st.chat_message("assistant").write(ans)
                # ميزة المعلم الناطق
                audio_fp = speak_text(ans)
                if audio_fp:
                    st.audio(audio_fp, format='audio/mp3')
                    st.caption("🔊 اضغط لتسمع شرح المعلم")
            
            st.divider()
            img_file = st.file_uploader("📸 ارفع صورة حلك للتصحيح", type=["jpg", "png"])
            if img_file and st.button("✨ تصحيح ذكي"):
                res = get_ai_response(f"صحح ورقة الطالب في {sub} وأعط علامة من 100.", Image.open(img_file))
                st.write(res)

        with t_plan:
            st.subheader("🗓️ صانع الجداول الذكي")
            col1, col2 = st.columns(2)
            with col1:
                days = st.number_input("كم يوم باقي للامتحان؟", 1, 100, 7)
                hours = st.slider("كم ساعة باليوم بتقدر تدرس؟", 1, 15, 5)
            with col2:
                level = st.select_slider("مستواك بالمادة الحالية:", ["ضعيف", "متوسط", "جيد جداً"])
            
            if st.button("🚀 صمم لي خطة الإنقاذ"):
                with st.spinner("جاري تصميم الجدول..."):
                    plan_prompt = f"أنت خبير تربوي، صمم جدول دراسي لمادة {sub} لصف {user['grade']} لمدة {days} أيام، بمعدل {hours} ساعات يومياً، علماً أن مستوى الطالب {level}. اجعل الجدول مكثفاً ومنظماً."
                    plan_res = get_ai_response(plan_prompt)
                    st.markdown(f'<div class="plan-box">{plan_res}</div>', unsafe_allow_html=True)
                    st.info("💡 نصيحة حسام: الالتزام بالجدول هو سر الـ 600!")
