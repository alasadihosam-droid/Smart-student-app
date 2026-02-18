import streamlit as st
import pandas as pd
from PIL import Image
import os
import google.generativeai as genai
from datetime import datetime

# --- 1. إعدادات الذكاء الاصطناعي ---
API_KEY = "AIzaSyBkrJ1cCsCQtoYGK361daqbaxdlyQWFPKw"
genai.configure(api_key=API_KEY)

def get_ai_response(prompt, image=None):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        if image:
            response = model.generate_content([prompt, image])
        else:
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ عذراً، المساعد الذكي مشغول. حاول مجدداً."

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

# --- 3. الوقت والثيم الذكي ---
now = datetime.now()
hour = now.hour
if 5 <= hour < 18:
    greeting, theme_mode = "☀️ صباح الخير", "light"
    bg, txt, card = "#F0F2F6", "#000000", "#FFFFFF"
else:
    greeting, theme_mode = "🌙 ليلة سعيدة", "dark"
    bg, txt, card = "#0E1117", "#FFFFFF", "#262730"

st.set_page_config(page_title="منصة حسام الذكية", layout="wide")

# تثبيت جلسة المستخدم (منع تسجيل الخروج عند التحديث)
if "user_data" not in st.session_state:
    st.session_state["user_data"] = None

st.markdown(f"""
    <style>
    html, body, [data-testid="stsidebar"] {{ overflow: auto !important; }}
    .stApp {{ background-color: {bg}; color: {txt}; }}
    .block-container {{ max-width: 850px !important; padding: 1rem !important; }}
    .stButton>button {{ 
        width: 100%; border-radius: 12px; height: 3.2em;
        background: linear-gradient(45deg, #D32F2F, #B71C1C);
        color: white; border: none; font-weight: bold;
    }}
    .greeting-box {{ 
        padding: 20px; background-color: {card}; border-radius: 15px; 
        border: 1px solid #D32F2F; text-align: center; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    </style>
    """, unsafe_allow_html=True)

subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "إنكليزي", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "عربي"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "فرنسي", "عربي"]
}

# --- 4. منطق الدخول ---
if st.session_state["user_data"] is None:
    st.markdown(f'<div class="greeting-box"><h1>{greeting}</h1><p>أهلاً بك في منصة الطالب السوري الذكية</p></div>', unsafe_allow_html=True)
    t_log, t_sign = st.tabs(["🔐 تسجيل الدخول", "📝 حساب جديد"])
    
    with t_log:
        u = st.text_input("اسم المستخدم", key="login_u")
        p = st.text_input("كلمة المرور", type="password", key="login_p")
        if st.button("دخول المنصة"):
            if u == "Hosam" and p == "Anahosam031007":
                st.session_state["user_data"] = {"name": u, "role": "Owner", "grade": "الكل"}
                st.rerun()
            else:
                users = load_data(USERS_DB, ["user", "pass", "role", "grade"])
                match = users[(users["user"] == u) & (users["pass"] == p)]
                if not match.empty:
                    st.session_state["user_data"] = {"name": u, "role": match.iloc[0]["role"], "grade": match.iloc[0]["grade"]}
                    st.rerun()
                else: st.error("عذراً، البيانات غير صحيحة")
    
    with t_sign:
        nu = st.text_input("الاسم الكامل", key="sign_u")
        np = st.text_input("كلمة مرور قوية", type="password", key="sign_p")
        nr = st.selectbox("أنا:", ["طالب", "أستاذ"])
        ng = st.selectbox("الصف الدراسي:", list(subs_map.keys())) if nr == "طالب" else "الكل"
        if st.button("إنشاء الحساب"):
            users = load_data(USERS_DB, ["user", "pass", "role", "grade"])
            if nu in users['user'].values: st.error("الاسم مستخدم بالفعل")
            else:
                pd.concat([users, pd.DataFrame([{"user": nu, "pass": np, "role": nr, "grade": ng}])]).to_csv(USERS_DB, index=False)
                st.success("تم الحفظ! سجل دخولك الآن")

else:
    user = st.session_state["user_data"]
    st.sidebar.markdown(f"### 👋 أهلاً {user['name']}\n**{greeting}**")
    if st.sidebar.button("تسجيل الخروج"):
        st.session_state["user_data"] = None
        st.rerun()

    # --- واجهة المدير (حسام) ---
    if user["role"] == "Owner":
        st.header("👑 لوحة تحكم الملك حسام")
        t_u, t_f, t_g = st.tabs(["👥 الأعضاء", "📁 ملفات السيرفر", "📊 الدرجات"])
        with t_u: st.dataframe(load_data(USERS_DB, ["user", "pass", "role", "grade"]), use_container_width=True)
        with t_f: st.dataframe(load_data(FILES_DB, ["name", "grade", "sub", "type", "date"]), use_container_width=True)
        with t_g: st.dataframe(load_data(GRADES_DB, ["student", "subject", "grade", "date"]), use_container_width=True)

    # --- واجهة الأستاذ (تصحيح الرفع) ---
    elif user["role"] == "أستاذ":
        st.header("👨‍🏫 مركز رفع المحتوى")
        tg = st.selectbox("استهداف الصف:", list(subs_map.keys()))
        ts = st.selectbox("المادة العلمية:", subs_map[tg])
        
        def smart_upload(label, folder, type_name):
            st.write(f"📍 {label}")
            up = st.file_uploader(f"ارفع ملف {label}", type=['pdf', 'jpg', 'png'], key=f"up_{type_name}_{tg}_{ts}")
            if up:
                if st.button(f"حفظ {label} نهائياً", key=f"btn_{type_name}_{tg}_{ts}"):
                    f_name = f"{type_name}_{ts}_{up.name.replace(' ','_')}"
                    with open(os.path.join(folder, f_name), "wb") as f: f.write(up.getbuffer())
                    f_db = load_data(FILES_DB, ["name", "grade", "sub", "type", "date"])
                    if f_name not in f_db['name'].values:
                        new_data = pd.DataFrame([{"name": f_name, "grade": tg, "sub": ts, "type": type_name, "date": datetime.now().strftime("%Y-%m-%d")}])
                        pd.concat([f_db, new_data]).to_csv(FILES_DB, index=False)
                    st.success(f"تم رفع {f_name} بنجاح!")
                    st.cache_data.clear()

        c1, c2 = st.columns(2)
        with c1: smart_upload("نوطة الدرس", "lessons", "بحث")
        with c2: smart_upload("نموذج امتحاني", "exams", "نموذج")

    # --- واجهة الطالب (الخدمات الذكية) ---
    elif user["role"] == "طالب":
        st.markdown(f'<div class="greeting-box"><h3>{greeting} يا بطل</h3><p>صفتك: {user["grade"]}</p></div>', unsafe_allow_html=True)
        sub = st.selectbox("اختر المادة للدراسة:", subs_map[user['grade']])
        t_study, t_ai = st.tabs(["📚 مكتبة الملفات", "🤖 المعلم الذكي"])
        
        with t_study:
            f_db = load_data(FILES_DB, ["name", "grade", "sub", "type", "date"])
            my_f = f_db[(f_db["grade"] == user["grade"]) & (f_db["sub"] == sub)]
            if not my_f.empty:
                for _, r in my_f.iterrows():
                    folder = {"بحث": "lessons", "نموذج": "exams", "سلم": "keys"}[r['type']]
                    path = os.path.join(folder, r['name'])
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            st.download_button(f"📥 تحميل {r['type']}: {r['name'].split('_')[-1]}", f, file_name=r['name'])
            else: st.info("لا توجد ملفات مرفوعة لهذه المادة بعد.")

        with t_ai:
            st.subheader("💬 اسأل عن أي شيء في المنهاج")
            q = st.chat_input("اكتب سؤالك هنا...")
            if q:
                with st.spinner("جاري استحضار الإجابة..."):
                    ans = get_ai_response(f"أنت معلم سوري خبير، أجب بدقة من منهاج {user['grade']}: {q}")
                    st.chat_message("assistant").write(ans)
            
            st.divider()
            st.subheader("📝 مصحح الأوراق الآلي")
            img = st.file_uploader("ارفع صورة حلك (واضحة)", type=["jpg", "png", "jpeg"])
            if img and st.button("بدء التصحيح الذكي"):
                with st.spinner("جاري تحليل الخط والحل..."):
                    ans = get_ai_response(f"صحح هذه الورقة لمادة {sub} صف {user['grade']} منهاج سوري، أعط علامة من 100 وملاحظات.", Image.open(img))
                    st.success("اكتمل التحليل!")
                    st.write(ans)
                    # حفظ العلامة في سجل المدير
                    g_db = load_data(GRADES_DB, ["student", "subject", "grade", "date"])
                    pd.concat([g_db, pd.DataFrame([{"student": user['name'], "subject": sub, "grade": ans[:20], "date": datetime.now().strftime("%Y-%m-%d")}])]).to_csv(GRADES_DB, index=False)
