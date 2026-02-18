import streamlit as st
import pandas as pd
from PIL import Image
import os
import google.generativeai as genai

# --- 1. إعدادات الذكاء الاصطناعي ---
# استبدل بمفتاحك إذا تغير
genai.configure(api_key="AIzaSyBkrJ1cCsCQtoYGK361daqbaxdlyQWFPKw")
model = genai.GenerativeModel("gemini-1.5-flash")

# --- 2. إنشاء نظام المجلدات وقواعد البيانات ---
# ملاحظة: المجلدات ستنشأ تلقائياً عند التشغيل الأول على السيرفر
for folder in ['lessons', 'exams', 'keys', 'db']:
    os.makedirs(folder, exist_ok=True)

USERS_DB = "db/users.csv"
FILES_DB = "db/files.csv"

def load_data(path, columns):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)

# --- 3. تصميم الواجهة ---
st.set_page_config(page_title="منصة الطالب الذكي - الأستاذ حسام", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #D32F2F; color: white; }
    </style>
    """, unsafe_allow_html=True)

# خريطة المواد
subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "إنكليزي", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "عربي"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "فرنسي", "عربي"]
}

# --- 4. نظام الدخول ---
if "user_data" not in st.session_state:
    st.title("🚀 مرحباً بك في منصة الأستاذ حسام")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔐 تسجيل الدخول")
        u = st.text_input("اسم المستخدم", key="login_u")
        p = st.text_input("كلمة المرور", type="password", key="login_p")
        if st.button("دخول"):
            if u == "Hosam" and p == "Anahosam031007":
                st.session_state["user_data"] = {"name": u, "role": "Owner", "grade": "الكل"}
                st.rerun()
            else:
                users = load_data(USERS_DB, ["user", "pass", "role", "grade"])
                match = users[(users["user"] == u) & (users["pass"] == p)]
                if not match.empty:
                    st.session_state["user_data"] = {"name": u, "role": match.iloc[0]["role"], "grade": match.iloc[0]["grade"]}
                    st.rerun()
                else: st.error("خطأ في البيانات!")

    with col2:
        st.subheader("📝 إنشاء حساب جديد")
        new_u = st.text_input("الاسم الجديد")
        new_p = st.text_input("كلمة المرور الجديدة", type="password")
        new_r = st.selectbox("نوع الحساب", ["طالب", "أستاذ"])
        new_g = st.selectbox("الصف", list(subs_map.keys())) if new_r == "طالب" else "الكل"
        if st.button("تأكيد الاشتراك"):
            users = load_data(USERS_DB, ["user", "pass", "role", "grade"])
            if new_u in users["user"].values: st.error("الاسم موجود مسبقاً!")
            else:
                new_entry = pd.DataFrame([{"user": new_u, "pass": new_p, "role": new_r, "grade": new_g}])
                pd.concat([users, new_entry]).to_csv(USERS_DB, index=False)
                st.success("تم بنجاح! سجل دخولك الآن.")

# --- 5. محتوى التطبيق بعد الدخول ---
else:
    data = st.session_state["user_data"]
    st.sidebar.title(f"👤 {data['name']}")
    st.sidebar.info(f"الرتبة: {data['role']}")
    if st.sidebar.button("تسجيل الخروج"):
        del st.session_state["user_data"]; st.rerun()

    # واجهة المدير (حسام)
    if data["role"] == "Owner":
        st.header("👑 لوحة التحكم المطلقة")
        tab1, tab2 = st.tabs(["👥 الأعضاء", "📂 المنشورات"])
        with tab1:
            users = load_data(USERS_DB, ["user", "pass", "role", "grade"])
            for i, r in users.iterrows():
                c1, c2 = st.columns([3, 1])
                c1.write(f"**{r['user']}** - {r['role']} ({r['grade']})")
                if c2.button("حذف", key=f"del_u_{i}"):
                    users.drop(i).to_csv(USERS_DB, index=False); st.rerun()
        with tab2:
            files = load_data(FILES_DB, ["name", "grade", "sub", "type"])
            for i, r in files.iterrows():
                c1, c2 = st.columns([3, 1])
                c1.write(f"**{r['name']}** - {r['grade']}")
                if c2.button("حذف", key=f"del_f_{i}"):
                    files.drop(i).to_csv(FILES_DB, index=False); st.rerun()

    # واجهة الأستاذ
    elif data["role"] == "أستاذ":
        st.header("👨‍🏫 لوحة نشر الدروس والاختبارات")
        tg = st.selectbox("الصف المستهدف", list(subs_map.keys()))
        ts = st.selectbox("المادة", subs_map[tg])
        tt = st.radio("نوع الملف", ["بحث PDF", "نموذج امتحان", "سلم تصحيح"])
        up = st.file_uploader("ارفع الملف")
        if up and st.button("نشر الآن"):
            f_db = load_data(FILES_DB, ["name", "grade", "sub", "type"])
            new_f = pd.DataFrame([{"name": up.name, "grade": tg, "sub": ts, "type": tt}])
            pd.concat([f_db, new_f]).to_csv(FILES_DB, index=False)
            folder = {"بحث PDF": "lessons", "نموذج امتحان": "exams", "سلم تصحيح": "keys"}[tt]
            with open(os.path.join(folder, up.name), "wb") as f: f.write(up.getbuffer())
            st.success("✅ تم النشر بنجاح!")

    # واجهة الطالب
    else:
        st.header(f"🎓 بوابة الطالب: {data['grade']}")
        subject = st.selectbox("اختر المادة", subs_map[data['grade']])
        t1, t2 = st.tabs(["📚 محتوى الدراسة", "🤖 ذكاء اصطناعي"])
        
        with t1:
            f_db = load_data(FILES_DB, ["name", "grade", "sub", "type"])
            my_files = f_db[(f_db["grade"] == data["grade"]) & (f_db["sub"] == subject)]
            if not my_files.empty:
                for _, r in my_files.iterrows():
                    folder = {"بحث PDF": "lessons", "نموذج امتحان": "exams", "سلم تصحيح": "keys"}[r['type']]
                    with open(os.path.join(folder, r['name']), "rb") as f:
                        st.download_button(f"تحميل {r['type']}: {r['name']}", f, file_name=r['name'])
            else: st.info("لا يوجد ملفات مرفوعة حالياً.")

        with t2:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("📝 توليد نموذج امتحان"):
                    with st.spinner("جاري كتابة الأسئلة..."):
                        res = model.generate_content(f"اكتب نموذج امتحان {subject} لصف {data['grade']} منهاج سوري مع الحل.")
                        st.write(res.text)
            with col_b:
                img = st.file_uploader("📸 ارفع صورة حلك للتصحيح", type=["jpg", "png", "jpeg"])
                if img and st.button("بدء التصحيح"):
                    with st.spinner("الذكاء الاصطناعي يحلل ورقتك..."):
                        res = model.generate_content([f"أنت أستاذ خبير، صحح ورقة {subject} لصف {data['grade']} وأعطِ علامة من 100.", Image.open(img)])
                        st.markdown("### 📝 النتيجة:")
                        st.write(res.text)
