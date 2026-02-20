import streamlit as st
import pandas as pd
from PIL import Image
import os
import google.generativeai as genai
from datetime import datetime
from gtts import gTTS
import io
import hashlib
import re 

# --- 1. إعدادات الأمان والذكاء الاصطناعي ---
try:
    if "GEMINI_API_KEY" in st.secrets:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    else:
        st.error("⚠️ مفتاح API غير موجود. يرجى التأكد من ملف Secrets في إعدادات Streamlit")
        st.stop()
except Exception as e:
    st.error(f"⚠️ خطأ في الوصول إلى Secrets: {e}")
    st.stop()

genai.configure(api_key=API_KEY)

# دالة ذكية لتجربة أحدث موديل، وإذا فشل تستخدم الموديل الأساسي المضمون
def get_ai_response(prompt, image=None):
    try:
        try:
            # المحاولة الأولى: الموديل الأحدث
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            if image:
                response = model.generate_content([prompt, image])
            else:
                response = model.generate_content(prompt)
            return response.text
        except Exception:
            # المحاولة البديلة (Fallback): موديل gemini-pro المضمون 100%
            if image:
                model = genai.GenerativeModel("gemini-pro-vision")
                response = model.generate_content([prompt, image])
            else:
                model = genai.GenerativeModel("gemini-pro")
                response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"⚠️ عذراً، هناك مشكلة في الاتصال بسيرفرات جوجل: {str(e)}"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def speak_text(text):
    try:
        tts = gTTS(text=text[:250], lang='ar')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except:
        return None

# --- 2. تهيئة قواعد البيانات والمجلدات ---
for folder in ['lessons', 'exams', 'db']:
    if not os.path.exists(folder):
        os.makedirs(folder)

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
    try: return pd.read_csv(path)
    except: return pd.DataFrame()

# --- 3. تصميم الواجهة ---
st.set_page_config(page_title="منصة حسام الذكية", layout="wide", page_icon="🎓")

hour = datetime.now().hour
if 5 <= hour < 18:
    greeting, bg, txt, card = "☀️ صباح الخير", "#F0F2F6", "#000000", "#FFFFFF"
else:
    greeting, bg, txt, card = "🌙 ليلة سعيدة", "#0E1117", "#FFFFFF", "#262730"

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
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); color: {txt};
    }}
    .plan-box {{ background-color: #fdf2f2; border-right: 5px solid #D32F2F; padding: 15px; border-radius: 8px; color: black; margin-top: 10px; white-space: pre-wrap; }}
    </style>
    """, unsafe_allow_html=True)

subs_map = {
    "التاسع": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "إنكليزي", "عربي"],
    "البكالوريا العلمي": ["فيزياء", "كيمياء", "علوم", "رياضيات", "فرنسي", "عربي"],
    "البكالوريا الأدبي": ["فلسفة", "تاريخ", "جغرافيا", "فرنسي", "عربي"]
}

if "user_data" not in st.session_state: st.session_state["user_data"] = None
if "chat_history" not in st.session_state: st.session_state["chat_history"] = []

# --- 4. الدخول ---
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
                if not users.empty:
                    match = users[(users["user"] == u) & (users["pass"] == hash_password(p))]
                    if not match.empty:
                        st.session_state["user_data"] = match.iloc[0].to_dict()
                        st.rerun()
                    else: st.error("عذراً، البيانات غير صحيحة")
                else: st.warning("لا يوجد مستخدمين مسجلين بعد.")
    
    with t_sign:
        nu = st.text_input("الاسم الكامل")
        np = st.text_input("كلمة السر", type="password")
        nr = st.selectbox("أنا:", ["طالب", "أستاذ"])
        ng = st.selectbox("الصف:", list(subs_map.keys())) if nr == "طالب" else "الكل"
        if st.button("تأكيد إنشاء الحساب"):
            if nu and np:
                users = load_data(USERS_DB)
                if not users.empty and nu in users['user'].values:
                    st.error("الاسم موجود مسبقاً")
                else:
                    new_user = pd.DataFrame([{"user": nu, "pass": hash_password(np), "role": nr, "grade": ng}])
                    pd.concat([users, new_user], ignore_index=True).to_csv(USERS_DB, index=False)
                    st.success("تم بنجاح! سجل دخولك الآن")

else:
    user = st.session_state["user_data"]
    st.sidebar.markdown(f"### 👋 أهلاً {user['user']}")
    if st.sidebar.button("🔴 تسجيل الخروج"):
        st.session_state["user_data"] = None
        st.session_state["chat_history"] = []
        st.rerun()

    # --- 5. لوحة التحكم ---
    if user["role"] == "Owner":
        st.header("👑 لوحة التحكم العليا")
        t_users, t_files, t_all_grades = st.tabs(["👥 الأعضاء", "📁 الملفات", "📊 درجات الطلاب"])
        with t_users:
            edited_u = st.data_editor(load_data(USERS_DB), num_rows="dynamic")
            if st.button("حفظ تعديلات المستخدمين"):
                edited_u.to_csv(USERS_DB, index=False)
                st.success("تم التحديث")
        with t_files:
            edited_f = st.data_editor(load_data(FILES_DB), num_rows="dynamic")
            if st.button("حفظ تعديلات الملفات"):
                edited_f.to_csv(FILES_DB, index=False)
                st.success("تم التحديث")
        with t_all_grades: st.dataframe(load_data(GRADES_DB), use_container_width=True)

    elif user["role"] == "أستاذ":
        st.header("👨‍🏫 مركز رفع الدروس")
        col1, col2 = st.columns(2)
        with col1: tg = st.selectbox("استهداف الصف:", list(subs_map.keys()))
        with col2: ts = st.selectbox("المادة:", subs_map[tg])
        
        type_f = st.radio("نوع الملف:", ["بحث", "نموذج امتحاني"])
        up = st.file_uploader("اختر الملف (PDF)", type=['pdf'])
        
        if st.button("🚀 رفع الملف الآن"):
            if up is not None:
                clean_name = up.name.replace(' ', '_')
                f_name = f"{type_f}_{ts}_{clean_name}"
                folder = "lessons" if type_f == "بحث" else "exams"
                
                with open(os.path.join(folder, f_name), "wb") as f:
                    f.write(up.read())
                
                f_db = load_data(FILES_DB)
                new_file = pd.DataFrame([{"name": f_name, "grade": tg, "sub": ts, "type": type_f, "date": datetime.now().strftime("%Y-%m-%d")}])
                pd.concat([f_db, new_file], ignore_index=True).to_csv(FILES_DB, index=False)
                st.success(f"تم رفع {f_name} بنجاح!")
                st.balloons()
            else: st.error("⚠️ يرجى اختيار ملف أولاً.")

    elif user["role"] == "طالب":
        st.markdown(f'<div class="greeting-box"><h3>{greeting} يا بطل</h3><p>صفتك: {user["grade"]}</p></div>', unsafe_allow_html=True)
        sub = st.selectbox("اختر المادة للدراسة:", subs_map[user['grade']])
        t_study, t_ai, t_plan, t_progress = st.tabs(["📚 المكتبة", "🤖 المعلم الذكي", "📅 المنقذ", "📊 مستواي"])
        
        with t_study:
            search_q = st.text_input("🔍 ابحث عن درس معين...")
            f_db = load_data(FILES_DB)
            if not f_db.empty:
                my_f = f_db[(f_db["grade"] == user["grade"]) & (f_db["sub"] == sub)]
                if search_q: my_f = my_f[my_f['name'].str.contains(search_q, case=False)]
                if my_f.empty: st.info("لا توجد ملفات مرفوعة لهذه المادة حالياً.")
                else:
                    for _, r in my_f.iterrows():
                        folder = "lessons" if r['type'] == "بحث" else "exams"
                        path = os.path.join(folder, r['name'])
                        if os.path.exists(path):
                            with open(path, "rb") as f:
                                st.download_button(f"📥 تحميل {r['name'].split('_')[-1]}", f, file_name=r['name'], key=r['name'])
            else: st.info("المكتبة فارغة حالياً.")

        with t_ai:
            st.subheader("🤖 اسأل المعلم الذكي")
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]): st.write(msg["content"])
            
            q = st.chat_input("اكتب سؤالك هنا...")
            if q:
                st.session_state["chat_history"].append({"role": "user", "content": q})
                with st.chat_message("user"): st.write(q)
                with st.spinner("جاري التفكير..."): ans = get_ai_response(f"أنت معلم خبير، أجب باختصار عن {sub} لصف {user['grade']}: {q}")
                st.session_state["chat_history"].append({"role": "assistant", "content": ans})
                with st.chat_message("assistant"):
                    st.write(ans)
                    audio = speak_text(ans)
                    if audio: st.audio(audio)

            st.divider()
            st.subheader("📸 مصحح الأوراق الذكي")
            img = st.file_uploader("ارفع صورة حلك (سيتم تقييمها بواسطة AI)", type=["jpg", "png", "jpeg"])
            if img and st.button("ابدأ التصحيح"):
                with st.spinner("جاري تحليل الصورة..."):
                    res = get_ai_response(f"صحح ورقة الطالب في {sub} لصف {user['grade']} واعط علامة من 100. ابدأ بالدرجة كأول كلمة.", Image.open(img))
                st.info(res)
                try:
                    match = re.search(r'\d+', res)
                    if match:
                        score = min(int(match.group()), 100)
                        g_db = load_data(GRADES_DB)
                        new_g = pd.DataFrame([{"user": user['user'], "sub": sub, "score": score, "date": datetime.now().strftime("%m-%d %H:%M")}])
                        pd.concat([g_db, new_g], ignore_index=True).to_csv(GRADES_DB, index=False)
                        st.toast(f"تم تسجيل الدرجة: {score}/100")
                except: st.warning("لم يتم استخراج الدرجة رقمياً.")

        with t_plan:
            st.subheader("📅 خطة الدراسة السريعة")
            col_a, col_b = st.columns(2)
            d = col_a.number_input("الأيام المتبقية:", 1, 100, 7)
            h = col_b.slider("الساعات يومياً:", 1, 15, 6)
            if st.button("توليد الخطة"):
                with st.spinner("جاري التصميم..."): plan = get_ai_response(f"اصنع خطة دراسة لـ {sub} لصف {user['grade']} في {d} أيام، {h} ساعات يومياً.")
                st.markdown(f'<div class="plan-box">{plan}</div>', unsafe_allow_html=True)

        with t_progress:
            st.subheader(f"📈 مستوى {user['user']} في {sub}")
            g_db = load_data(GRADES_DB)
            my_scores = g_db[(g_db["user"] == user["user"]) & (g_db["sub"] == sub)]
            if not my_scores.empty:
                st.line_chart(my_scores.set_index("date")["score"])
                st.metric("متوسط الدرجات", f"{my_scores['score'].mean():.1f}%")
            else: st.info("ابدأ بالتصحيح الآلي لتظهر نتائجك هنا!")
