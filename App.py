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

# ==========================================
# 1. إعدادات الأمان والذكاء الاصطناعي
# ==========================================
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

def get_ai_response(prompt, image=None):
    """دالة الاتصال بالذكاء الاصطناعي - تم استخدام الموديل المعتمد والمستقر"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        if image:
            response = model.generate_content([prompt, image])
        else:
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ عذراً، هناك مشكلة في الاتصال بسيرفرات جوجل. تفاصيل الخطأ: {str(e)}"

def hash_password(password):
    """تشفير كلمات المرور لحمايتها"""
    return hashlib.sha256(password.encode()).hexdigest()

def speak_text(text):
    """تحويل النص إلى صوت"""
    try:
        tts = gTTS(text=text[:250], lang='ar')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except:
        return None

# ==========================================
# 2. تهيئة قواعد البيانات والمجلدات
# ==========================================
# إنشاء المجلدات إذا لم تكن موجودة
for folder in ['lessons', 'exams', 'db']:
    if not os.path.exists(folder):
        os.makedirs(folder)

USERS_DB = "db/users.csv"
FILES_DB = "db/files.csv"
GRADES_DB = "db/grades.csv"

def init_db(path, columns):
    """إنشاء ملفات CSV فارغة برؤوس الأعمدة إذا لم تكن موجودة"""
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)

init_db(USERS_DB, ["user", "pass", "role", "grade"])
init_db(FILES_DB, ["name", "grade", "sub", "type", "date"])
init_db(GRADES_DB, ["user", "sub", "score", "date"])

def load_data(path):
    """قراءة البيانات من ملفات CSV"""
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

# ==========================================
# 3. إعدادات الواجهة والتصميم العام
# ==========================================
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

# ==========================================
# 4. نظام الجلسات (Session State)
# ==========================================
if "user_data" not in st.session_state:
    st.session_state["user_data"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ==========================================
# 5. شاشة تسجيل الدخول وإنشاء الحساب
# ==========================================
if st.session_state["user_data"] is None:
    st.markdown(f'<div class="greeting-box"><h1>{greeting}</h1><p>أهلاً بك في منصة حسام التعليمية المطورة</p></div>', unsafe_allow_html=True)
    t_log, t_sign = st.tabs(["🔐 تسجيل الدخول", "📝 إنشاء حساب"])
    
    with t_log:
        login_col1, login_col2 = st.columns([1, 1])
        with login_col1:
            u = st.text_input("اسم المستخدم", key="login_u")
        with login_col2:
            p = st.text_input("كلمة المرور", type="password", key="login_p")
            
        if st.button("دخول المنصة"):
            if u == "Hosam" and p == "Anahosam031007":
                st.session_state["user_data"] = {"user": u, "role": "Owner", "grade": "الكل"}
                st.rerun()
            else:
                users = load_data(USERS_DB)
                if not users.empty:
                    hashed_p = hash_password(p)
                    match = users[(users["user"] == u) & (users["pass"] == hashed_p)]
                    if not match.empty:
                        st.session_state["user_data"] = match.iloc[0].to_dict()
                        st.rerun()
                    else:
                        st.error("عذراً، البيانات غير صحيحة")
                else:
                    st.warning("لا يوجد مستخدمين مسجلين بعد.")
    
    with t_sign:
        sign_col1, sign_col2 = st.columns([1, 1])
        with sign_col1:
            nu = st.text_input("الاسم الكامل")
            nr = st.selectbox("أنا:", ["طالب", "أستاذ"])
        with sign_col2:
            np = st.text_input("كلمة السر", type="password")
            ng = st.selectbox("الصف:", list(subs_map.keys())) if nr == "طالب" else "الكل"
            
        if st.button("تأكيد إنشاء الحساب"):
            if nu and np:
                users = load_data(USERS_DB)
                if not users.empty and nu in users['user'].values:
                    st.error("الاسم موجود مسبقاً، يرجى اختيار اسم آخر.")
                else:
                    new_user = pd.DataFrame([{
                        "user": nu, "pass": hash_password(np), "role": nr, "grade": ng
                    }])
                    pd.concat([users, new_user], ignore_index=True).to_csv(USERS_DB, index=False)
                    st.success("تم إنشاء الحساب بنجاح! يمكنك تسجيل الدخول الآن.")
            else:
                st.warning("يرجى تعبئة جميع الحقول.")

# ==========================================
# 6. شاشات المستخدمين (بعد تسجيل الدخول)
# ==========================================
else:
    user = st.session_state["user_data"]
    
    # القائمة الجانبية (Sidebar)
    st.sidebar.markdown(f"### 👋 أهلاً {user['user']}")
    st.sidebar.markdown("---")
    if st.sidebar.button("🔴 تسجيل الخروج"):
        st.session_state["user_data"] = None
        st.session_state["chat_history"] = []
        st.rerun()

    # ----------------------------------------
    # واجهة الإدارة (Owner)
    # ----------------------------------------
    if user["role"] == "Owner":
        st.header("👑 لوحة التحكم العليا")
        t_users, t_files, t_all_grades = st.tabs(["👥 الأعضاء", "📁 الملفات", "📊 درجات الطلاب"])
        
        with t_users:
            st.subheader("إدارة المستخدمين")
            edited_u = st.data_editor(load_data(USERS_DB), num_rows="dynamic", use_container_width=True)
            if st.button("حفظ تعديلات المستخدمين"):
                edited_u.to_csv(USERS_DB, index=False)
                st.success("تم التحديث بنجاح")
                
        with t_files:
            st.subheader("إدارة الملفات المرفوعة")
            edited_f = st.data_editor(load_data(FILES_DB), num_rows="dynamic", use_container_width=True)
            if st.button("حفظ تعديلات الملفات"):
                edited_f.to_csv(FILES_DB, index=False)
                st.success("تم التحديث بنجاح")
                
        with t_all_grades:
            st.subheader("سجل درجات الطلاب")
            st.dataframe(load_data(GRADES_DB), use_container_width=True)

    # ----------------------------------------
    # واجهة الأستاذ (المعلم)
    # ----------------------------------------
    elif user["role"] == "أستاذ":
        st.header("👨‍🏫 مركز رفع الدروس والملفات")
        st.info("ملاحظة: يمكنك رفع ملفات PDF للطلاب هنا. يرجى اختيار الصف والمادة بدقة.")
        
        col1, col2 = st.columns(2)
        with col1:
            tg = st.selectbox("استهداف الصف:", list(subs_map.keys()))
        with col2:
            ts = st.selectbox("المادة:", subs_map[tg])
        
        type_f = st.radio("نوع الملف:", ["بحث", "نموذج امتحاني"])
        
        # أداة الرفع موجودة خارج الـ form لمنع خطأ Network Error على الموبايل
        up = st.file_uploader("اختر الملف (PDF)", type=['pdf'])
        
        if st.button("🚀 رفع الملف الآن"):
            if up is not None:
                try:
                    # تنظيف اسم الملف من المسافات
                    clean_name = up.name.replace(' ', '_')
                    f_name = f"{type_f}_{ts}_{clean_name}"
                    folder = "lessons" if type_f == "بحث" else "exams"
                    file_path = os.path.join(folder, f_name)
                    
                    # حفظ الملف في المجلد المخصص
                    with open(file_path, "wb") as f:
                        f.write(up.read())
                    
                    # تسجيل الملف في قاعدة البيانات
                    f_db = load_data(FILES_DB)
                    new_file = pd.DataFrame([{
                        "name": f_name, 
                        "grade": tg, 
                        "sub": ts, 
                        "type": type_f, 
                        "date": datetime.now().strftime("%Y-%m-%d")
                    }])
                    pd.concat([f_db, new_file], ignore_index=True).to_csv(FILES_DB, index=False)
                    
                    st.success(f"تم رفع الملف '{f_name}' بنجاح! ✅")
                    st.balloons()
                except Exception as e:
                    st.error(f"حدث خطأ أثناء حفظ الملف: {e}")
            else:
                st.error("⚠️ يرجى اختيار ملف من جهازك أولاً قبل الضغط على زر الرفع.")

    # ----------------------------------------
    # واجهة الطالب
    # ----------------------------------------
    elif user["role"] == "طالب":
        st.markdown(f'<div class="greeting-box"><h3>{greeting} يا بطل</h3><p>الصف: {user["grade"]}</p></div>', unsafe_allow_html=True)
        
        sub = st.selectbox("اختر المادة التي ترغب بدراستها:", subs_map[user['grade']])
        t_study, t_ai, t_plan, t_progress = st.tabs(["📚 المكتبة والدروس", "🤖 المعلم والمصحح الذكي", "📅 منقذ الامتحانات", "📊 مستواي الدراسي"])
        
        # 1. المكتبة
        with t_study:
            search_q = st.text_input("🔍 ابحث عن اسم درس أو ملف معين...")
            f_db = load_data(FILES_DB)
            
            if not f_db.empty:
                # فلترة الملفات حسب صف الطالب والمادة المختارة
                my_f = f_db[(f_db["grade"] == user["grade"]) & (f_db["sub"] == sub)]
                if search_q:
                    my_f = my_f[my_f['name'].str.contains(search_q, case=False)]
                
                if my_f.empty:
                    st.info("لا توجد ملفات مرفوعة لهذه المادة حتى الآن. عد لاحقاً!")
                else:
                    for _, r in my_f.iterrows():
                        folder = "lessons" if r['type'] == "بحث" else "exams"
                        path = os.path.join(folder, r['name'])
                        
                        if os.path.exists(path):
                            with open(path, "rb") as f:
                                st.download_button(
                                    label=f"📥 تحميل: {r['name'].split('_')[-1]} ({r['type']})", 
                                    data=f, 
                                    file_name=r['name'], 
                                    key=r['name']
                                )
                        else:
                            st.warning(f"الملف {r['name']} مسجل في النظام لكنه غير موجود في المجلد.")
            else:
                st.info("المكتبة فارغة حالياً. لا يوجد أي ملفات في النظام.")

        # 2. المعلم الذكي
        with t_ai:
            st.subheader("💬 اسأل المعلم الذكي")
            
            # عرض المحادثات السابقة
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # مربع إدخال السؤال
            q = st.chat_input("اكتب سؤالك هنا...")
            if q:
                st.session_state["chat_history"].append({"role": "user", "content": q})
                with st.chat_message("user"):
                    st.write(q)
                
                with st.spinner("المعلم يكتب الإجابة..."):
                    ai_prompt = f"أنت معلم خبير ومتعاون. أجب بشكل مبسط ومباشر عن هذا السؤال في مادة {sub} لصف {user['grade']}: {q}"
                    ans = get_ai_response(ai_prompt)
                
                st.session_state["chat_history"].append({"role": "assistant", "content": ans})
                with st.chat_message("assistant"):
                    st.write(ans)
                    # زر قراءة الصوت
                    audio = speak_text(ans)
                    if audio:
                        st.audio(audio, format="audio/mp3")

            st.markdown("---")
            
            # المصحح الآلي
            st.subheader("📸 مصحح الأوراق الذكي")
            st.write("قم بتصوير حلك على الورقة، وسيقوم المعلم الذكي بتصحيحه وإعطائك درجة من 100.")
            img = st.file_uploader("ارفع صورة الحل", type=["jpg", "png", "jpeg"])
            
            if img and st.button("ابدأ عملية التصحيح"):
                with st.spinner("جاري قراءة الصورة وتحليل الحل..."):
                    img_opened = Image.open(img)
                    grader_prompt = f"قم بدور مصحح امتحانات صارم وعادل. هذه صورة لحل طالب في مادة {sub} لصف {user['grade']}. صحح الورقة، اذكر الأخطاء إن وجدت، وفي النهاية أعطِ علامة من 100. (اكتب الدرجة كرقم بوضوح)."
                    res = get_ai_response(grader_prompt, img_opened)
                
                st.info(res)
                
                # محاولة استخراج الدرجة لحفظها في قاعدة البيانات
                try:
                    match = re.search(r'\d+', res)
                    if match:
                        score = min(int(match.group()), 100) # الدرجة لا تتجاوز 100
                        g_db = load_data(GRADES_DB)
                        new_g = pd.DataFrame([{
                            "user": user['user'], 
                            "sub": sub, 
                            "score": score, 
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
                        }])
                        pd.concat([g_db, new_g], ignore_index=True).to_csv(GRADES_DB, index=False)
                        st.toast(f"تم تسجيل نتيجتك في هذه المادة: {score}/100")
                except:
                    st.warning("تم التصحيح، لكن لم يتمكن النظام من استخراج رقم الدرجة تلقائياً لحفظها في السجل.")

        # 3. المنقذ (خطة الدراسة)
        with t_plan:
            st.subheader("📅 خطة الدراسة السريعة (المنقذ)")
            st.write("هل اقترب الامتحان وأنت متوتر؟ دع الذكاء الاصطناعي يضع لك خطة طوارئ.")
            
            col_a, col_b = st.columns(2)
            d = col_a.number_input("كم يوماً متبقي للامتحان؟", min_value=1, max_value=100, value=7)
            h = col_b.slider("كم ساعة يمكنك أن تدرس يومياً؟", min_value=1, max_value=15, value=5)
            
            if st.button("توليد خطة الإنقاذ"):
                with st.spinner("جاري تصميم خطة تناسب وقتك..."):
                    plan_prompt = f"أنا طالب في صف {user['grade']} ولدي امتحان في مادة {sub}. متبقي للامتحان {d} أيام، ويمكنني الدراسة {h} ساعات يومياً. ضع لي جدولاً دراسياً مقسماً بالساعات والأيام للانتهاء من المادة بنجاح."
                    plan = get_ai_response(plan_prompt)
                
                st.markdown(f'<div class="plan-box">{plan}</div>', unsafe_allow_html=True)

        # 4. مستوى الطالب
        with t_progress:
            st.subheader(f"📈 تطور مستواك في مادة {sub}")
            g_db = load_data(GRADES_DB)
            
            my_scores = g_db[(g_db["user"] == user["user"]) & (g_db["sub"] == sub)]
            if not my_scores.empty:
                # تحويل البيانات لرسم بياني
                chart_data = my_scores.set_index("date")["score"]
                st.line_chart(chart_data)
                
                avg_score = my_scores['score'].mean()
                st.metric(label="متوسط درجاتك (من 100)", value=f"{avg_score:.1f}%")
            else:
                st.info("لا يوجد لك درجات مسجلة في هذه المادة بعد. استخدم 'المصحح الذكي' لتبدأ درجاتك بالظهور هنا!")
