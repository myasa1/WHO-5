

import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
import openai
import os

# ---------- 1️⃣ استخدام مفتاح OpenAI من Secrets ----------
# تأكدي من إضافة مفتاحك في Streamlit Cloud: OPENAI_API_KEY="sk-XXXX"
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------- 2️⃣ نموذج ML صناعي لتنبؤ الخطورة ----------
@st.cache_data
def generate_and_train_model():
    np.random.seed(42)
    num_users = 1000
    Q = np.random.randint(0,6,(num_users,5))
    SleepHours = np.random.randint(4,11, num_users)
    ActivityLevel = np.random.randint(1,6, num_users)
    TotalScore = Q.sum(axis=1)
    HighRisk = (TotalScore <= 12).astype(int)
    
    X = np.column_stack((Q, SleepHours, ActivityLevel))
    y = HighRisk
    model = LogisticRegression()
    model.fit(X, y)
    return model

model = generate_and_train_model()

# ---------- 3️⃣ واجهة المستخدم ----------
st.title("🧠 تقييم الرفاهية النفسية WHO-5 + خطة AI شخصية")
st.markdown("أدخل بياناتك للحصول على تقرير شخصي وخطة تحسين لمدة 6 أسابيع:")

q1 = st.slider("المزاج الإيجابي (Q1)", 0,5,2)
q2 = st.slider("الهدوء والاسترخاء (Q2)", 0,5,2)
q3 = st.slider("الطاقة والحيوية (Q3)", 0,5,2)
q4 = st.slider("جودة النوم (Q4)", 0,5,2)
q5 = st.slider("الإحساس بالمعنى (Q5)", 0,5,2)
sleep = st.slider("عدد ساعات النوم", 4, 10, 7)
activity = st.slider("مستوى النشاط البدني (1 منخفض - 5 عالي)", 1,5,3)

# ---------- 4️⃣ زر التنبؤ وخطة AI ----------
if st.button("احصل على تقريري وخطتي AI"):
    user_input = np.array([[q1,q2,q3,q4,q5,sleep,activity]])
    probability = model.predict_proba(user_input)[0][1]
    
    # التقرير السريع
    st.markdown(f"### 📊 نتائج تقييمك")
    st.markdown(f"- درجات WHO-5: Q1={q1}, Q2={q2}, Q3={q3}, Q4={q4}, Q5={q5}")
    st.markdown(f"- ساعات النوم: {sleep}")
    st.markdown(f"- النشاط البدني: {activity}")
    st.markdown(f"- احتمال انخفاض شديد في الرفاهية: **{probability*100:.1f}%**")
    
    if probability > 0.75:
        st.warning("⚠️ يوصى بالحصول على تقييم متخصص من مختص نفسي.")
    else:
        st.success("المستوى ضمن النطاق المقبول حاليًا.")
    
    st.info("💡 هذا التقييم أداة داعمة فقط، ولا يُعد تشخيصًا طبيًا.")
    
    # ---------- توليد خطة AI ديناميكية ----------
    prompt = f"""
أنت مستشار نفسي خبير. 
المستخدم لديه درجات WHO-5 كالتالي: 
Q1={q1}, Q2={q2}, Q3={q3}, Q4={q4}, Q5={q5}
عدد ساعات النوم: {sleep}
مستوى النشاط البدني: {activity}
احتمال انخفاض الرفاهية: {probability*100:.1f}%

اصنع له **خطة تحسين شخصية باللغة العربية لمدة 6 أسابيع**، تشمل:
- تمارين تحسين المزاج
- تمارين تفعيل سلوكي للطاقة والنشاط
- تحسين جودة النوم
- تعزيز الإحساس بالمعنى
- تمارين استرخاء
- تكون ودية، داعمة، واقعية، بدون تشخيص
قسّم كل أسبوع بمسمى الأسبوع وشرح تمارين يومية قصيرة.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7
    )
    
    plan_ai = response['choices'][0]['message']['content']
    
    st.markdown("### 📅 خطتك الشخصية لمدة 6 أسابيع")
    st.markdown(plan_ai)
