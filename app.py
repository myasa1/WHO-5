import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from openai import OpenAI
import os

# ----------------------------
# 1️⃣ إعداد مفتاح OpenAI
# ----------------------------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ----------------------------
# 2️⃣ تدريب نموذج ML بسيط
# ----------------------------
@st.cache_data
def generate_and_train_model():
    np.random.seed(42)
    num_users = 1000

    Q = np.random.randint(0, 6, (num_users, 5))
    SleepHours = np.random.randint(4, 11, num_users)
    ActivityLevel = np.random.randint(1, 6, num_users)

    TotalScore = Q.sum(axis=1)
    HighRisk = (TotalScore <= 12).astype(int)

    X = np.column_stack((Q, SleepHours, ActivityLevel))
    y = HighRisk

    model = LogisticRegression()
    model.fit(X, y)

    return model

model = generate_and_train_model()

# ----------------------------
# 3️⃣ واجهة المستخدم
# ----------------------------
st.title("🧠 تقييم الرفاهية النفسية WHO-5 + خطة AI شخصية")
st.markdown("أدخل بياناتك للحصول على تقرير وخطة تحسين لمدة 6 أسابيع")

q1 = st.slider("المزاج الإيجابي (Q1)", 0, 5, 2)
q2 = st.slider("الهدوء والاسترخاء (Q2)", 0, 5, 2)
q3 = st.slider("الطاقة والحيوية (Q3)", 0, 5, 2)
q4 = st.slider("جودة النوم (Q4)", 0, 5, 2)
q5 = st.slider("الإحساس بالمعنى (Q5)", 0, 5, 2)

sleep = st.slider("عدد ساعات النوم", 4, 10, 7)
activity = st.slider("مستوى النشاط البدني (1 منخفض - 5 عالي)", 1, 5, 3)

# ----------------------------
# 4️⃣ عند الضغط على الزر
# ----------------------------
if st.button("احصل على تقريري وخطتي AI"):

    user_input = np.array([[q1, q2, q3, q4, q5, sleep, activity]])
    probability = model.predict_proba(user_input)[0][1]

    st.markdown("### 📊 نتائج تقييمك")
    st.write(f"احتمال انخفاض الرفاهية: {probability*100:.1f}%")

    if probability > 0.75:
        st.warning("⚠️ يوصى بمراجعة مختص نفسي.")
    else:
        st.success("المستوى ضمن النطاق المقبول حالياً.")

    st.info("هذا التقييم أداة داعمة فقط ولا يُعد تشخيصاً طبياً.")

    # ----------------------------
    # 5️⃣ إنشاء الخطة عبر OpenAI (الطريقة الصحيحة v1)
    # ----------------------------

    prompt = f"""
    المستخدم حصل على:
    Q1={q1}, Q2={q2}, Q3={q3}, Q4={q4}, Q5={q5}
    ساعات النوم={sleep}
    مستوى النشاط={activity}
    احتمال انخفاض الرفاهية={probability*100:.1f}%

    أنشئ خطة تحسين نفسية لمدة 6 أسابيع باللغة العربية،
    تكون داعمة وودية وغير تشخيصية،
    وتشمل:
    - تحسين المزاج
    - زيادة الطاقة
    - تحسين النوم
    - تمارين استرخاء
    - تعزيز المعنى
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "أنت مستشار نفسي خبير."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        plan_ai = response.choices[0].message.content

        st.markdown("### 📅 خطتك الشخصية")
        st.write(plan_ai)

    except Exception as e:
        st.error("حدث خطأ في الاتصال بـ OpenAI. تأكدي من المفتاح.")
        st.write(e)
