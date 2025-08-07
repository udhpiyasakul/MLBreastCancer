
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(layout="wide")
st.title("📊 ระบบวิเคราะห์ข้อมูล Classification พร้อม Holdout และ Export")

# =============================
# 📁 อัปโหลดไฟล์ CSV หรือใช้ Dataset ตัวอย่าง
# =============================
st.sidebar.header("📥 อัปโหลดข้อมูล")
use_example = st.sidebar.checkbox("ใช้ตัวอย่าง Breast Cancer Dataset", value=True)

if use_example:
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    target_col = "target"
    class_names = list(data.target_names)
else:
    uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ .csv", type=["csv"])
    target_col = st.sidebar.text_input("🧪 ระบุชื่อ column ที่เป็นเป้าหมาย (target)", value="target")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        class_names = list(df[target_col].unique())

if 'df' in locals():
    st.subheader("👁️ ข้อมูลตัวอย่าง (head)")
    st.dataframe(df.head())

    st.markdown("### 📌 ข้อมูลเบื้องต้น")
    st.write(f"จำนวนข้อมูลทั้งหมด: {df.shape[0]} ตัวอย่าง")
    st.write(f"จำนวนฟีเจอร์: {df.shape[1] - 1}")
    st.write(f"จำนวนคลาส: {df[target_col].nunique()}")
    st.write(f"ชื่อคลาส: {class_names}")

    # ตรวจ Missing Values
    st.markdown("### ❗ ตรวจสอบ Missing Values")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("✅ ไม่มี Missing Values")
    else:
        st.warning("⚠️ พบ Missing Values")
        st.dataframe(missing[missing > 0])

    # แยก holdout set
    holdout_count = st.sidebar.number_input("🔢 จำนวน Holdout ต่อคลาส", min_value=1, max_value=30, value=5, step=1)
    holdout = df.groupby(target_col).apply(lambda x: x.sample(min(holdout_count, len(x)), random_state=42)).reset_index(drop=True)
    df_clean = df.drop(holdout.index)

    X = df_clean.drop(target_col, axis=1)
    y = df_clean[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # สถิติข้อมูล
    st.markdown("### 📊 สถิติ Training Set")
    st.dataframe(X_train.describe())

    # Histogram
    st.markdown("### 📈 การกระจายข้อมูล")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(X_train.columns[:6]):
        sns.histplot(X_train[col], kde=True, ax=axes[i])
        axes[i].set_title(col)
    st.pyplot(fig)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    pca_df["target"] = y_train.values

    st.markdown("### 🌈 การกระจายข้อมูล PCA (2D)")
    fig2, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="target", palette="Set2", ax=ax)
    st.pyplot(fig2)

    # Model training
    model1 = LogisticRegression(max_iter=1000)
    model2 = RandomForestClassifier()

    model1.fit(X_train_scaled, y_train)
    model2.fit(X_train_scaled, y_train)

    y_pred1 = model1.predict(X_test_scaled)
    y_pred2 = model2.predict(X_test_scaled)

    report1 = classification_report(y_test, y_pred1, target_names=[str(c) for c in class_names], output_dict=True)
    report2 = classification_report(y_test, y_pred2, target_names=[str(c) for c in class_names], output_dict=True)

    st.markdown("### 🧪 ประเมินผลบน Test Set")
    st.write("🔹 Logistic Regression")
    st.dataframe(pd.DataFrame(report1).transpose())
    st.write("🔹 Random Forest")
    st.dataframe(pd.DataFrame(report2).transpose())

    # Evaluate on holdout
    X_holdout = holdout.drop(target_col, axis=1)
    y_holdout = holdout[target_col]
    X_holdout_scaled = scaler.transform(X_holdout)

    holdout_acc1 = accuracy_score(y_holdout, model1.predict(X_holdout_scaled))
    holdout_acc2 = accuracy_score(y_holdout, model2.predict(X_holdout_scaled))

    st.markdown("### ✅ ประเมินบน Holdout Set")
    st.write(f"🎯 Logistic Regression Accuracy: {holdout_acc1:.2f}")
    st.write(f"🌲 Random Forest Accuracy: {holdout_acc2:.2f}")

    # ✅ Export to Excel
    st.markdown("### 📤 ดาวน์โหลดผลลัพธ์")
    output_excel = BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        pd.DataFrame(report1).transpose().to_excel(writer, sheet_name="Logistic_Regression")
        pd.DataFrame(report2).transpose().to_excel(writer, sheet_name="Random_Forest")
        holdout.to_excel(writer, sheet_name="Holdout_Samples", index=False)
        pd.DataFrame(X_train.describe()).to_excel(writer, sheet_name="Train_Stats")

    st.download_button(
        label="⬇️ ดาวน์โหลดรายงาน (Excel)",
        data=output_excel.getvalue(),
        file_name="classification_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("โปรดอัปโหลดไฟล์ CSV หรือเลือกใช้ dataset ตัวอย่างทางด้านซ้าย")
