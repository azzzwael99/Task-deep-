# استيراد المكتبات الأساسية مع بعض التغييرات
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, BatchNormalization, MaxPool1D, Flatten, Activation
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# تحميل البيانات مع تعديلات على أسماء المتغيرات والدوال
def تحميل_البيانات(المسار):
    بيانات = pd.read_csv(المسار, header=None, names=['المعرف', 'النشاط', 'زمن', 'X', 'Y', 'Z'])
    بيانات['Z'] = بيانات['Z'].str.replace(';', '').astype(float)
    return بيانات

# قراءة البيانات
البيانات = تحميل_البيانات('Dataset/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')

# ترميز الفئات
المحول = LabelEncoder()
البيانات['النشاط'] = المحول.fit_transform(البيانات['النشاط'])

# تقسيم البيانات إلى مدخلات ومخرجات
X_القيم = البيانات[['X', 'Y', 'Z']].values
y_التصنيفات = to_categorical(البيانات['النشاط'])

# إعادة تشكيل البيانات لتمريرها إلى الشبكة العصبية
X_القيم = X_القيم.reshape(X_القيم.shape[0], X_القيم.shape[1], 1)

# إنشاء النموذج العصبي بطريقة معدلة
النموذج = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_القيم.shape[1], 1)),
    BatchNormalization(),
    MaxPool1D(pool_size=2),
    LSTM(50, return_sequences=True),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(y_التصنيفات.shape[1], activation='softmax')
])

# إعداد التدريب
النموذج.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# طباعة ملخص النموذج
النموذج.summary()
