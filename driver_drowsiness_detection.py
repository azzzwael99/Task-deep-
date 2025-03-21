# استيراد المكتبات
from pandas import read_csv, unique
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from tensorflow import stack
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling1D, BatchNormalization, MaxPool1D, Reshape, Activation
from keras.layers import Conv1D, LSTM, Dropout, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping

import warnings
warnings.filterwarnings("ignore")

# تحميل البيانات
def read_data(filepath):
    df = read_csv(filepath, header=None, names=['user-id', 'activity', 'timestamp', 'X', 'Y', 'Z'])
    df['Z'] = df['Z'].replace(to_replace=r';', value='', regex=True).astype(float)
    return df

df = read_data('Dataset/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')

# تجهيز البيانات
le = LabelEncoder()
df['activity'] = le.fit_transform(df['activity'])
x_data = df[['X', 'Y', 'Z']].values
y_data = df['activity'].values
y_data = to_categorical(y_data)

# تقسيم البيانات
from sklearn.model_selection import train_test_split
x_train, x_test, y_train_hot, y_test_hot = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# بناء النموذج CNN-LSTM
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)),
    MaxPool1D(pool_size=2),
    BatchNormalization(),
    Dropout(0.3),  # تقليل Overfitting
    
    Bidirectional(LSTM(64, return_sequences=True)),  # استخدام LSTM ثنائي الاتجاه
    Dropout(0.3),
    
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_train_hot.shape[1], activation='softmax')  # عدد الفئات
])

# إعداد التدريب
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج مع عدد Epochs أقل
history = model.fit(x_train, y_train_hot, batch_size=192, epochs=50, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

# تقييم النموذج
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_hot, axis=1)

# عرض Confusion Matrix بشكل أوضح
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.show()

# تقرير الأداء
print(classification_report(y_true, y_pred_classes))
