import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

data = pd.read_csv("master_dataset.csv")

x = data.drop(['label'], axis = 1)
y = data['label']
split_size = 0.3
seed = 42
scoring = 'accuracy'

# y data needs to be encoded for TensorFlow
# Convert string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y_encoded, test_size = split_size, random_state = seed)
x_train_set, x_test, y_train_set, y_test = model_selection.train_test_split(x_train, y_train, test_size = split_size, random_state = seed)

# Prepare data for TensorFlow model
num_classes = len(np.unique(y))
y_train_tf = to_categorical(y_train, num_classes)
y_test_tf = to_categorical(y_test, num_classes)
y_validation_tf = to_categorical(y_validation, num_classes)

# Create TensorFlow model
model = Sequential([
    Dense(128, activation='relu', input_shape=(x.shape[1],)),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile TensorFlow model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train TensorFlow model
model.fit(x_train, y_train_tf, epochs=5, batch_size=32, validation_split = split_size)

# Evaluate TensorFlow model
test_loss, test_acc = model.evaluate(x_test, y_test_tf)
print(f"TensorFlow model test accuracy: {test_acc}")

# Save TensorFlow model
model.save('tensorflow_model.h5')
tf_model = load_model('tensorflow_model.h5')

models = []
#models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('My TensorFlow Model', tf_model))

# Model Evaluation on Training Data Set
results = []
names = []
for name, model in models:
    if name == 'My TensorFlow Model':
        # Evaluate TensorFlow model on training data
        _, accuracy = model.evaluate(x_train, y_train_tf, verbose=0)
        cv_results = np.array([accuracy] * 5)  # Mock cross-validation results
    else:
        kfold = model_selection.KFold(n_splits = 5, shuffle = True, random_state = seed)
        cv_results = model_selection.cross_val_score(model, x_train, y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s Accuracy: %f (+/- %f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Training Data
cart = KNeighborsClassifier()
cart.fit(x_train_set, y_train_set)
filename = 'finalized_DT_model.sav'
joblib.dump(cart, filename)

# Test Data
print("\n CART results on 30% test set \n")
loaded_model = joblib.load(filename)
result = loaded_model.score(x_test, y_test)
print(result)
predictions_rfc = cart.predict(x_test)
print("\nCART accuracy test: \n")
print(accuracy_score(y_test, predictions_rfc))
print(confusion_matrix(y_test, predictions_rfc))
print(classification_report(y_test, predictions_rfc))

# Validation Data
print("\nCART results on final 30% validation \n")
newcart = KNeighborsClassifier()
newcart.fit(x_train_set, y_train_set)
newpredictions_rfc = newcart.predict(x_validation)
print("\nCART accuracy validation: \n")
print(accuracy_score(y_validation, newpredictions_rfc))
print(confusion_matrix(y_validation, newpredictions_rfc))
print(classification_report(y_validation, newpredictions_rfc))
