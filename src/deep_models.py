import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D,
                                         Flatten, LSTM, BatchNormalization)
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    KERAS_AVAILABLE = True
except Exception:
    try:
        from keras.models import Sequential
        from keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D,
                                  Flatten, LSTM, BatchNormalization)
        from keras.utils import to_categorical
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        KERAS_AVAILABLE = True
    except Exception:
        KERAS_AVAILABLE = False


def _ensure_keras():
    if not KERAS_AVAILABLE:
        raise RuntimeError('Keras/TensorFlow is not available')


def build_mlp(input_dim, hidden_units=(256, 128), dropout=0.5, n_classes=None):
    _ensure_keras()
    model = Sequential()
    model.add(Dense(hidden_units[0], input_shape=(input_dim,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    for units in hidden_units[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    if n_classes is None:
        model.add(Dense(1, activation='sigmoid'))
    else:
        if n_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(n_classes, activation='softmax'))
    return model


def build_cnn(input_shape, filters=(64, 128), kernel_size=3, pool_size=2, dropout=0.5, n_classes=None):
    _ensure_keras()
    model = Sequential()
    model.add(Conv1D(filters[0], kernel_size, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=pool_size))
    for f in filters[1:]:
        model.add(Conv1D(f, kernel_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dropout(dropout))
    if n_classes is None:
        model.add(Dense(1, activation='sigmoid'))
    else:
        if n_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(n_classes, activation='softmax'))
    return model


def build_lstm(input_shape, units=128, dropout=0.5, n_classes=None):
    _ensure_keras()
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    if n_classes is None:
        model.add(Dense(1, activation='sigmoid'))
    else:
        if n_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(n_classes, activation='softmax'))
    return model


def _prepare_targets(y, encoder=None):
    if encoder is None:
        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y)
    else:
        y_enc = encoder.transform(y)
    classes = np.unique(y_enc)
    n_classes = len(classes)
    if n_classes == 2:
        return y_enc, encoder, n_classes
    y_cat = to_categorical(y_enc, num_classes=n_classes)
    return y_cat, encoder, n_classes


def train_model(model, X, y, epochs=50, batch_size=32, validation_split=0.2, save_path=None, callbacks=None, optimizer='adam', loss=None):
    _ensure_keras()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(np.unique(y_enc))
    if n_classes == 2:
        y_train = y_enc
        if loss is None:
            loss = 'binary_crossentropy'
    else:
        y_train = to_categorical(y_enc, num_classes=n_classes)
        if loss is None:
            loss = 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    cb = [] if callbacks is None else list(callbacks)
    if save_path is not None:
        dirname = os.path.dirname(save_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        cb.append(ModelCheckpoint(save_path, save_best_only=True))
    cb.append(EarlyStopping(patience=8, restore_best_weights=True))
    history = model.fit(X, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=cb, verbose=2)
    return model, history, le


def evaluate_model(model, X, y, label_encoder=None):
    _ensure_keras()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if label_encoder is None:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
    else:
        le = label_encoder
        y_enc = le.transform(y)
    n_classes = len(np.unique(y_enc))
    y_pred_prob = model.predict(X)
    if n_classes == 2:
        if y_pred_prob.ndim > 1 and y_pred_prob.shape[1] > 1:
            probs = y_pred_prob[:, 1]
        else:
            probs = y_pred_prob.ravel()
        y_pred = (probs >= 0.5).astype(int)
        acc = np.mean(y_pred == y_enc)
        return {'accuracy': float(acc), 'y_pred': y_pred, 'y_prob': probs}
    else:
        if y_pred_prob.ndim == 1:
            probs = np.column_stack([1 - y_pred_prob, y_pred_prob])
        else:
            probs = y_pred_prob
        y_pred = np.argmax(probs, axis=1)
        acc = np.mean(y_pred == y_enc)
        return {'accuracy': float(acc), 'y_pred': y_pred, 'y_prob': probs}
