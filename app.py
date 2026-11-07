import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

app_dir = Path(__file__).parent
st.set_page_config(page_title='Подборка вин', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(app_dir / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(app_dir / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
test = load_test()
config = json.loads((app_dir / 'config.json').read_text())
features = model.feature_names_in_.tolist()

names = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
    'is_red': 'Красное вино'
}

st.title('Подборка вин')
t = st.sidebar.slider('Порог отбора', 0.0, 1.0, config['threshold'], step=0.001, format='%g')
one, report, upload, weights = st.tabs(['Одно вино', 'Модельные метрики', 'Пакетная оценка', 'Что выучила модель'])

with one:
    wine = {}
    cols = st.columns(2)
    for i, name in enumerate(features):
        with cols[i % 2]:
            if name == 'is_red':
                wine[name] = int(st.toggle('Красное вино', help='Выключено для белого вина'))
            else:
                values = test[name]
                wine[name] = st.number_input(names[name], float(values.min()),
                                             float(values.max()), float(values.median()), format='%g')

    p = model.predict_proba(pd.DataFrame([wine], columns=features))[0, 1]
    st.metric('Вероятность хорошего вина', p, format='percent')
    if p >= t:
        st.success('Включить в подборку')
    else:
        st.info('Пропустить')

with report:
    y = test['target']
    p_test = test_proba(model, test[features])
    pred = p_test >= t
    cm = metrics.confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = metrics.precision_score(y, pred, zero_division=1)
    recall = metrics.recall_score(y, pred)
    if pred.any():
        precision_value = precision
    else:
        precision_value = None
    a, b, c = st.columns(3)
    a.metric('Precision', precision_value, format='percent')
    b.metric('Recall', recall, format='percent')
    c.metric('F1', metrics.f1_score(y, pred), format='percent')

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    metrics.ConfusionMatrixDisplay(cm).plot(ax=ax[0], cmap='Purples', colorbar=False)
    ax[0].set(title='Матрица ошибок', xlabel='Прогноз', ylabel='Факт')
    metrics.RocCurveDisplay.from_predictions(y, p_test, ax=ax[1])
    ax[1].scatter(fp / (fp + tn), recall, color='tab:orange', zorder=3)
    ax[1].set_title('ROC')
    metrics.PrecisionRecallDisplay.from_predictions(y, p_test, ax=ax[2])
    ax[2].scatter(recall, precision, color='tab:orange', zorder=3)
    ax[2].set_title('Precision-Recall')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write('При высоком пороге подборка меньше, и в ней обычно выше доля хороших вин. '
             'При низком пороге в подборку попадает больше хороших вин, но и больше посредственных')

with upload:
    file = st.file_uploader('CSV с признаками вин', type='csv')

    if file is not None:
        try:
            batch = pd.read_csv(file)
            missing = [name for name in features if name not in batch]

            if missing:
                st.error(f'В файле не хватает столбцов: {", ".join(missing)}')
            else:
                p_batch = model.predict_proba(batch[features])[:, 1]
                batch['Вероятность'] = p_batch
                batch['Решение'] = np.where(p_batch >= t, 'В подборку', 'Пропустить')

                st.dataframe(batch, hide_index=True)
                st.download_button(
                    'Скачать результаты',
                    batch.to_csv(index=False).encode('utf-8-sig'),
                    'wine_predictions.csv',
                    'text/csv'
                )
        except ValueError:
            st.error('Проверьте формат CSV и заполните все признаки числами')

with weights:
    lr = model.named_steps['logisticregression']
    coef = pd.Series(lr.coef_[0], index=features).rename(index=names).sort_values()
    fig, ax = plt.subplots()
    coef.plot.barh(ax=ax)
    ax.axvline(0, color='gray')
    ax.set_xlabel('Коэффициент после масштабирования')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write(f'Сильнее всего за хорошее вино голосует признак «{coef.idxmax()}», '
             f'а против сильнее всего голосует «{coef.idxmin()}»')
