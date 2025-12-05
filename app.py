import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from pathlib import Path

st.set_page_config(page_title="Предсказание цен авто", page_icon="🚗", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent 
MODEL_PATH = MODEL_DIR / "models" / "car_price_model.pkl"

#ЗАГРУЗКА
@st.cache_resource
def load_artifacts():
    try:
        with open(MODEL_PATH, 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except FileNotFoundError:
        return None

#ПРЕПРОЦЕССИНГ
def prepare_features(df_input, medians, feature_names):
    df = df_input.copy()
    
    for col, pattern in [('mileage', r' kmpl| km/kg'), 
                         ('engine', r' CC'), 
                         ('max_power', r' bhp')]:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(pattern, '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'torque' in df.columns and df['torque'].dtype == 'object':
        torque_clean = df['torque'].astype(str).str.lower()
        torque_clean = torque_clean.str.replace(' at ', ' ', regex=False).str.replace('@', ' ', regex=False)
        torque_clean = torque_clean.str.replace(r'[\(\),]', '', regex=True).str.strip()
        
        df['torque'] = torque_clean.str.extract(r'^(\d+\.?\d*)', expand=False).astype(float)
        
        extracted_rpm = torque_clean.str.extract(r'(\d+)-?\d*\s*rpm', expand=False).astype(float)
        df['max_torque_rpm'] = extracted_rpm

    for col, val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
        else:
            df[col] = val
            
    if 'engine' in df.columns: df['engine'] = df['engine'].astype(int)
    if 'seats' in df.columns: df['seats'] = df['seats'].astype(int)
    
    if 'name' in df.columns:
        df['brand'] = df['name'].str.split().str[0].str.lower().str.strip()
    
    if 'year' in df.columns:
        df['age'] = 2020 - df['year']
        if 'km_driven' in df.columns:
            df['km_per_year'] = df['km_driven'] / (df['age'] + 1)
        if 'max_power' in df.columns and 'engine' in df.columns:
             df['power_per_liter'] = df['max_power'] / (df['engine'] / 1000 + 1e-3)

    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)


    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
            
    return df[feature_names]

# ИНИЦИАЛИЗАЦИЯ
artifacts = load_artifacts()

if artifacts is None:
    st.error(f"Файл модели не найден по пути: {MODEL_PATH}")
    st.stop()

MODEL = artifacts['model']
MEDIANS = artifacts['medians']
DF_SAMPLE = artifacts['df_sample']
CAT_FEATURES = artifacts['cat_features']
NUM_FEATURES = artifacts['num_features']
ALL_FEATURES = MODEL.feature_names_

# ИНТЕРФЕЙС
st.title("Предсказание стоимости автомобиля")

tab1, tab2, tab3 = st.tabs(["Одиночное предсказание", "Загрузка CSV", "Аналитика"])

#РУЧНОЙ ВВОД
with tab1:
    st.header("Параметры автомобиля")

    available_brands = sorted(DF_SAMPLE['name'].str.split().str[0].str.lower().unique())
    available_brands = [b.capitalize() for b in available_brands]

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            brand_val = st.selectbox("Бренд", available_brands)
            
            min_year = int(DF_SAMPLE['year'].min())
            year_val = st.slider("Год выпуска", min_year, 2024, 2017)
            
        with c2:
            seller_val = st.selectbox("Продавец", sorted(DF_SAMPLE['seller_type'].unique()))
            owner_val = st.selectbox("Владелец", sorted(DF_SAMPLE['owner'].unique()))
            
        with c3:
            km_val = st.number_input("Пробег (км)", min_value=0, value=50000, step=1000)
            trans_val = st.radio("Коробка", sorted(DF_SAMPLE['transmission'].unique()), horizontal=True)
            fuel_val = st.selectbox("Топливо", sorted(DF_SAMPLE['fuel'].unique()))

        st.markdown("---")
        st.caption("Технические характеристики")
        
        c4, c5, c6, c7 = st.columns(4)
        
        with c4:
            engine_val = st.number_input("Объем (CC)", min_value=500, max_value=5000, value=1248)
        with c5:
            power_val = st.number_input("Мощность (bhp)", min_value=30.0, max_value=500.0, value=80.0)
        with c6:
            mileage_val = st.number_input("Расход (kmpl)", min_value=0.0, max_value=50.0, value=20.0, step=0.1)
        with c7:
            torque_val = st.number_input("Крутящий момент (Nm)", min_value=10.0, max_value=1000.0, value=190.0)
            seats_val = st.slider("Мест", 2, 14, 5)

        submitted = st.form_submit_button("Рассчитать стоимость", use_container_width=True)

    if submitted:
        input_data = {
            'name': [brand_val],
            'year': [year_val],
            'km_driven': [km_val],
            'fuel': [fuel_val],
            'seller_type': [seller_val],
            'transmission': [trans_val],
            'owner': [owner_val],
            'mileage': [mileage_val],
            'engine': [engine_val],
            'max_power': [power_val],
            'seats': [seats_val],
            'torque': [torque_val] 
        }
        
        input_df = pd.DataFrame(input_data)
        
        if 'max_torque_rpm' in MEDIANS:
            input_df['max_torque_rpm'] = MEDIANS['max_torque_rpm']
        else:
             input_df['max_torque_rpm'] = 2500.0

        try:
            X_processed = prepare_features(input_df, MEDIANS, ALL_FEATURES)
            log_pred = MODEL.predict(X_processed)
            price = np.expm1(log_pred)[0]
            
            st.success(f"Оценочная стоимость: **{price:,.0f}**")
            
            with st.expander("Показать параметры, отправленные модели"):
                st.dataframe(X_processed)
                
        except Exception as e:
            st.error(f"Ошибка при расчете: {e}")


#ЗАГРУЗКА CSV
with tab2:
    st.header("Пакетная загрузка")
    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Исходные данные:")
        st.dataframe(df.head())

        if st.button("Предсказать для всех"):
            try:
                X_processed = prepare_features(df, MEDIANS, ALL_FEATURES)
                
                log_preds = MODEL.predict(X_processed)
                prices = np.expm1(log_preds)
                
                df['Predicted_Price'] = prices
                
                st.subheader("Результаты")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Всего авто", len(df))
                col2.metric("Средняя цена", f"{df['Predicted_Price'].mean():,.0f}")
                col3.metric("Общая сумма", f"{df['Predicted_Price'].sum():,.0f}")
                
                st.dataframe(df[['name', 'year', 'Predicted_Price']].head())
                
                st.subheader("Анализ загруженных данных")
                fig1 = px.histogram(df, x='Predicted_Price', nbins=30, 
                                    title="Распределение предсказанных цен",
                                    color_discrete_sequence=['#3498db'])
                st.plotly_chart(fig1, use_container_width=True)
                
            except Exception as e:
                st.error(f"Ошибка обработки: {e}")

#АНАЛИТИКА

with tab3:
    st.header("EDA: Обучающая выборка")
    st.info("Статистика построена на сэмпле обучающих данных.")
    
    col1, col2 = st.columns(2)
    with col1:
        df_filtered = DF_SAMPLE[DF_SAMPLE['selling_price'] < DF_SAMPLE['selling_price'].quantile(0.99)]

        fig_eda1 = px.histogram(
            df_filtered, 
            x="selling_price", 
            nbins=30, 
            title="Распределение цен (без 1% самых дорогих авто)",
            color_discrete_sequence=['#2ecc71']
        )
        
        st.plotly_chart(fig_eda1, use_container_width=True)

    with col2:
        # Топ брендов
        if 'name' in DF_SAMPLE.columns:
            sample_brand = DF_SAMPLE['name'].str.split().str[0]
            top_brands = sample_brand.value_counts().head(10)
            fig_eda2 = px.bar(x=top_brands.index, y=top_brands.values, 
                              title="Топ-10 брендов в выборке", labels={'x':'Бренд', 'y':'Кол-во'})
            st.plotly_chart(fig_eda2, use_container_width=True)

    st.subheader("Матрица корреляций")
    numeric_df = DF_SAMPLE.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", 
                             color_continuous_scale='RdBu_r', title="Heatmap корреляций")
        st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()
    st.subheader("Веса модели")
    
    try:
        feat_imp = pd.DataFrame({
            'feature': MODEL.feature_names_,
            'importance': MODEL.get_feature_importance()
        }).sort_values('importance', ascending=False).head(15)

        fig_imp = px.bar(feat_imp, x='importance', y='feature', orientation='h', 
                         title="Топ факторов, влияющих на цену", 
                         color='importance', color_continuous_scale='Viridis')
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:

        st.warning("Не удалось извлечь важность признаков из модели.")
