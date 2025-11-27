import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Prediksi Promosi Karyawan PT XYZ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📊 Sistem Prediksi Promosi Karyawan</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">PT TOTO Indonesia - Analisis Machine Learning untuk Keputusan Promosi</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=PT+TOTO", use_container_width=True)
    
    st.markdown("### 📊 Cara Menggunakan")
    st.markdown("""
    1. Upload file **Data_Promosi_2025.xlsx**
    2. Sistem otomatis akan memproses data
    3. Lihat hasil prediksi dan analisis
    4. Download hasil prediksi
    """)
    
    st.markdown("---")
    st.markdown("**Developed by:** Elisabet Lumban Tobing")
    st.markdown("**Version:** 1.0")

# Fungsi preprocessing data
def preprocess_data(df):
    """Preprocessing data untuk modeling"""
    
    # Hapus baris yang tidak relevan (baris header duplikat)
    df = df[df['NIK'].notna()].copy()
    df = df[df['NIK'] != 'NIK'].copy()
    
    # Konversi NIK ke numeric
    df['NIK'] = pd.to_numeric(df['NIK'], errors='coerce')
    
    # Drop baris dengan NIK null
    df = df[df['NIK'].notna()].copy()
    
    # Target variable: KETERANGAN (C = Cukup, K = Kurang, B = Baik)
    # Kita buat target binary: Layak Promosi (B, C) vs Tidak Layak (K)
    df['Target_Promosi'] = df['KETERANGAN'].apply(lambda x: 1 if x in ['B', 'C'] else 0)
    
    # Feature engineering
    features = []
    
    # 1. MASA JABATAN
    if 'MASA JABATAN' in df.columns:
        df['MASA JABATAN'] = pd.to_numeric(df['MASA JABATAN'], errors='coerce')
        df['MASA JABATAN'].fillna(df['MASA JABATAN'].median(), inplace=True)
        features.append('MASA JABATAN')
    
    # 2. AVG SCORE
    if 'AVG SCORE' in df.columns:
        df['AVG SCORE'] = pd.to_numeric(df['AVG SCORE'], errors='coerce')
        df['AVG SCORE'].fillna(df['AVG SCORE'].median(), inplace=True)
        features.append('AVG SCORE')
    
    # 3. LEVEL JABATAN (encode)
    if 'LEVEL JABATAN' in df.columns:
        le_jabatan = LabelEncoder()
        df['LEVEL JABATAN_encoded'] = le_jabatan.fit_transform(df['LEVEL JABATAN'].fillna('Unknown'))
        features.append('LEVEL JABATAN_encoded')
    
    # 4. Extract numerical features dari kolom tahun
    year_columns = [col for col in df.columns if isinstance(col, int) and col >= 2020]
    
    for year_col in year_columns:
        # Ambil kolom performance dari setiap tahun
        try:
            # Cari kolom yang berisi nilai numeric untuk tahun tersebut
            year_idx = df.columns.get_loc(year_col)
            # Ambil beberapa kolom setelahnya yang mungkin berisi performance data
            for i in range(1, 6):
                if year_idx + i < len(df.columns):
                    col_name = df.columns[year_idx + i]
                    if df[col_name].dtype in ['float64', 'int64'] or pd.to_numeric(df[col_name], errors='coerce').notna().sum() > 10:
                        feature_name = f'{year_col}_feature_{i}'
                        df[feature_name] = pd.to_numeric(df[col_name], errors='coerce')
                        df[feature_name].fillna(df[feature_name].median(), inplace=True)
                        features.append(feature_name)
        except:
            pass
    
    # Pastikan minimal ada 3 features
    if len(features) < 3:
        st.error("❌ Data tidak memiliki cukup fitur untuk modeling. Pastikan file Excel memiliki kolom yang benar.")
        return None, None, None
    
    # Prepare X and y
    X = df[features].copy()
    y = df['Target_Promosi'].copy()
    
    # Handle any remaining NaN
    X.fillna(X.median(), inplace=True)
    
    return X, y, df

# Fungsi training models
def train_models(X, y):
    """Training 3 model: RF, XGBoost, KNN"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    results = {}
    
    # 1. Random Forest (Terbaik berdasarkan analisis)
    with st.spinner('Training Random Forest...'):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        results['Random Forest'] = {
            'model': rf_model,
            'predictions': rf_pred,
            'probabilities': rf_pred_proba,
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred, zero_division=0),
            'recall': recall_score(y_test, rf_pred, zero_division=0),
            'f1': f1_score(y_test, rf_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, rf_pred),
            'y_test': y_test
        }
    
    # 2. XGBoost
    with st.spinner('Training XGBoost...'):
        xgb_model = XGBClassifier(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        results['XGBoost'] = {
            'model': xgb_model,
            'predictions': xgb_pred,
            'probabilities': xgb_pred_proba,
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred, zero_division=0),
            'recall': recall_score(y_test, xgb_pred, zero_division=0),
            'f1': f1_score(y_test, xgb_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, xgb_pred),
            'y_test': y_test
        }
    
    # 3. K-Nearest Neighbors
    with st.spinner('Training K-Nearest Neighbors...'):
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        knn_pred = knn_model.predict(X_test)
        knn_pred_proba = knn_model.predict_proba(X_test)[:, 1]
        
        results['KNN'] = {
            'model': knn_model,
            'predictions': knn_pred,
            'probabilities': knn_pred_proba,
            'accuracy': accuracy_score(y_test, knn_pred),
            'precision': precision_score(y_test, knn_pred, zero_division=0),
            'recall': recall_score(y_test, knn_pred, zero_division=0),
            'f1': f1_score(y_test, knn_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, knn_pred),
            'y_test': y_test
        }
    
    return results

# Fungsi prediksi semua data
def predict_all_employees(models, X, df):
    """Prediksi untuk semua karyawan"""
    
    predictions = []
    
    for idx, row in df.iterrows():
        employee_data = X.loc[idx:idx]
        
        # Prediksi dari setiap model
        rf_prob = models['Random Forest']['model'].predict_proba(employee_data)[0][1]
        xgb_prob = models['XGBoost']['model'].predict_proba(employee_data)[0][1]
        knn_prob = models['KNN']['model'].predict_proba(employee_data)[0][1]
        
        # Average probability (ensemble)
        avg_prob = (rf_prob + xgb_prob + knn_prob) / 3
        
        # Prediksi final (voting)
        rf_pred = models['Random Forest']['model'].predict(employee_data)[0]
        xgb_pred = models['XGBoost']['model'].predict(employee_data)[0]
        knn_pred = models['KNN']['model'].predict(employee_data)[0]
        
        final_pred = 1 if (rf_pred + xgb_pred + knn_pred) >= 2 else 0
        
        predictions.append({
            'NIK': int(df.loc[idx, 'NIK']),
            'LEVEL_JABATAN': df.loc[idx, 'LEVEL JABATAN'],
            'MASA_JABATAN': df.loc[idx, 'MASA JABATAN'],
            'AVG_SCORE': df.loc[idx, 'AVG SCORE'],
            'KETERANGAN_ASLI': df.loc[idx, 'KETERANGAN'],
            'RF_Probability': round(rf_prob * 100, 2),
            'XGB_Probability': round(xgb_prob * 100, 2),
            'KNN_Probability': round(knn_prob * 100, 2),
            'Avg_Probability': round(avg_prob * 100, 2),
            'Prediksi_Promosi': 'Layak' if final_pred == 1 else 'Tidak Layak',
            'Ranking_Score': round(avg_prob * 100, 2)
        })
    
    return pd.DataFrame(predictions)

# Fungsi visualisasi
def plot_model_comparison(results):
    """Plot perbandingan performa model"""
    
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig = go.Figure()
    
    for metric in metrics:
        values = [results[model][metric] * 100 for model in models]
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=models,
            y=values,
            text=[f'{v:.2f}%' for v in values],
            textposition='auto',
        ))
    
    fig.update_layout(
        title='📊 Perbandingan Performa Model',
        xaxis_title='Model',
        yaxis_title='Score (%)',
        barmode='group',
        height=500,
        showlegend=True
    )
    
    return fig

def plot_confusion_matrices(results):
    """Plot confusion matrix untuk semua model"""
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=list(results.keys()),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]]
    )
    
    for i, (model_name, model_data) in enumerate(results.items(), 1):
        cm = model_data['confusion_matrix']
        
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Tidak Layak', 'Layak'],
                y=['Tidak Layak', 'Layak'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorscale='Blues',
                showscale=False
            ),
            row=1, col=i
        )
    
    fig.update_layout(
        title_text='Confusion Matrix - Semua Model',
        height=400,
    )
    
    return fig

def plot_ranking_distribution(df_predictions):
    """Plot distribusi ranking score"""
    
    fig = px.histogram(
        df_predictions,
        x='Ranking_Score',
        nbins=20,
        color='Prediksi_Promosi',
        title='📈 Distribusi Ranking Score Karyawan',
        labels={'Ranking_Score': 'Ranking Score (%)', 'count': 'Jumlah Karyawan'},
        color_discrete_map={'Layak': '#2ecc71', 'Tidak Layak': '#e74c3c'}
    )
    
    fig.update_layout(height=400)
    
    return fig

# Main app
def main():
    
    # File uploader
    st.markdown("### 📁 Upload Data Promosi")
    uploaded_file = st.file_uploader(
        "Pilih file Excel (Data_Promosi_2025.xlsx)",
        type=['xlsx', 'xls'],
        help="Upload file Excel yang berisi data promosi karyawan"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            with st.spinner('⏳ Memuat data...'):
                df_raw = pd.read_excel(uploaded_file, skiprows=2)
                st.success(f'✅ Data berhasil dimuat: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom')
            
            # Show raw data preview
            with st.expander("Preview Data Asli"):
                st.dataframe(df_raw.head(10), use_container_width=True)
            
            # Preprocessing
            with st.spinner('🔄 Preprocessing data...'):
                X, y, df_clean = preprocess_data(df_raw)
            
            if X is not None:
                st.success(f'✅ Preprocessing selesai: {X.shape[0]} sampel, {X.shape[1]} fitur')
                
                # Show feature info
                with st.expander("📊 Informasi Fitur"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Fitur yang Digunakan:**")
                        for feat in X.columns:
                            st.markdown(f"- {feat}")
                    with col2:
                        st.markdown("**Distribusi Target:**")
                        target_counts = y.value_counts()
                        st.markdown(f"- Layak Promosi: {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(y)*100:.1f}%)")
                        st.markdown(f"- Tidak Layak: {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(y)*100:.1f}%)")
                
                # Training models
                st.markdown("---")
                st.markdown("###Training Model Machine Learning")
                
                if st.button('Mulai Training & Prediksi', type='primary', use_container_width=True):
                    
                    # Train models
                    results = train_models(X, y)
                    st.success('✅ Training selesai untuk 3 model!')
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Hasil Evaluasi Model")
                    
                    # Metrics cards
                    cols = st.columns(3)
                    for i, (model_name, model_data) in enumerate(results.items()):
                        with cols[i]:
                            st.markdown(f"#### {model_name}")
                            st.metric("Accuracy", f"{model_data['accuracy']*100:.2f}%")
                            st.metric("Precision", f"{model_data['precision']*100:.2f}%")
                            st.metric("Recall", f"{model_data['recall']*100:.2f}%")
                            st.metric("F1-Score", f"{model_data['f1']*100:.2f}%")
                    
                    # Visualizations
                    st.markdown("---")
                    st.markdown("### 📈 Visualisasi Performa Model")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        fig_comparison = plot_model_comparison(results)
                        st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    with col2:
                        fig_cm = plot_confusion_matrices(results)
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Predictions
                    st.markdown("---")
                    st.markdown("###Hasil Prediksi Promosi Karyawan")
                    
                    with st.spinner('Memprediksi semua karyawan...'):
                        df_predictions = predict_all_employees(results, X, df_clean)
                    
                    st.success(f'✅ Prediksi selesai untuk {len(df_predictions)} karyawan!')
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_layak = (df_predictions['Prediksi_Promosi'] == 'Layak').sum()
                        st.metric("Total Layak Promosi", total_layak, 
                                 delta=f"{total_layak/len(df_predictions)*100:.1f}%")
                    
                    with col2:
                        total_tidak = (df_predictions['Prediksi_Promosi'] == 'Tidak Layak').sum()
                        st.metric("Total Tidak Layak", total_tidak,
                                 delta=f"{total_tidak/len(df_predictions)*100:.1f}%")
                    
                    with col3:
                        avg_score = df_predictions['Ranking_Score'].mean()
                        st.metric("Rata-rata Score", f"{avg_score:.2f}%")
                    
                    with col4:
                        max_score = df_predictions['Ranking_Score'].max()
                        st.metric("Score Tertinggi", f"{max_score:.2f}%")
                    
                    # Distribution plot
                    fig_dist = plot_ranking_distribution(df_predictions)
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Top performers
                    st.markdown("### 🌟 Top 10 Karyawan dengan Ranking Tertinggi")
                    top_10 = df_predictions.nlargest(10, 'Ranking_Score')
                    st.dataframe(
                        top_10[['NIK', 'LEVEL_JABATAN', 'MASA_JABATAN', 'AVG_SCORE', 
                               'Ranking_Score', 'Prediksi_Promosi']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Full predictions table
                    st.markdown("### 📋 Tabel Lengkap Prediksi")
                    
                    # Filter options
                    col1, col2 = st.columns(2)
                    with col1:
                        filter_prediksi = st.selectbox(
                            "Filter berdasarkan Prediksi",
                            ['Semua', 'Layak', 'Tidak Layak']
                        )
                    with col2:
                        sort_by = st.selectbox(
                            "Urutkan berdasarkan",
                            ['Ranking_Score', 'NIK', 'AVG_SCORE', 'MASA_JABATAN']
                        )
                    
                    # Apply filters
                    df_filtered = df_predictions.copy()
                    if filter_prediksi != 'Semua':
                        df_filtered = df_filtered[df_filtered['Prediksi_Promosi'] == filter_prediksi]
                    
                    df_filtered = df_filtered.sort_values(sort_by, ascending=False)
                    
                    st.dataframe(df_filtered, use_container_width=True, hide_index=True)
                    
                    # Download button
                    st.markdown("### 💾 Download Hasil Prediksi")
                    
                    # Prepare download
                    csv = df_predictions.to_csv(index=False).encode('utf-8')
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        st.download_button(
                            label="📥 Download Hasil Prediksi (CSV)",
                            data=csv,
                            file_name='hasil_prediksi_promosi.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                    
                    # Model insights
                    st.markdown("---")
                    st.markdown("### 💡 Insight & Rekomendasi")
                    
                    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
                    
                    st.info(f"""
                    **Model Terbaik:** {best_model[0]} dengan akurasi {best_model[1]['accuracy']*100:.2f}%
                    
                    **Rekomendasi:**
                    - Dari {len(df_predictions)} karyawan, {total_layak} karyawan ({total_layak/len(df_predictions)*100:.1f}%) diprediksi layak untuk promosi
                    - Fokus pada karyawan dengan Ranking Score > 70% untuk prioritas promosi
                    - Pertimbangkan faktor lain seperti kebutuhan organisasi dan budget
                    """)
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.error("Pastikan file Excel memiliki format yang benar dan kolom yang sesuai.")
    
    else:
        # Instructions
        st.info("""
        ### 
        Untuk memulai analisis:
        1. Klik tombol **Browse files** di atas
        2. Upload file **Data_Promosi_2025.xlsx**
        3. Sistem akan otomatis memproses dan menampilkan hasil
        
        **Format Data yang Dibutuhkan:**
        - File Excel (.xlsx atau .xls)
        - Minimal memiliki kolom: NIK, LEVEL JABATAN, MASA JABATAN, AVG SCORE, KETERANGAN
        """)
        
        # Sample preview
        st.markdown("### 📖 Contoh Format Data")
        sample_data = pd.DataFrame({
            'NIK': [2247, 2869, 2862],
            'LEVEL JABATAN': ['Asst. Manager', 'Group Leader', 'Group Leader'],
            'MASA JABATAN': [15, 12, 10],
            'AVG SCORE': [1.5, 1.36, 1.44],
            'KETERANGAN': ['B', 'C', 'C']
        })
        st.dataframe(sample_data, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
