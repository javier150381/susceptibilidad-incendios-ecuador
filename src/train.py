"""
Experimento Realista: Entrenamiento con Ausencias Cercanas (Hard Negatives)
-------------------------------------------------------------------------
Este script evalúa los modelos usando el nuevo dataset con pseudo-ausencias
restringidas por buffer de 1-8 km cerca de los incendios reales.

Genera:
- Nuevas métricas realistas (Accuracy, Precision, Recall, F1, ROC-AUC)
- Nuevos pesos de atención e importancias Gini
- Gráfico comparativo 'comparativa_modelos_factores_realista.png'
- CSV 'comparativa_factores_realista.csv'

Ejecución:
    python src/train.py
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Rutas portables, configurables desde Colab o terminal.
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FILE_REALISTA = Path(os.environ.get('FIRE_DATA', ROOT / 'data' / 'dataset_entrenamiento_sierra_realista.csv'))
OUTPUT_DIR = Path(os.environ.get('FIRE_OUTPUT', ROOT / 'results'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FILE_CSV_FACTORES = OUTPUT_DIR / 'comparativa_factores_realista.csv'
FILE_PNG_FACTORES = OUTPUT_DIR / 'comparativa_modelos_factores_realista.png'

# Semillas
np.random.seed(42)
torch.manual_seed(42)

# Modelo de Autotensión
class FeatureSelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(FeatureSelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim, bias=False)
        self.key = nn.Linear(input_dim, input_dim, bias=False)
        self.value = nn.Linear(input_dim, input_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_scores = torch.matmul(q.unsqueeze(2), k.unsqueeze(1)) / np.sqrt(x.shape[1])
        attention_weights = self.softmax(attention_scores.mean(dim=2))
        attended_x = v * attention_weights
        return attended_x, attention_weights

class FireAttentionMLP(nn.Module):
    def __init__(self, input_dim):
        super(FireAttentionMLP, self).__init__()
        self.attention = FeatureSelfAttention(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attended_x, weights = self.attention(x)
        output = self.classifier(attended_x)
        return output, weights

def main():
    print("="*75)
    print("  ENTRENAMIENTO REALISTA CON AUSENCIAS CERCANAS (HARD NEGATIVE BUFFER)")
    print("="*75)

    if not os.path.exists(FILE_REALISTA):
        raise FileNotFoundError(f"Carga el CSV de entrenamiento en: {FILE_REALISTA}")

    df = pd.read_csv(FILE_REALISTA)
    features = ['elevacion', 'pendiente', 'orientacion', 'ndvi', 'evi', 'nbr', 'temp_c', 'precip_mm']
    
    nombres_es = {
        'temp_c': 'Temperatura',
        'precip_mm': 'Precipitación',
        'elevacion': 'Elevación',
        'ndvi': 'NDVI (Verdor)',
        'nbr': 'NBR (Humedad)',
        'evi': 'EVI (Biomasa)',
        'pendiente': 'Pendiente',
        'orientacion': 'Orientación'
    }

    # Limpiar nulos
    missing = set(features + ['clase']) - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {sorted(missing)}")
    df[features] = df[features].apply(pd.to_numeric, errors='raise')
    df = df.dropna(subset=features + ['clase']).copy()
    if not np.isfinite(df[features].to_numpy()).all():
        raise ValueError('Las variables contienen valores infinitos.')
    if set(df['clase'].unique()) != {0, 1}:
        raise ValueError('clase debe contener ambas clases: 0 y 1.')
    if df['clase'].value_counts().min() < 5:
        raise ValueError('Se necesitan al menos 5 filas por clase.')
    X = df[features].values
    y = df['clase'].values

    print(f"[INFO] Dataset cargado: {len(df)} muestras.")
    print(f"       - Presencias (Incendios): {sum(y == 1)}")
    print(f"       - Ausencias Difíciles:    {sum(y == 0)}")

    # 1. DIVISIÓN ALEATORIA ESTRATIFICADA (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Escalado
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- MODELO 1: RANDOM FOREST ---
    print("\n[1/2] Entrenando Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred)
    rf_rec = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_proba)
    rf_cm = confusion_matrix(y_test, rf_pred)

    # --- MODELO 2: RED DE AUTOTENSIÓN (PYTORCH) ---
    print("[2/2] Entrenando Red Neuronal de Autotensión (PyTorch)...")
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_train).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    X_test_t = torch.FloatTensor(X_test_s)

    model = FireAttentionMLP(input_dim=len(features))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)

    model.train()
    for epoch in range(int(os.environ.get('FIRE_EPOCHS', '50'))):
        for bx, by in train_loader:
            optimizer.zero_grad()
            out, _ = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        nn_proba_t, nn_weights_t = model(X_test_t)
        nn_proba = nn_proba_t.numpy().flatten()
        nn_pred = (nn_proba >= 0.5).astype(int)
        
        nn_attn_mean = nn_weights_t.mean(dim=0).numpy()
        nn_importances = nn_attn_mean / np.sum(nn_attn_mean)

    nn_acc = accuracy_score(y_test, nn_pred)
    nn_prec = precision_score(y_test, nn_pred)
    nn_rec = recall_score(y_test, nn_pred)
    nn_f1 = f1_score(y_test, nn_pred)
    nn_auc = roc_auc_score(y_test, nn_proba)
    nn_cm = confusion_matrix(y_test, nn_pred)

    # IMPRIMIR RESULTADOS REALISTAS
    print("\n" + "="*65)
    print("  RESULTADOS REALISTAS (CON AUSENCIAS DIFICILES POR BUFFER 1-8KM)")
    print("="*65)
    print(f"Random Forest  -> Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%) | ROC-AUC: {rf_auc:.4f}")
    print(f"Attention Net  -> Accuracy: {nn_acc:.4f} ({nn_acc*100:.2f}%) | ROC-AUC: {nn_auc:.4f}")
    
    print("\n[Matriz de Confusión Random Forest]:")
    print(f"  TN: {rf_cm[0][0]}, FP: {rf_cm[0][1]} | FN: {rf_cm[1][0]}, TP: {rf_cm[1][1]}")

    print("\n[Matriz de Confusión Red de Atención]:")
    print(f"  TN: {nn_cm[0][0]}, FP: {nn_cm[0][1]} | FN: {nn_cm[1][0]}, TP: {nn_cm[1][1]}")

    pd.DataFrame([
        dict(modelo='Random Forest', accuracy=rf_acc, precision=rf_prec, recall=rf_rec, f1=rf_f1, roc_auc=rf_auc),
        dict(modelo='Attention MLP', accuracy=nn_acc, precision=nn_prec, recall=nn_rec, f1=nn_f1, roc_auc=nn_auc),
    ]).to_csv(OUTPUT_DIR / 'metricas.csv', index=False)
    import json
    import hashlib
    import importlib.metadata
    metadata = dict(seed=42, test_size=0.2, epochs=int(os.environ.get('FIRE_EPOCHS', '50')),
                    rows=len(df), features=features, split='aleatorio estratificado',
                    data_sha256=hashlib.sha256(FILE_REALISTA.read_bytes()).hexdigest(),
                    versions={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'torch']})
    (OUTPUT_DIR / 'ejecucion.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    # IMPORTANCIA DE VARIABLES REALISTA
    rf_importances = rf.feature_importances_
    df_comp = pd.DataFrame({
        'Variable': [nombres_es[f] for f in features],
        'Random Forest (Gini)': rf_importances * 100,
        'Attention Net (Pesos)': nn_importances * 100
    }).sort_values(by='Attention Net (Pesos)', ascending=False).reset_index(drop=True)

    print("\n" + "="*65)
    print("  IMPORTANCIA COMPARATIVA DE LAS 8 VARIABLES (%)")
    print("="*65)
    print(df_comp.round(2).to_string(index=False))
    df_comp.to_csv(FILE_CSV_FACTORES, index=False)

    # GRAFICAR IMPORTANCIA DE VARIABLES REALISTA
    plt.figure(figsize=(12, 7), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    y_pos = np.arange(len(features))
    width = 0.35
    df_plot = df_comp.sort_values(by='Attention Net (Pesos)', ascending=True).reset_index(drop=True)
    
    plt.barh(y_pos - width/2, df_plot['Random Forest (Gini)'], width, label='Random Forest (Gini)', color='#d95f0e')
    plt.barh(y_pos + width/2, df_plot['Attention Net (Pesos)'], width, label='Red de Atención (Pesos)', color='#2b8cbe')
    
    plt.yticks(y_pos, df_plot['Variable'], fontsize=11, fontweight='semibold')
    plt.xlabel('Importancia Relativa / Peso del Factor (%)', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('Factores de Incendios en la Sierra Ecuatoriana (Dataset Realista)', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right', fontsize=11)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    
    plt.tight_layout()
    plt.savefig(FILE_PNG_FACTORES, bbox_inches='tight')
    plt.close()
    
    print(f"\n[OK] Gráfico comparativo guardado en: {FILE_PNG_FACTORES}")
    print("="*75)

if __name__ == "__main__":
    main()
