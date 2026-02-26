# =================================================================================================
# Trabalho Computacional 3 - Reconhecimento de Padrões
# Base: Wall-Following Robot Navigation Data (24 atributos, 4 classes)
#
# Objetivo:
#   - Avaliar o desempenho de classificadores supervisionados (CQG e DMP)
#   - Analisar o impacto da redução de dimensionalidade via PCA
#
# Implementa e atende:
#
#   1.1 Identificação do problema:
#       - Número de classes
#       - Dimensão do vetor de atributos
#       - Número de instâncias por classe
#
#   1.2 Verificação das matrizes de covariância:
#       - Checagem de invertibilidade por classe (via rank da matriz)
#
#   1.3 Classificadores (sem PCA):
#       - CQG (Classificador Quadrático Gaussiano / QDA implementado "na mão")
#       - DMP (Distância Mínima ao Protótipo)
#           * Protótipos obtidos por K-means aplicado separadamente em cada classe
#           * Seleção automática do número "ótimo" de protótipos por classe via Silhouette
#
#   Nr = 100 rodadas independentes:
#       - Acurácia global (média e desvio padrão)
#       - Acurácia por classe (média e desvio padrão)
#
#   2.1 PCA:
#       - Geração do gráfico da variância explicada acumulada VE(q)
#
#   2.2 Classificadores com PCA:
#       - Reaplicação de CQG e DMP após PCA
#       - Seleção de q "adequado" visando reduzir dimensionalidade sem piorar desempenho médio
#
# Observações:
#   - Os dados são padronizados (StandardScaler) antes dos classificadores e do PCA
#   - A divisão treino/teste é estratificada (stratify=y) para preservar proporções de classe
# ===================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ===================================================
# Carregamento da Base de Dados
# ===================================================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00194/sensor_readings_24.data"
column_names = [f"s{i}" for i in range(1, 25)] + ["class"]
data = pd.read_csv(url, names=column_names)

X = data.drop("class", axis=1).to_numpy(dtype=float)
y = data["class"].to_numpy()
classes = np.unique(y)


# ====================================================
# 1.1 Identificação do problema
# ====================================================
print("=== 1.1 Identificação do problema ===")
print(f"Número de classes: {len(classes)}")
print(f"Dimensão do vetor de atributos: {X.shape[1]}")
for c in classes:
    print(f"Instâncias da classe {c}: {(y == c).sum()}")


# =======================================================
# 1.2 Invertibilidade das covariâncias por classe
# (mais estável checar posto/rank do que determinante)
# =======================================================
print("\n=== 1.2 Invertibilidade das matrizes de covariância ===")
d = X.shape[1]
for c in classes:
    Xc = X[y == c]
    cov = np.cov(Xc.T, bias=False)
    rank = np.linalg.matrix_rank(cov)
    inv = (rank == d)
    print(f"Classe {c}: rank={rank}/{d} -> {'Invertível' if inv else 'Não invertível (singular)'}")


# ============================================================
# 1.3 (1) CQG = Classificador Quadrático Gaussiano (QDA "na mão")
# ============================================================
def train_cqg(Xtr, ytr, reg=1e-6):
    """
    Estima pi_c, mu_c, Sigma_c (com regularização diagonal para evitar singularidade numérica).
    """
    model = {}
    n = Xtr.shape[0]
    for c in np.unique(ytr):
        Xc = Xtr[ytr == c]
        pi = Xc.shape[0] / n
        mu = Xc.mean(axis=0)
        Sigma = np.cov(Xc.T, bias=False) + reg * np.eye(Xtr.shape[1])
        model[c] = {"pi": pi, "mu": mu, "Sigma": Sigma}
    return model

def predict_cqg(Xte, model):
    """
    g_c(x) = -1/2 ln|Sigma_c| - 1/2 (x-mu)^T Sigma^{-1} (x-mu) + ln(pi_c)
    """
    cls = list(model.keys())
    scores = np.zeros((Xte.shape[0], len(cls)))
    for j, c in enumerate(cls):
        mu = model[c]["mu"]
        Sigma = model[c]["Sigma"]
        pi = model[c]["pi"]

        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            Sigma = Sigma + 1e-4 * np.eye(Sigma.shape[0])
            sign, logdet = np.linalg.slogdet(Sigma)

        inv = np.linalg.inv(Sigma)
        diff = Xte - mu
        quad = np.einsum("ij,jk,ik->i", diff, inv, diff)
        scores[:, j] = -0.5 * logdet - 0.5 * quad + np.log(pi + 1e-12)

    return np.array([cls[i] for i in np.argmax(scores, axis=1)])


# ============================================================
# 1.3 (2) DMP = Distância mínima ao protótipo
# Protótipos: K-means por classe; k ótimo por classe via silhouette
# ============================================================
def choose_k_silhouette(Xc, k_min=1, k_max=10, random_state=0):
    """
    Escolhe k "ótimo" para uma classe usando silhouette (critério de clusterização).
    - Para k=1, silhouette não existe; usamos como fallback se a classe for pequena.
    """
    n = Xc.shape[0]
    if n < 2:
        return 1
    k_max_eff = min(k_max, n - 1)  # silhouette exige k <= n-1
    if k_max_eff < 2:
        return 1

    best_k = 2
    best_s = -np.inf

    for k in range(2, k_max_eff + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(Xc)
        # silhouette exige pelo menos 2 clusters e não pode ter cluster vazio (KMeans evita)
        s = silhouette_score(Xc, labels)
        if s > best_s:
            best_s = s
            best_k = k

    # comparado também com k=1 como opção "simples"
    # (não há silhouette para k=1; só escolhemos 1 se dados muito pequenos)
    return best_k

def train_dmp(Xtr, ytr, k_min=1, k_max=10, random_state=0):
    """
    Aplica K-means separadamente em cada classe.
    Cada classe tem seu k "ótimo" e gera seus protótipos (centroides).
    """
    protos = []
    proto_labels = []
    k_by_class = {}

    for c in np.unique(ytr):
        Xc = Xtr[ytr == c]
        k_opt = choose_k_silhouette(Xc, k_min=k_min, k_max=k_max, random_state=random_state)
        k_by_class[c] = k_opt

        km = KMeans(n_clusters=k_opt, n_init=10, random_state=random_state)
        km.fit(Xc)
        protos.append(km.cluster_centers_)
        proto_labels.extend([c] * k_opt)

    return np.vstack(protos), np.array(proto_labels), k_by_class

def predict_dmp(Xte, protos, proto_labels):
    """
    Classifica pela classe do protótipo mais próximo (distância Euclidiana).
    """
    preds = []
    for x in Xte:
        dists = np.linalg.norm(protos - x, axis=1)
        preds.append(proto_labels[np.argmin(dists)])
    return np.array(preds)


# =========================
# Acurácia global e por classe
# =========================
def acc_global(y_true, y_pred):
    return (y_true == y_pred).mean()

def acc_by_class(y_true, y_pred, classes):
    out = {}
    for c in classes:
        m = (y_true == c)
        out[c] = (y_true[m] == y_pred[m]).mean()
    return out

def summarize(records, classes):
    """
    records: lista de dicts com:
      - 'global'
      - 'per_class' (dict)
    Retorna DataFrame com Média ± Desvio para global e classes.
    """
    rows = []
    g = np.array([r["global"] for r in records], dtype=float)
    rows.append(["Global", g.mean(), g.std(ddof=1)])

    for c in classes:
        v = np.array([r["per_class"][c] for r in records], dtype=float)
        rows.append([f"Classe {c}", v.mean(), v.std(ddof=1)])

    return pd.DataFrame(rows, columns=["Métrica", "Média", "Desvio Padrão"])


# =========================
# Experimento principal (Nr=100) + PCA
# 2.1: mostrar VE(q)
# 2.2: repetir 1.3 para PCA com q "adequado" sem piorar desempenho
# =========================
Nr = 100
test_size = 0.30  # treino/teste - padrão 70/30

# Padronização ANTES do PCA e dos classificadores
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X)

# 2.1 Gráfico VE(q)
pca_full = PCA().fit(X_all_scaled)
VE = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(VE) + 1), VE, marker="o")
plt.xlabel("Número de Componentes (q)")
plt.ylabel("Variância Explicada Acumulada VE(q)")
plt.title("2.1 - PCA: Variância Explicada VE(q)")
plt.grid(True)
plt.ylim(0, 1.01)
plt.show()

# ---------- Baseline (sem PCA) ----------
raw_cqg_records = []
raw_dmp_records = []

for r in range(Nr):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=r
    )

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # CQG
    cqg_model = train_cqg(Xtr_s, ytr, reg=1e-6)
    yhat = predict_cqg(Xte_s, cqg_model)
    raw_cqg_records.append({
        "global": acc_global(yte, yhat),
        "per_class": acc_by_class(yte, yhat, classes),
    })

    # DMP
    protos, proto_labels, _ = train_dmp(Xtr_s, ytr, k_min=1, k_max=10, random_state=r)
    yhat = predict_dmp(Xte_s, protos, proto_labels)
    raw_dmp_records.append({
        "global": acc_global(yte, yhat),
        "per_class": acc_by_class(yte, yhat, classes),
    })

raw_cqg_mean = np.mean([r["global"] for r in raw_cqg_records])
raw_dmp_mean = np.mean([r["global"] for r in raw_dmp_records])

print("\n=== 1.3 Resultados (SEM PCA) ===")
print("\nCQG (SEM PCA):")
print(summarize(raw_cqg_records, classes))

print("\nDMP (SEM PCA):")
print(summarize(raw_dmp_records, classes))


# ---------- 2.1/2.2: escolher q "adequado" sem piorar desempenho ----------
# Critério objetivo: reduzir dimensão SEM piorar desempenho
# Implementação: procurar o MENOR q tal que a média global (Nr=100) do CQG e do DMP
# não seja inferior à do caso SEM PCA (com uma tolerância numérica pequena).
tol = 1e-4

def run_with_pca(q, Nr=100):
    cqg_rec = []
    dmp_rec = []
    for r in range(Nr):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=r
        )

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        pca = PCA(n_components=q, random_state=r)
        Xtr_p = pca.fit_transform(Xtr_s)
        Xte_p = pca.transform(Xte_s)

        # CQG + PCA
        cqg_model = train_cqg(Xtr_p, ytr, reg=1e-6)
        yhat = predict_cqg(Xte_p, cqg_model)
        cqg_rec.append({
            "global": acc_global(yte, yhat),
            "per_class": acc_by_class(yte, yhat, classes),
        })

        # DMP + PCA
        protos, proto_labels, _ = train_dmp(Xtr_p, ytr, k_min=1, k_max=10, random_state=r)
        yhat = predict_dmp(Xte_p, protos, proto_labels)
        dmp_rec.append({
            "global": acc_global(yte, yhat),
            "per_class": acc_by_class(yte, yhat, classes),
        })

    cqg_mean = np.mean([r["global"] for r in cqg_rec])
    dmp_mean = np.mean([r["global"] for r in dmp_rec])
    return cqg_rec, dmp_rec, cqg_mean, dmp_mean

best_q = None
best = None

for q in range(1, X.shape[1] + 1):
    cqg_rec, dmp_rec, cqg_mean, dmp_mean = run_with_pca(q, Nr=Nr)
    if (cqg_mean + tol >= raw_cqg_mean) and (dmp_mean + tol >= raw_dmp_mean):
        best_q = q
        best = (cqg_rec, dmp_rec)
        break

# Se nenhum q satisfizer (pode acontecer), escolhe q que maximiza o pior dos dois desempenhos.
if best_q is None:
    best_score = -np.inf
    for q in range(1, X.shape[1] + 1):
        cqg_rec, dmp_rec, cqg_mean, dmp_mean = run_with_pca(q, Nr=Nr)
        score = min(cqg_mean, dmp_mean)  # maximiza o "pior caso"
        if score > best_score:
            best_score = score
            best_q = q
            best = (cqg_rec, dmp_rec)

pca_cqg_records, pca_dmp_records = best

print(f"\n=== 2.2 Resultados (COM PCA) | q escolhido = {best_q} ===")
print("\nCQG (COM PCA):")
print(summarize(pca_cqg_records, classes))

print("\nDMP (COM PCA):")
print(summarize(pca_dmp_records, classes))
