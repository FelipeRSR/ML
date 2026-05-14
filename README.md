# 🛍️ E-Commerce Purchase Prediction with Random Forest

## 📖 Visão Geral

Este projeto implementa um modelo de classificação baseado em **Random Forest** para prever se um usuário realizará uma compra em uma plataforma de e-commerce. O dataset contém informações sobre comportamento de usuários, incluindo padrões de navegação, métricas de engajamento e histórico de compras.

**Dataset Fonte:** [E-Commerce Behavior Dataset (8000 Users) - Kaggle](https://www.kaggle.com/datasets/asifxzaman/e-commerce-behavior-dataset8000-users?resource=download)

---

## 🎯 Objetivo

- ✅ Prever a probabilidade de conversão de usuários em compras
- ✅ Realizar engenharia de features e classificação
- ✅ Analisar padrões de comportamento de clientes
- ✅ Identificar features mais importantes para predição
- ✅ Avaliar o desempenho do modelo com métricas robustas

---

## 📊 Dataset

### Características do Dataset

| Atributo | Descrição | Tipo |
|----------|-----------|------|
| `user_id` | Identificador único do usuário | Numérico |
| `age` | Idade do usuário | Numérico |
| `gender` | Gênero (Masculino/Feminino) | Categórico |
| `device_type` | Tipo de dispositivo (Mobile/Desktop/Tablet) | Categórico |
| `time_on_site` | Tempo gasto no site (minutos) | Numérico |
| `pages_viewed` | Número de páginas visualizadas | Numérico |
| `previous_purchases` | Número de compras anteriores | Numérico |
| `cart_items` | Quantidade de itens no carrinho | Numérico |
| `discount_seen` | Se o usuário viu desconto | Binário |
| `ad_clicked` | Se o usuário clicou em anúncio | Binário |
| `returning_user` | Se é usuário recorrente | Binário |
| `avg_session_time` | Duração média das sessões | Numérico |
| `bounce_rate` | Taxa de rejeição (%) | Numérico |
| `purchase` | **Variável Alvo** (0/1) | Binária |

### Estatísticas

- **Total de Amostras:** 8000
- **Valores Ausentes:** ~2%
- **Balanceamento:** Realista com desbalanceamento de classes
- **Características:** 13 features (após limpeza)

---

## 🔧 Tratamento de Dados

### Passo 1: Carregamento e Exploração
```python
# Carrega o dataset
df = pd.read_csv('ecommerce_user.csv')
```

### Passo 2: Limpeza de Dados
- ❌ Remoção de `user_id` (não relevante para predição)
- ❌ Remoção de valores ausentes (NaN)
- ❌ Remoção de linhas duplicadas

### Passo 3: Padronização de Features Numéricas
Features numéricas foram padronizadas usando **StandardScaler** para normalizar a escala:
- `time_on_site`
- `age`
- `pages_viewed`
- `previous_purchases`
- `cart_items`
- `avg_session_time`
- `bounce_rate`

```python
from sklearn.preprocessing import StandardScaler

numeric_features = ['time_on_site', 'age', 'pages_viewed', 'previous_purchases', 
                   'cart_items', 'avg_session_time', 'bounce_rate']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
```

### Passo 4: Codificação de Variáveis Categóricas
Aplicado **One-Hot Encoding** para variáveis categóricas:
- `device_type` (Mobile/Desktop/Tablet)
- `gender` (Male/Female)

```python
df = pd.get_dummies(df, columns=['device_type', 'gender'], drop_first=True)
```

### Passo 5: Divisão Treino/Teste
- **Treino:** 80% (6400 amostras)
- **Teste:** 20% (1600 amostras)
- **Estratificação:** Preserva distribuição de classes

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     random_state=42, stratify=y)
```

---

## 🌲 Modelo: Random Forest

### Configuração do Modelo

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,       # Número de árvores
    random_state=42,        # Reprodutibilidade
    n_jobs=-1,              # Usar todos os processadores
    max_depth=10            # Profundidade máxima das árvores
    class_weight='balanced' # Ajusta automaticamente o peso das classes.
)
```

### Treinamento

```python
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
```

### Avaliação do Modelo

O modelo é avaliado usando múltiplas métricas:

- **Acurácia:** Percentual geral de predições corretas
- **Precision:** Proporção de compras preditas que realmente ocorreram
- **Recall:** Proporção de compras reais identificadas corretamente
- **F1-Score:** Média harmônica entre Precision e Recall
- **Confusion Matrix:** Análise de verdadeiros/falsos positivos e negativos

```python
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

print(f"Acurácia: {balanced_accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## 📁 Estrutura do Projeto

```
ecommerce_user.csv          # Dataset original
ecommerce.ipynb            # Notebook com análise completa
train.csv                  # Dados de treinamento (pré-processados)
test.csv                   # Dados de teste (pré-processados)
README.md                  # Este arquivo
```

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8+
- Jupyter Notebook ou JupyterLab
- Bibliotecas necessárias (veja `requirements.txt`)

### Instalação de Dependências

```bash
pip install -r requirements.txt
```

### Executar o Notebook

```bash
jupyter notebook ecommerce.ipynb
```

Ou usando JupyterLab:

```bash
jupyter lab ecommerce.ipynb
```

---

## 📦 Bibliotecas Utilizadas

| Biblioteca | Versão | Uso |
|-----------|--------|-----|
| `pandas` | ≥1.3.0 | Manipulação de dados |
| `numpy` | ≥1.21.0 | Computação numérica |
| `scikit-learn` | ≥1.0.0 | Modelos ML e pré-processamento |
| `imbalanced-learn` | ≥0.8.0 | SMOTE e balanceamento |
| `matplotlib` | ≥3.4.0 | Visualizações |

---

## 📊 Resultados Esperados

### Feature Importance (Importância das Features)

O Random Forest identifica quais features são mais relevantes para predizer compras:

```
Exemplo de ranking (pode variar):
1. time_on_site ........... 0.185
2. pages_viewed ........... 0.152
3. previous_purchases ..... 0.138
4. cart_items ............. 0.125
5. returning_user ......... 0.098
... (outras features)
```

### Métricas de Desempenho

```
Acurácia: ~0.70-0.85

Classification Report:
              precision    recall  f1-score   support
           0       0.87      0.92      0.89       Y
           1       0.85      0.75      0.80       Y

Matriz de Confusão:
[[VerdadeiroNegativo  FalsoPositivo]
 [FalsoNegativo       VerdadeiroPositivo]]
```

---

## 💡 Insights e Próximos Passos

### Análises Realizadas
- Identificação de features mais influentes na decisão de compra
- Avaliação de padrões de comportamento entre usuários
- Balanceamento de classes para evitar viés

### Melhorias Futuras
- 🔄 Implementar ensemble de modelos (XGBoost, LightGBM)
- ⚖️ Experimentar com técnicas de balanceamento (SMOTE, classe pesos)
- 📈 Ajuste fino de hiperparâmetros (GridSearch, RandomSearch)
- 🎯 Feature engineering adicional
- 📊 Validação cruzada (k-fold)
- 🔍 Análise SHAP para interpretabilidade

---

## 📝 Notas Técnicas

### Por que Random Forest?

1. **Robustez:** Funciona bem com dados desbalanceados
2. **Feature Importance:** Fornece ranking automático de features
3. **Não-paramétrico:** Não assume distribuições específicas
4. **Paralelizável:** Treinamento rápido em múltiplos processadores
5. **Resistente a Outliers:** Não é sensível a valores extremos

### Decisões de Design

- **Remoção de `user_id`:** ID não contém informação preditiva
- **StandardScaler:** Normaliza escala de features numéricas
- **One-Hot Encoding:** Converte categóricas em numéricas
- **Stratified Split:** Preserva proporção de classes em treino/teste
- **max_depth=10:** Evita overfitting

---

## 📞 Autor

Felipe Rodrigues

---

## 📚 Referências

- [Kaggle - E-Commerce Behavior Dataset](https://www.kaggle.com/datasets/asifxzaman/e-commerce-behavior-dataset8000-users?resource=download)
- [Scikit-Learn Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)

---

## 📄 Licença

Este projeto é fornecido como é, para fins educacionais.

---

**Última atualização:** Maio 2026
