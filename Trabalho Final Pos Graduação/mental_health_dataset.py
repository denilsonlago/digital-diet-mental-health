

# Análise do Dataset: Digital Diet Mental Health Insights
# Análise de Dados sobre Dieta Digital e Saúde Mental

# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuração dos gráficos
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*60)
print("ANÁLISE DO DATASET: DIGITAL DIET MENTAL HEALTH INSIGHTS")
print("="*60)

# ==========================================
# 1. CONTEXTUALIZAÇÃO DO DATASET
# ==========================================

print("\n1. CONTEXTUALIZAÇÃO DO DATASET")
print("-" * 40)

contextualizacao = """
OBJETIVO DO DATASET:
Este dataset investiga a relação entre hábitos de consumo digital (tempo gasto em dispositivos,
redes sociais, streaming) e indicadores de saúde mental (ansiedade, depressão, qualidade do sono).

DESCRIÇÃO DAS PRINCIPAIS COLUNAS ESPERADAS:
- Age: Idade do participante
- Gender: Gênero do participante
- Screen_Time_Hours: Horas diárias gastas em telas
- Social_Media_Hours: Horas diárias em redes sociais
- Streaming_Hours: Horas diárias assistindo conteúdo streaming
- Gaming_Hours: Horas diárias jogando
- Exercise_Hours: Horas semanais de exercício físico
- Sleep_Hours: Horas de sono por noite
- Anxiety_Level: Nível de ansiedade (escala 1-10)
- Depression_Level: Nível de depressão (escala 1-10)
- Stress_Level: Nível de estresse (escala 1-10)
- Mental_Health_Score: Pontuação geral de saúde mental
- Physical_Symptoms: Presença de sintomas físicos
- Mood_Rating: Nível de humor 
"""

print(contextualizacao)

# ==========================================
# 2. CARREGAMENTO DO DATASET
# ==========================================

print("\n2. CARREGAMENTO DO DATASET")
print("-" * 40)

try:
    # Carregamento do arquivo CSV real
    df = pd.read_csv('digital_diet_mental_health.csv')
    print("Dataset carregado com sucesso!")
    print(f"Formato do dataset: {df.shape}")
    print(f"Colunas: {list(df.columns)}")
    
except FileNotFoundError:
    print("AVISO: Arquivo 'digital_diet_mental_health.csv' não encontrado.")
    
    print(f"Colunas: {list(df.columns)}")

# ==========================================
# 3. TRATAMENTO DE COLUNAS E LINHAS
# ==========================================

print("\n3. TRATAMENTO DE DADOS")
print("-" * 40)

# Informações gerais do dataset
print("Informações gerais do dataset:")
print(df.info())

print("\nPrimeiras 5 linhas:")
print(df.head())

print("\nEstatísticas descritivas:")
print(df.describe())

# Verificação de valores nulos
print("\nVerificação de valores nulos:")
null_counts = df.isnull().sum()
print(null_counts)

if null_counts.sum() > 0:
    print("Valores nulos encontrados. Realizando tratamento...")
    df = df.dropna()
else:
    print("Nenhum valor nulo encontrado!")

# Verificação de duplicatas
duplicates = df.duplicated().sum()
print(f"\nLinhas duplicadas: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicatas removidas!")

print(f"\nDataset após tratamento: {df.shape}")

# ==========================================
# 4. ANÁLISE GRÁFICA DOS DADOS (UM GRÁFICO POR VEZ)
# ==========================================

print("\n4. ANÁLISE GRÁFICA DOS DADOS")
print("-" * 40)

def show_plot(plot_function, title, description):
    """Função para mostrar um gráfico por vez com descrição"""
    print(f"\n{title}")
    print("-" * len(title))
    print(description)
    
    plt.figure(figsize=(10, 6))
    plot_function()
    plt.tight_layout()
    plt.show()
    
    input("Pressione Enter para continuar para o próximo gráfico...")

# GRÁFICO 1: Distribuição das Idades
def plot_age_distribution():
    plt.hist(df['Age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribuição das Idades dos Participantes', fontsize=14, fontweight='bold')
    plt.xlabel('Idade (anos)')
    plt.ylabel('Frequência')
    plt.grid(True, alpha=0.3)

show_plot(
    plot_age_distribution,
    "GRÁFICO 1: DISTRIBUIÇÃO DAS IDADES",
    "Este histograma mostra a distribuição etária dos participantes do estudo.\nPermite identificar se há concentração em determinadas faixas etárias."
)

# GRÁFICO 2: Distribuição por Gênero
def plot_gender_distribution():
    gender_counts = df['Gender'].value_counts()
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Distribuição por Gênero', fontsize=14, fontweight='bold')

show_plot(
    plot_gender_distribution,
    "GRÁFICO 2: DISTRIBUIÇÃO POR GÊNERO",
    "Gráfico de pizza mostrando a proporção de participantes por gênero.\nImportante para entender a representatividade da amostra."
)

# GRÁFICO 3: Tempo de Tela Diário
def plot_screen_time():
    plt.hist(df['Screen_Time_Hours'], bins=25, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Distribuição do Tempo de Tela Diário', fontsize=14, fontweight='bold')
    plt.xlabel('Horas por dia')
    plt.ylabel('Frequência')
    plt.axvline(df['Screen_Time_Hours'].mean(), color='red', linestyle='--', 
                label=f'Média: {df["Screen_Time_Hours"].mean():.1f}h')
    plt.legend()
    plt.grid(True, alpha=0.3)

show_plot(
    plot_screen_time,
    "GRÁFICO 3: TEMPO DE TELA DIÁRIO",
    "Distribuição das horas diárias gastas em telas pelos participantes.\nA linha vermelha indica a média, permitindo identificar padrões de uso."
)

# GRÁFICO 4: Correlação Tempo de Tela vs Ansiedade
def plot_screen_anxiety():
    plt.scatter(df['Screen_Time_Hours'], df['Anxiety_Level'], alpha=0.6, color='red')
    plt.title('Relação: Tempo de Tela vs Nível de Ansiedade', fontsize=14, fontweight='bold')
    plt.xlabel('Tempo de Tela (horas/dia)')
    plt.ylabel('Nível de Ansiedade (1-10)')
    
    # Linha de tendência
    z = np.polyfit(df['Screen_Time_Hours'], df['Anxiety_Level'], 1)
    p = np.poly1d(z)
    plt.plot(df['Screen_Time_Hours'], p(df['Screen_Time_Hours']), "r--", alpha=0.8)
    
    # Correlação
    corr = df['Screen_Time_Hours'].corr(df['Anxiety_Level'])
    plt.text(0.05, 0.95, f'Correlação: {corr:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    plt.grid(True, alpha=0.3)

show_plot(
    plot_screen_anxiety,
    "GRÁFICO 4: TEMPO DE TELA vs ANSIEDADE",
    "Scatter plot investigando a relação entre tempo gasto em telas e níveis de ansiedade.\nA linha tracejada mostra a tendência e o valor de correlação indica a força da relação."
)

# GRÁFICO 5: Saúde Mental por Gênero
def plot_mental_health_by_gender():
    sns.boxplot(data=df, x='Gender', y='Mental_Health_Score', palette='Set2')
    plt.title('Pontuação de Saúde Mental por Gênero', fontsize=14, fontweight='bold')
    plt.xlabel('Gênero')
    plt.ylabel('Pontuação de Saúde Mental')
    plt.xticks(rotation=0)

show_plot(
    plot_mental_health_by_gender,
    "GRÁFICO 5: SAÚDE MENTAL POR GÊNERO",
    "Boxplot comparando as pontuações de saúde mental entre diferentes gêneros.\nPermite identificar diferenças na distribuição e possíveis outliers por grupo."
)

# GRÁFICO 6: Heatmap de Correlações
def plot_correlation_heatmap():
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    # Focar nas correlações com Mental_Health_Score
    mental_health_corr = correlation_matrix[['Mental_Health_Score']].sort_values(
        'Mental_Health_Score', ascending=False)
    
    sns.heatmap(mental_health_corr, annot=True, cmap='RdYlBu_r', center=0, 
                cbar_kws={'label': 'Correlação'})
    plt.title('Correlações com Pontuação de Saúde Mental', fontsize=14, fontweight='bold')
    plt.tight_layout()

show_plot(
    plot_correlation_heatmap,
    "GRÁFICO 6: CORRELAÇÕES COM SAÚDE MENTAL",
    "Heatmap mostrando as correlações entre todas as variáveis numéricas e a pontuação de saúde mental.\nCores mais vermelhas indicam correlação negativa, azuis indicam correlação positiva."
)

# GRÁFICO 7: Distribuição das Horas de Sono
def plot_sleep_distribution():
    plt.hist(df['Sleep_Hours'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Distribuição das Horas de Sono por Noite', fontsize=14, fontweight='bold')
    plt.xlabel('Horas de Sono')
    plt.ylabel('Frequência')
    plt.axvline(df['Sleep_Hours'].mean(), color='red', linestyle='--', 
                label=f'Média: {df["Sleep_Hours"].mean():.1f}h')
    plt.axvline(8, color='green', linestyle=':', label='Recomendado: 8h')
    plt.legend()
    plt.grid(True, alpha=0.3)

show_plot(
    plot_sleep_distribution,
    "GRÁFICO 7: DISTRIBUIÇÃO DAS HORAS DE SONO",
    "Histograma das horas de sono por noite dos participantes.\nA linha verde mostra a recomendação de 8 horas, permitindo avaliar se a amostra segue as diretrizes de sono saudável."
)

# GRÁFICO 8: Exercício vs Saúde Mental
def plot_exercise_mental_health():
    plt.scatter(df['Exercise_Hours'], df['Mental_Health_Score'], alpha=0.6, color='green')
    plt.title('Relação: Exercício vs Saúde Mental', fontsize=14, fontweight='bold')
    plt.xlabel('Horas de Exercício por Semana')
    plt.ylabel('Pontuação de Saúde Mental')
    
    # Linha de tendência
    z = np.polyfit(df['Exercise_Hours'], df['Mental_Health_Score'], 1)
    p = np.poly1d(z)
    plt.plot(df['Exercise_Hours'], p(df['Exercise_Hours']), "r--", alpha=0.8)
    
    # Correlação
    corr = df['Exercise_Hours'].corr(df['Mental_Health_Score'])
    plt.text(0.05, 0.95, f'Correlação: {corr:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8))
    plt.grid(True, alpha=0.3)

show_plot(
    plot_exercise_mental_health,
    "GRÁFICO 8: EXERCÍCIO vs SAÚDE MENTAL",
    "Scatter plot explorando a relação entre horas de exercício semanal e pontuação de saúde mental.\nA correlação positiva indicaria benefícios do exercício para a saúde mental."
)

# GRÁFICO 9: Indicadores de Saúde Mental
def plot_mental_indicators():
    mental_indicators = ['Stress_Level', 'Anxiety_Level', 'Depression_Level']
    data_to_plot = [df[col] for col in mental_indicators]
    
    plt.boxplot(data_to_plot, labels=['Estresse', 'Ansiedade', 'Depressão'])
    plt.title('Distribuição dos Indicadores de Saúde Mental', fontsize=14, fontweight='bold')
    plt.ylabel('Nível (1-10)')
    plt.grid(True, alpha=0.3)

show_plot(
    plot_mental_indicators,
    "GRÁFICO 9: INDICADORES DE SAÚDE MENTAL",
    "Boxplots comparando a distribuição dos três principais indicadores de saúde mental.\nPermite identificar qual indicador apresenta maior variabilidade e valores extremos."
)

# GRÁFICO 10: Redes Sociais vs Nível de Humor
def plot_social_media_isolation():
    plt.scatter(df['Social_Media_Hours'], df['Mood_Rating'], alpha=0.6, color='purple')
    plt.title('Redes Sociais vs Humor', fontsize=14, fontweight='bold')
    plt.xlabel('Horas em Redes Sociais por Dia')
    plt.ylabel('Nível de Humor')
    
    # Linha de tendência
    z = np.polyfit(df['Social_Media_Hours'], df['Mood_Rating'], 1)
    p = np.poly1d(z)
    plt.plot(df['Social_Media_Hours'], p(df['Social_Media_Hours']), "r--", alpha=0.8)
    
    # Correlação
    corr = df['Social_Media_Hours'].corr(df['Mood_Rating'])
    plt.text(0.05, 0.95, f'Correlação: {corr:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round", facecolor='plum', alpha=0.8))
    plt.grid(True, alpha=0.3)

show_plot(
    plot_social_media_isolation,
    "GRÁFICO 10: REDES SOCIAIS vs NÍVEL DE HUMOR",
    "Investigação da relação entre tempo em redes sociais e nível de humor.\nUma correlação negativa sugeriria que mais tempo online pode reduzir o humor."
)

# ==========================================
# 5. SEPARAÇÃO EM TREINO E TESTE
# ==========================================

print("\n5. SEPARAÇÃO DOS DADOS EM TREINO E TESTE")
print("-" * 40)

# Preparação dos dados para machine learning
le = LabelEncoder()
df_encoded = df.copy()

# Codificação de variáveis categóricas
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if col in df_encoded.columns:
        df_encoded[col] = le.fit_transform(df[col])

# Definindo features (X) e target (y)
target_column = 'Mental_Health_Score'
feature_columns = [col for col in df_encoded.columns if col != target_column]

X = df_encoded[feature_columns]
y = df_encoded[target_column]

print(f"Features selecionadas: {len(feature_columns)} variáveis")
print(f"Target: {target_column}")
print(f"Formato dos dados - X: {X.shape}, y: {y.shape}")

# Separação em treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nDados de treino - X: {X_train.shape}, y: {y_train.shape}")
print(f"Dados de teste - X: {X_test.shape}, y: {y_test.shape}")

# Salvando os datasets processados
print("\nSalvando os datasets...")

df_encoded.to_csv('digital_diet_processed.csv', index=False)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
train_data.to_csv('digital_diet_train.csv', index=False)
test_data.to_csv('digital_diet_test.csv', index=False)

print("Arquivos salvos:")
print("- digital_diet_processed.csv")
print("- digital_diet_train.csv (70%)")
print("- digital_diet_test.csv (30%)")

# ==========================================
# 6. RESUMO DA ANÁLISE
# ==========================================

print("\n" + "="*60)
print("RESUMO DA ANÁLISE")
print("="*60)

# Calculando estatísticas principais
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_with_mental_health = df[numeric_cols].corr()['Mental_Health_Score'].abs().sort_values(ascending=False)

print(f"""
DATASET PROCESSADO:
- Total de registros: {df.shape[0]}
- Total de features: {len(feature_columns)}
- Registros de treino: {X_train.shape[0]} (70%)
- Registros de teste: {X_test.shape[0]} (30%)

ESTATÍSTICAS PRINCIPAIS:
- Idade média: {df['Age'].mean():.1f} anos
- Tempo médio de tela: {df['Screen_Time_Hours'].mean():.1f} horas/dia
- Horas médias de sono: {df['Sleep_Hours'].mean():.1f} horas/noite
- Pontuação média de saúde mental: {df['Mental_Health_Score'].mean():.1f}

CORRELAÇÕES MAIS FORTES COM SAÚDE MENTAL:
{correlation_with_mental_health.head(4).to_string()}

PRÓXIMOS PASSOS:
1. Aplicar algoritmos de machine learning
2. Validar modelos com dados de teste
3. Interpretar resultados para insights acionáveis
""")

print("\nAnálise concluída com sucesso!")