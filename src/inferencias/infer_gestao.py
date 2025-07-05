import pandas as pd
from pgmpy.inference import VariableElimination
from src.Bayesian_networks.bn_gestao import model  # Importa o modelo para o grupo gestao

# Carrega os dados
df = pd.read_csv("data/input.csv")

# Inicializa o inferidor
inference = VariableElimination(model)

# Converte valores [1–5] para índices [0–4]
df["1.2 - Qual o nível médio de complexidade dos projetos nos quais você trabalha?"] = df["1.2 - Qual o nível médio de complexidade dos projetos nos quais você trabalha?"] - 1
df["1.4 - Qual o tamanho médio da(s) equipe(s) em que você trabalha?"] = df["1.4 - Qual o tamanho médio da(s) equipe(s) em que você trabalha?"] - 1

# Lista para armazenar os resultados
resultados = []

for idx, row in df.iterrows():
    evidencias = {
        "complexidade": int(row["1.2 - Qual o nível médio de complexidade dos projetos nos quais você trabalha?"]),
        "tamanhoEquipe": int(row["1.4 - Qual o tamanho médio da(s) equipe(s) em que você trabalha?"])
    }

    prob = inference.query(variables=["riscoGestao"], evidence=evidencias)
    risco = round(prob.values[1] * 100, 2)
    resultados.append({
        "Instancia": idx + 1,
        "Risco Gestao": f"{risco}%"
    })

# Salva como CSV
df_saida = pd.DataFrame(resultados)
df_saida.to_csv("results/csv/resultado_inferencia_gestao.csv", index=False)
print("✅ Resultados salvos em 'resultado_inferencia_gestao.csv'")
