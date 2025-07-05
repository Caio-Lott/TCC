import pandas as pd
from pgmpy.inference import VariableElimination
from src.Bayesian_networks.bn_aplicacao import model  # Importa o modelo do seu script anterior

# Carrega os dados
df = pd.read_csv("data\\input.csv")  # deve ter colunas: nivelReuso, nivelInterface (valores de 1 a 5)

# Inicializa o inferidor
inference = VariableElimination(model)

# Converte valores [1–5] para índices [0–4]
df["2.9 - Qual o nível de reuso dos componentes?"] = df["2.9 - Qual o nível de reuso dos componentes?"] - 1
df["2.10 - Qual o nível de usabilidade da interface?"] = df["2.10 - Qual o nível de usabilidade da interface?"] - 1

# Lista para armazenar os resultados
resultados = []

for idx, row in df.iterrows():
    evidencias = {
        "nivelReuso": int(row["2.9 - Qual o nível de reuso dos componentes?"]),
        "nivelInterface": int(row["2.10 - Qual o nível de usabilidade da interface?"])
    }

    prob = inference.query(variables=["riscoAplicacao"], evidence=evidencias)
    risco = round(prob.values[1] * 100, 2)  # Probabilidade de risco = 1
    resultados.append({
        "Instancia": idx + 1,
        "Risco de Aplicação": f"{risco}%"
    })

# Salva como CSV
df_saida = pd.DataFrame(resultados)
df_saida.to_csv("results/csv/resultado_inferencia_aplicacao.csv", index=False)

print("✅ Resultados salvos em 'resultado_inferencia_aplicacao.csv'")