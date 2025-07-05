import pandas as pd
from pgmpy.inference import VariableElimination
from src.Bayesian_networks.bn_economico import model  # Importa o modelo da rede econômica

# Carrega os dados
df = pd.read_csv("data/input.csv")

# Inicializa o inferidor
inference = VariableElimination(model)

# Ajusta os valores de 1–5 para 0–4
df["2.6 - Qual o nível de dependência externa (Terceirização)?"] -= 1
df["2.11 - Qual a disponibilidade dos recursos adequados ao projeto?"] -= 1

# Lista de resultados
resultados = []

for idx, row in df.iterrows():
    evidencias = {
        "dependencia": int(row["2.6 - Qual o nível de dependência externa (Terceirização)?"]),
        "disponibilidade": int(row["2.11 - Qual a disponibilidade dos recursos adequados ao projeto?"])
    }

    prob = inference.query(variables=["riscoEconomico"], evidence=evidencias)
    risco = round(prob.values[1] * 100, 2)
    resultados.append({
        "Instancia": idx + 1,
        "Risco Econômico": f"{risco}%"
    })

# Salva o resultado
df_saida = pd.DataFrame(resultados)
df_saida.to_csv("results/csv/resultado_inferencia_economico.csv", index=False)
print("✅ Resultados salvos em 'resultado_inferencia_economico.csv'")
