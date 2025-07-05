import pandas as pd
from pgmpy.inference import VariableElimination
from src.Bayesian_networks.bn_cultural import model  # Importa o modelo da rede cultural

# Carrega os dados
df = pd.read_csv("data/input.csv")

# Inicializa o inferidor
inference = VariableElimination(model)

# Ajusta os valores
df["1.6 - Qual a forma de organização da sua equipe?"] -= 1
df["3.4 - Qual o nível de conflito interno no time?"] -= 1

# Lista de resultados
resultados = []

for idx, row in df.iterrows():
    evidencias = {
        "organizacao": int(row["1.6 - Qual a forma de organização da sua equipe?"]),
        "conflito": int(row["3.4 - Qual o nível de conflito interno no time?"])
    }

    prob = inference.query(variables=["riscoCultural"], evidence=evidencias)
    risco = round(prob.values[1] * 100, 2)
    resultados.append({
        "Instancia": idx + 1,
        "Risco Cultural": f"{risco}%"
    })

# Salva o resultado
df_saida = pd.DataFrame(resultados)
df_saida.to_csv("results/csv/resultado_inferencia_cultural.csv", index=False)
print("✅ Resultados salvos em 'resultado_inferencia_cultural.csv'")
