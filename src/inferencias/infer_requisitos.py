import pandas as pd
from pgmpy.inference import VariableElimination
from src.Bayesian_networks.bn_requisitos import model  # Importa o modelo da rede de requisitos

# Carrega os dados
df = pd.read_csv("data/input.csv")

# Inicializa o inferidor
inference = VariableElimination(model)

# Ajusta os valores
df["2.2 - Qual o nível de estabilidade dos requisitos?"] -= 1
df["2.3 - Qual o nível de complexidade dos requisitos?"] -= 1

# Lista de resultados
resultados = []

for idx, row in df.iterrows():
    evidencias = {
        "estabilidade": int(row["2.2 - Qual o nível de estabilidade dos requisitos?"]),
        "complexidadeReq": int(row["2.3 - Qual o nível de complexidade dos requisitos?"])
    }

    prob = inference.query(variables=["riscoRequisitos"], evidence=evidencias)
    risco = round(prob.values[1] * 100, 2)
    resultados.append({
        "Instancia": idx + 1,
        "Risco de Requisitos": f"{risco}%"
    })

# Salva o resultado
df_saida = pd.DataFrame(resultados)
df_saida.to_csv("results/csv/resultado_inferencia_requisitos.csv", index=False)
print("✅ Resultados salvos em 'resultado_inferencia_requisitos.csv'")
