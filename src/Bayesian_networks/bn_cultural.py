from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import json

# Distribuições marginais dos indicadores
organizacao_dist = [0.4091, 0.0909, 0.0455, 0.3182, 0.1364]
conflito_dist = [0.3636, 0.3182, 0.2273, 0.0909, 0.0]

# P(Risco | indicador)
p_risco_dado_organizacao = [0.111109, 0.0, 0.0, 0.0, 0.333244]
p_risco_dado_conflito = [0.0, 0.285698, 0.0, 0.0, 0.0]

# Criação da estrutura da rede
model = DiscreteBayesianNetwork([
    ("organizacao", "riscoCultural"),
    ("conflito", "riscoCultural")
])

# CPDs dos indicadores
cpd_organizacao = TabularCPD("organizacao", 5, [[p] for p in organizacao_dist])
cpd_conflito = TabularCPD("conflito", 5, [[p] for p in conflito_dist])

# CPD do risco
cpd_risco_values = []
for i in range(5):
    for j in range(5):
        prob = (p_risco_dado_organizacao[i] + p_risco_dado_conflito[j]) / 2
        c0 = round(1 - prob, 6)
        c1 = round(prob, 6)
        cpd_risco_values.append([c0, c1])

cpd_risco_matrix = [
    [pair[0] for pair in cpd_risco_values],
    [pair[1] for pair in cpd_risco_values]
]

cpd_risco = TabularCPD(
    variable="riscoCultural",
    variable_card=2,
    values=cpd_risco_matrix,
    evidence=["organizacao", "conflito"],
    evidence_card=[5, 5]
)

model.add_cpds(cpd_organizacao, cpd_conflito, cpd_risco)
assert model.check_model()

# Salva em JSON
rede_info = {
    "edges": list(model.edges()),
    "cpds": {
        "organizacao": organizacao_dist,
        "conflito": conflito_dist,
        "riscoCultural (0=Não, 1=Sim)": cpd_risco_matrix
    }
}

with open("results/json/rede_cultural.json", "w", encoding="utf-8") as f:
    json.dump(rede_info, f, indent=4, ensure_ascii=False)

print("✅ Rede bayesiana salva como 'rede_cultural.json'")
