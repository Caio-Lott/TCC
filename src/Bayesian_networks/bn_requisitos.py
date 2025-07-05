from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import json

# Distribuições marginais dos indicadores
estabilidade_dist = [0.0455, 0.1364, 0.4545, 0.3182, 0.0455]
complexidade_req_dist = [0.0, 0.0455, 0.6364, 0.2727, 0.0455]

# P(Risco | indicador)
p_risco_dado_estabilidade = [0.999, 0.666488, 0.50005, 0.285698, 0.999]
p_risco_dado_complexidade = [0.0, 0.0, 0.499972, 0.666733, 0.0]

# Criação da estrutura da rede
model = DiscreteBayesianNetwork([
    ("estabilidade", "riscoRequisitos"),
    ("complexidadeReq", "riscoRequisitos")
])

# CPDs dos indicadores
cpd_estabilidade = TabularCPD("estabilidade", 5, [[p] for p in estabilidade_dist])
cpd_complexidade_req = TabularCPD("complexidadeReq", 5, [[p] for p in complexidade_req_dist])

# CPD do risco
cpd_risco_values = []
for i in range(5):
    for j in range(5):
        prob = (p_risco_dado_estabilidade[i] + p_risco_dado_complexidade[j]) / 2
        c0 = round(1 - prob, 6)
        c1 = round(prob, 6)
        cpd_risco_values.append([c0, c1])

cpd_risco_matrix = [
    [pair[0] for pair in cpd_risco_values],
    [pair[1] for pair in cpd_risco_values]
]

cpd_risco = TabularCPD(
    variable="riscoRequisitos",
    variable_card=2,
    values=cpd_risco_matrix,
    evidence=["estabilidade", "complexidadeReq"],
    evidence_card=[5, 5]
)

model.add_cpds(cpd_estabilidade, cpd_complexidade_req, cpd_risco)
assert model.check_model()

# Salva em JSON
rede_info = {
    "edges": list(model.edges()),
    "cpds": {
        "estabilidade": estabilidade_dist,
        "complexidadeReq": complexidade_req_dist,
        "riscoRequisitos (0=Não, 1=Sim)": cpd_risco_matrix
    }
}

with open("results/json/rede_requisitos.json", "w", encoding="utf-8") as f:
    json.dump(rede_info, f, indent=4, ensure_ascii=False)

print("✅ Rede bayesiana salva como 'rede_requisitos.json'")
