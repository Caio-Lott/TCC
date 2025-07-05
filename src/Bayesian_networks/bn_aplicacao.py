from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import json

# Distribuições marginais dos indicadores
nivel_reuso_dist = [0.0, 0.0909, 0.2273, 0.4545, 0.2273]
nivel_interface_dist = [0.0, 0.0909, 0.3182, 0.4545, 0.1364]

# P(Risco | indicador)
p_risco_dado_nivel_reuso = [0.0, 0.500048, 0.799903, 0.40004, 0.599928]
p_risco_dado_nivel_interface = [0.0, 0.0, 0.571395, 0.60006, 0.66649]

# Criação da estrutura da rede
model = DiscreteBayesianNetwork([
    ("nivelReuso", "riscoAplicacao"),
    ("nivelInterface", "riscoAplicacao")
])

# CPDs dos indicadores
cpd_nivel_reuso = TabularCPD("nivelReuso", 5, [[p] for p in nivel_reuso_dist])
cpd_nivel_interface = TabularCPD("nivelInterface", 5, [[p] for p in nivel_interface_dist])

# CPD do risco de aplicação (binário: 0 = não ocorre, 1 = ocorre)
# Assumimos independência condicional: média entre os efeitos individuais
cpd_risco_aplicacao_values = []
for i in range(5):
    for j in range(5):
        prob = (p_risco_dado_nivel_reuso[i] + p_risco_dado_nivel_interface[j]) / 2
        c0 = round(1 - prob, 6)
        c1 = round(prob, 6)
        cpd_risco_aplicacao_values.append([c0, c1])

# Reformata em matriz (2, 25)
cpd_risco_aplicacao_matrix = [
    [pair[0] for pair in cpd_risco_aplicacao_values],  # risco = 0
    [pair[1] for pair in cpd_risco_aplicacao_values]   # risco = 1
]

cpd_risco_aplicacao = TabularCPD(
    variable="riscoAplicacao",
    variable_card=2,
    values=cpd_risco_aplicacao_matrix,
    evidence=["nivelReuso", "nivelInterface"],
    evidence_card=[5, 5]
)

# Adiciona os CPDs ao modelo
model.add_cpds(cpd_nivel_reuso, cpd_nivel_interface, cpd_risco_aplicacao)

# Verifica consistência
assert model.check_model()

# Salva estrutura e CPDs em JSON
rede_info = {
    "edges": list(model.edges()),
    "cpds": {
        "nivelReuso": nivel_reuso_dist,
        "nivelInterface": nivel_interface_dist,
        "riscoAplicacao (0=Não, 1=Sim)": cpd_risco_aplicacao_matrix
    }
}

with open("results/json/rede_aplicacao.json", "w", encoding="utf-8") as f:
    json.dump(rede_info, f, indent=4, ensure_ascii=False)

print("✅ Rede bayesiana salva como 'rede_aplicacao.json'")
