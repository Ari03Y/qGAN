import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.circuit import ParameterVector, Parameter
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import TwoLocal
import pickle 
from qiskit_machine_learning.connectors import TorchConnector

num_qubits = 2

# Generator Initialization
generator = TwoLocal(num_qubits, ['rx', 'rz', 'cx', 'rzz'], entanglement_blocks=None, entanglement='full', reps=3, parameter_prefix='θ_g', name='Generator')
generator = generator.decompose()

# Real Distribution Initialization
real_distr_circuit = QuantumCircuit(num_qubits)
real_distr_circuit.h(0)
real_distr_circuit.cx(0, 1)

# Discriminator Initialization
θ_d = Parameter("θ_d[38]")
discriminator = TwoLocal(num_qubits + 1, ['rx', 'rz'], entanglement_blocks=['cx', 'rzz'], entanglement='linear', reps=2, parameter_prefix='θ_d', name='Discriminator')
discriminator = discriminator.decompose()
for i in range(num_qubits):
    discriminator.cx(i, i + 1)
discriminator.rzz(θ_d, 0, 1)
θ_d = Parameter("θ_d[39]")
discriminator.rzz(θ_d, 1, 2)

gen_para = generator.num_parameters
disc_para = discriminator.num_parameters

gen_disc_circuit = QuantumCircuit(num_qubits + 1)
gen_disc_circuit.compose(generator, inplace=True)
gen_disc_circuit.barrier()
gen_disc_circuit.compose(discriminator, inplace=True)

real_disc_circuit = QuantumCircuit(num_qubits + 1)
real_disc_circuit.compose(real_distr_circuit, inplace=True)
real_disc_circuit.barrier()
real_disc_circuit.compose(discriminator, inplace=True)

# Generator cost function
def generator_cost(gen_params, disc_params, gen_disc_circuit):
    curr_params = np.append(disc_params.detach().numpy(), gen_params.detach().numpy())
    state_probs = Statevector(gen_disc_circuit.assign_parameters(curr_params)).probabilities()
    prob_fake_true = np.sum(state_probs[0b100:])
    cost = abs(-prob_fake_true) 
    return torch.tensor(cost, requires_grad=True)

# Discriminator cost function
def discriminator_cost(disc_params, gen_params, gen_disc_circuit, real_disc_circuit):
    curr_params = np.append(disc_params.detach().numpy(), gen_params.detach().numpy())
    gendisc_probs = Statevector(gen_disc_circuit.assign_parameters(curr_params)).probabilities()
    realdisc_probs = Statevector(real_disc_circuit.assign_parameters(disc_params.detach().numpy())).probabilities()
    prob_fake_true = np.sum(gendisc_probs[0b100:])
    prob_real_true = np.sum(realdisc_probs[0b100:])
    cost = abs(prob_fake_true - prob_real_true)
    return torch.tensor(cost, requires_grad=True)

# Kullback-Leibler Divergence
def calculate_kl_div(model_distribution: dict, target_distribution: dict):
    kl_div = 0
    for bitstring, p_data in target_distribution.items():
        if np.isclose(p_data, 0, atol=1e-8):
            continue
        if bitstring in model_distribution.keys():
            kl_div += (p_data * np.log(p_data) - p_data * np.log(model_distribution[bitstring]))
        else:
            kl_div += p_data * np.log(p_data) - p_data * np.log(1e-6)
    return kl_div

# Neural Network Initialization
sampler = Sampler()

gen_qnn = SamplerQNN(circuit=gen_disc_circuit, sampler=sampler, input_params=gen_disc_circuit.parameters[:disc_para], 
                     weight_params=gen_disc_circuit.parameters[disc_para:], sparse=False) 
disc_fake_qnn = SamplerQNN(circuit=gen_disc_circuit, sampler=sampler, input_params=gen_disc_circuit.parameters[disc_para:], 
                            weight_params=gen_disc_circuit.parameters[:disc_para], sparse=False)
disc_real_qnn = SamplerQNN(circuit=real_disc_circuit, sampler=sampler, input_params=[], 
                           weight_params=gen_disc_circuit.parameters[:disc_para], sparse=False)

# Initializing NN parameters
init_gen_params = np.random.uniform(low=-np.pi, high=np.pi, size=gen_para)
init_disc_params = np.random.uniform(low=-np.pi, high=np.pi, size=disc_para)
gen_params = torch.tensor(init_gen_params, requires_grad=True)
disc_params = torch.tensor(init_disc_params, requires_grad=True)

# Creating G prob distribution
init_gen_circuit = generator.assign_parameters(init_gen_params)
init_prob_dict = Statevector(init_gen_circuit).probabilities_dict()
real_prob_dict = Statevector(real_distr_circuit).probabilities_dict()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.set_title("Initial generator distribution")
plot_histogram(init_prob_dict, ax=ax1)
ax2.set_title("Real distribution")
plot_histogram(real_prob_dict, ax=ax2)
fig.tight_layout()

# ML Training
generator_optimizer = optim.Adam([gen_params], lr=0.02)
discriminator_optimizer = optim.Adam([disc_params], lr=0.02)
best_gen_params = torch.tensor(init_gen_params, requires_grad=True)
gloss = []
dloss = []
kl_div = []

TABLE_HEADERS = "Epoch | Generator cost | Discriminator cost | KL Div. |"
print(TABLE_HEADERS)

for epoch in range(100):  
    D_STEPS = 5
    for disc_train_step in range(D_STEPS):
        d_fake = torch.tensor(disc_fake_qnn.backward(gen_params.detach().numpy(), disc_params.detach().numpy())[1]).to_dense()[0, 0b100:]
        d_fake = torch.sum(d_fake, axis=0)
        d_real = torch.tensor(disc_real_qnn.backward([], disc_params.detach().numpy())[1]).to_dense()[0, 0b100:]
        d_real = torch.sum(d_real, axis=0)
        grad_dcost = [d_fake[i] - d_real[i] for i in range(discriminator.num_parameters)]
        grad_dcost = torch.tensor(grad_dcost, dtype=torch.double)

        discriminator_optimizer.zero_grad()
        disc_params.grad = grad_dcost
        discriminator_optimizer.step()

        if disc_train_step % D_STEPS == 0:
            dloss.append(discriminator_cost(disc_params, gen_params, gen_disc_circuit, real_disc_circuit).item())

    for gen_train_step in range(1):
        grads = torch.tensor(gen_qnn.backward(disc_params.detach().numpy(), gen_params.detach().numpy())[1]).to_dense()[0][0b100:]
        grads = -torch.sum(grads, axis=0)
        grads = torch.tensor(grads, dtype=torch.double)

        generator_optimizer.zero_grad()
        gen_params.grad = grads
        generator_optimizer.step()
        gloss.append(generator_cost(gen_params, disc_params, gen_disc_circuit).item())

    gen_checkpoint_circuit = generator.assign_parameters(gen_params.detach().numpy())
    gen_prob_dict = Statevector(gen_checkpoint_circuit).probabilities_dict()
    real_prob_dict = Statevector(real_distr_circuit).probabilities_dict()
    current_kl = calculate_kl_div(gen_prob_dict, real_prob_dict)
    kl_div.append(current_kl)

    if np.min(kl_div) == current_kl:
        best_gen_params = pickle.loads(pickle.dumps(gen_params.detach().numpy()))

    if epoch % 10 == 0:
        for header, val in zip(TABLE_HEADERS.split('|'), (epoch, gloss[-1], dloss[-1], kl_div[-1])):
            print(f"{val:.3g} ".rjust(len(header)), end="|")
        print()

fig, (loss, kl) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [0.75, 1]}, figsize=(6, 4))
fig.suptitle('QGAN training stats')
fig.supxlabel('Training epoch')
loss.plot(range(len(gloss)), gloss, label="Generator loss")
loss.plot(range(len(dloss)), dloss, label="Discriminator loss", color="C3")
loss.legend()
loss.set(ylabel='Loss')
kl.plot(range(len(kl_div)), kl_div, label="Kullback-Leibler Divergence", color="C1")
kl.set(ylabel='Kullback-Leibler Divergence')
kl.legend()
fig.tight_layout()

# Create test circuit with new parameters
gen_checkpoint_circuit = generator.assign_parameters(best_gen_params)
gen_prob_dict = Statevector(gen_checkpoint_circuit).probabilities_dict()
real_prob_dict = Statevector(real_distr_circuit).probabilities_dict()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
plot_histogram(gen_prob_dict, ax=ax1)
ax1.set_title("Trained generator distribution")
plot_histogram(real_prob_dict, ax=ax2)
ax2.set_title("Real distribution")
fig.tight_layout()
