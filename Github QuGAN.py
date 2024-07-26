import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import TwoLocal
from torch import optim

#Num layers
REAL_DIST_NQUBITS = 2

#Real sample
real_circuit = QuantumCircuit(REAL_DIST_NQUBITS)
real_circuit.h(0)
real_circuit.cx(0, 1)
real_circuit.draw('mpl')

#Generator
generator = TwoLocal(REAL_DIST_NQUBITS, ['ry', 'rz'], 'cz', 'full', reps=2, parameter_prefix='θ_g', name='Generator')
generator = generator.decompose()
generator.draw('mpl')

#Discriminator
disc_weights = ParameterVector('θ_d', 12)
discriminator = QuantumCircuit(3, name="Discriminator")
discriminator.h(0)
for i in range(3):
    discriminator.rx(disc_weights[i * 3], i)
    discriminator.ry(disc_weights[i * 3 + 1], i)
    discriminator.rz(disc_weights[i * 3 + 2], i)
discriminator.cx(0, 2)
discriminator.cx(1, 2)
discriminator.rx(disc_weights[9], 2)
discriminator.ry(disc_weights[10], 2)
discriminator.rz(disc_weights[11], 2)
discriminator.draw('mpl')

N_GPARAMS = generator.num_parameters
N_DPARAMS = discriminator.num_parameters

gen_disc_circuit = QuantumCircuit(REAL_DIST_NQUBITS + 1)
gen_disc_circuit.compose(generator, inplace=True)
gen_disc_circuit.compose(discriminator, inplace=True)
gen_disc_circuit.draw('mpl')

# Real ceation
real_disc_circuit = QuantumCircuit(REAL_DIST_NQUBITS + 1)
real_disc_circuit.compose(real_circuit, inplace=True)
real_disc_circuit.compose(discriminator, inplace=True)
real_disc_circuit.draw('mpl')

# Functions
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


# Gen cost function
def generator_cost(gen_params, disc_params, gen_disc_circuit):
    curr_params = np.append(disc_params.detach().numpy(), gen_params.detach().numpy())
    state_probs = Statevector(gen_disc_circuit.assign_parameters(curr_params)).probabilities()
    prob_fake_true = np.sum(state_probs[0b100:])
    cost = -prob_fake_true
    return torch.tensor(cost, requires_grad=True)


# Discrim cost function
def discriminator_cost(disc_params, gen_params, gen_disc_circuit, real_disc_circuit):
    curr_params = np.append(disc_params.detach().numpy(), gen_params.detach().numpy())
    gendisc_probs = Statevector(gen_disc_circuit.assign_parameters(curr_params)).probabilities()
    realdisc_probs = Statevector(real_disc_circuit.assign_parameters(disc_params.detach().numpy())).probabilities()
    prob_fake_true = np.sum(gendisc_probs[0b100:])
    prob_real_true = np.sum(realdisc_probs[0b100:])
    cost = prob_fake_true - prob_real_true
    return torch.tensor(cost, requires_grad=True)


# Neural Network initialization
sampler = Sampler()
gen_qnn = SamplerQNN(circuit=gen_disc_circuit, sampler=sampler, input_params=generator.parameters,
                     weight_params=discriminator.parameters, sparse=None)
disc_fake_qnn = SamplerQNN(circuit=gen_disc_circuit, sampler=sampler, input_params=discriminator.parameters,
                           weight_params=generator.parameters, sparse=None)
disc_real_qnn = SamplerQNN(circuit=real_disc_circuit, input_params=[], weight_params=discriminator.parameters,
                           sampler=sampler, sparse=None)



# Initialization
init_gen_params = np.random.uniform(low=-np.pi, high=np.pi, size=(N_GPARAMS,))
init_disc_params = np.random.uniform(low=-np.pi, high=np.pi, size=(N_DPARAMS,))

gen_params = torch.tensor(init_gen_params, requires_grad=True)
disc_params = torch.tensor(init_disc_params, requires_grad=True)

init_gen_circuit = generator.assign_parameters(init_gen_params)
init_prob_dict = Statevector(init_gen_circuit).probabilities_dict()

fig, ax1 = plt.subplots(1, 1, sharey=True)
ax1.set_title("Initial generator distribution")
plot_histogram(init_prob_dict, ax=ax1)

generator_optimizer = optim.Adam([gen_params], lr=0.02)
discriminator_optimizer = optim.Adam([disc_params], lr=0.02)

best_gen_params = gen_params.clone().detach().requires_grad_(True)
gloss = []
dloss = []
kl_div = []


#ML Training
TABLE_HEADERS = "Epoch | Generator cost | Discriminator cost | KL Div. |"
print(TABLE_HEADERS)

for epoch in range(100):
    D_STEPS = 5  
    for disc_train_step in range(D_STEPS):
        d_fake = torch.tensor(disc_fake_qnn.backward(gen_params.detach().numpy(), disc_params.detach().numpy())[1]).to_dense()[0, 0b100:]
        d_fake = torch.sum(d_fake, axis=0)
        d_real = torch.tensor(disc_real_qnn.backward([], disc_params.detach().numpy())[1]).to_dense()[0, 0b100:]
        d_real = torch.sum(d_real, axis=0)
        grad_dcost = [d_fake[i] - d_real[i] for i in range(N_DPARAMS)]
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
    real_prob_dict = Statevector(real_circuit).probabilities_dict()
    current_kl = calculate_kl_div(gen_prob_dict, real_prob_dict)
    kl_div.append(current_kl)

    if np.min(kl_div) == current_kl:
        best_gen_params = pickle.loads(pickle.dumps(gen_params.detach().numpy()))

    if epoch % 10 == 0:
        for header, val in zip(TABLE_HEADERS.split('|'), (epoch, gloss[-1], dloss[-1], kl_div[-1])):
            print(f"{val:.3g} ".rjust(len(header)), end="|")
        print()
