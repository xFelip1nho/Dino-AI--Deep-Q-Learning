import os
import time
import random
import numpy as np
from collections import deque

import pygame
import torch
import torch.nn as nn
import torch.optim as optim

from Dino_GAME import DinoEnv

# ===========================================================
# CONFIGURAÇÕES DO TREINAMENTO
# ===========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Se houver GPU disponível, usa CUDA. Senão, usa CPU.

SAVE_PATH = "dino_dqn.pth"        # Caminho para salvar o modelo atual
BEST_PATH = "best_dino_dqn.pth"   # Modelo com melhor score
STATE_PATH = "training_state.npz" # Histórico de scores e recompensas

EPISODES = 150000          # Número total de episódios para treinar
REPLAY_SIZE = 150000       # Capacidade do Replay Buffer
BATCH_SIZE = 128           # Tamanho do batch para treino
MIN_REPLAY = 2000          # Mínimo de amostras antes de começar treinar

GAMMA = 0.99               # Fator de desconto do Q-Learning
LR = 5e-4                  # Taxa de aprendizado
FRAME_SKIP = 1             # Executa mesma ação X frames
TAU = 0.005                # Taxa de atualização da rede alvo
SAVE_EVERY = 500           # Frequência para salvar checkpoints


# ===========================================================
# REDE NEURAL (Deep Q-Network)
# ===========================================================

class DQN(nn.Module):
    def __init__(self, inp=8, hid=256, out=3):
        # 8 entradas = estado do ambiente
        # 3 saídas = ações (correr, pular, abaixar)
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, out)
        )

    def forward(self, x):
        return self.net(x)


# ===========================================================
# REPLAY BUFFER (Memória de experiência)
# ===========================================================

class ReplayBuffer:
    def __init__(self, capacity):
        # Usamos deque pelo desempenho
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        # Armazena: estado, ação, recompensa, próximo estado, done
        self.buffer.append(args)

    def sample(self, batch):
        # Amostra aleatória para treino
        s, a, r, ns, d = zip(*random.sample(self.buffer, batch))
        return map(np.array, (s, a, r, ns, d))

    def __len__(self):
        return len(self.buffer)


# ===========================================================
# FUNÇÕES AUXILIARES
# ===========================================================

def soft_update(target, policy):
    # Atualiza a rede-alvo suavemente para evitar instabilidade
    for tp, pp in zip(target.parameters(), policy.parameters()):
        tp.data.mul_(1 - TAU).add_(TAU * pp.data)

def tensor(x):
    # Converte para tensor PyTorch com suporte CUDA
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)

def load_or_init_hist():
    # Carrega histórico de treinamento caso exista
    if os.path.exists(STATE_PATH):
        data = np.load(STATE_PATH)
        return list(data["rewards"]), list(data["scores"])
    return [], []

def load_or_init_epsilon(episode):
    # Atualiza epsilon de forma decrescente (epsilon-decay)
    eps = max(0.05, 1.0 * (0.99985 ** episode))
    return eps


# ===========================================================
# FUNÇÃO PRINCIPAL DE TREINAMENTO
# ===========================================================

def treinar():
    env = DinoEnv(render=False)  # Ambiente sem renderização para maior velocidade

    policy = DQN().to(DEVICE)
    target = DQN().to(DEVICE)

    # Carrega modelo salvo (se existir)
    if os.path.exists(SAVE_PATH):
        policy.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        target.load_state_dict(policy.state_dict())

    optimizer = optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)
    replay = ReplayBuffer(REPLAY_SIZE)

    rewards, scores = load_or_init_hist()   # histórico de aprendizado
    episode = len(scores)
    epsilon = load_or_init_epsilon(episode) # inicializa epsilon
    best_score = max(scores) if scores else -1

    print("\n==== TREINAMENTO INICIADO ====\n")

    # LOOP PRINCIPAL DOS EPISÓDIOS
    while episode < EPISODES:
        episode += 1
        state = env.reset()
        done = False
        total_reward = 0
        score = 0

        # LOOP DO JOGO
        while not done:

            # Decide ação baseado em política epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, 2)  # ação aleatória
            else:
                with torch.no_grad():
                    q = policy(tensor(state).unsqueeze(0))[0]
                    action = int(q.argmax().item())  # melhor ação segundo a rede

            # Executa ação no ambiente (com frame-skip)
            cum_reward = 0
            for _ in range(FRAME_SKIP):
                next_state, r, done, info = env.step(action)
                cum_reward += r
                if done:
                    break

            replay.push(state, action, cum_reward, next_state, float(done))
            state = next_state
            total_reward += cum_reward
            score = info.get("score", score)

            # --------- TREINAMENTO DA REDE NEURAL ---------
            if len(replay) >= MIN_REPLAY:

                # Amostra batch
                s, a, r, ns, d = replay.sample(BATCH_SIZE)
                s, a, r, ns, d = map(tensor, (s, a, r, ns, d))

                # Q(s,a)
                q_values = policy(s).gather(1, a.long().unsqueeze(1)).squeeze(1)

                # Q-target
                with torch.no_grad():
                    next_q = target(ns).max(1)[0]
                    target_q = r + (1 - d) * GAMMA * next_q

                # Loss
                loss = nn.functional.mse_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 10)
                optimizer.step()

                # Atualiza rede alvo
                soft_update(target, policy)

        # ========== Ao final do episódio ==========
        rewards.append(total_reward)
        scores.append(score)
        epsilon = max(0.05, epsilon * 0.99985)

        print(f"EP {episode} | Score: {score} | Best: {best_score}")

        # Salva melhor modelo
        if score > best_score:
            best_score = score
            torch.save(policy.state_dict(), BEST_PATH)
            print("🔥 Novo melhor modelo salvo!")

        # Salva checkpoints
        if episode % SAVE_EVERY == 0:
            torch.save(policy.state_dict(), SAVE_PATH)
            np.savez(STATE_PATH, rewards=np.array(rewards), scores=np.array(scores))
            print("💾 Checkpoint salvo.")

    print("\nTREINAMENTO ENCERRADO!\n")


# ===========================================================
# FUNÇÃO PARA JOGAR COM O MODELO TREINADO
# ===========================================================

def jogar():
    if not os.path.exists(BEST_PATH):
        print("Nenhum modelo encontrado! Treine primeiro.")
        return

    model = DQN().to(DEVICE)
    model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
    model.eval()

    env = DinoEnv(render=True)

    while True:
        state = env.reset()
        done = False
        score = 0

        while not done:
            # Permite fechar janela
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            # IA escolhe melhor ação sempre (sem exploração)
            with torch.no_grad():
                q = model(tensor(state).unsqueeze(0))[0]
                action = int(q.argmax().item())

            state, _, done, info = env.step(action)
            score = info.get("score", score)

            env.render()

        print(f"Score: {score}")


# ===========================================================
# MENU DE EXECUÇÃO VIA TERMINAL
# ===========================================================

def menu():
    print("\n==== DINO IA ====")
    print("1 - Treinar IA")
    print("2 - Jogar com IA")
    print("3 - Sair")

    op = input("Opção: ")

    match op:
        case "1": treinar()
        case "2": jogar()
        case "3": print("Saindo...")
        case _: 
            print("Opção inválida.")
            menu()


# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":
    menu()
