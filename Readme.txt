# 🦖 Dino IA – Agente DQN jogando Chrome Dino Game

Este projeto implementa uma **Inteligência Artificial que aprende a jogar o jogo do Dino do Google Chrome**, utilizando **Reinforcement Learning (DQN)** e **PyTorch**.  
O ambiente do jogo foi recriado em **Pygame**, com detecção de obstáculos, estados normalizados e controle total sobre o personagem.

---

## 🚀 Tecnologias usadas
- **Python 3.10+**
- **Pygame 2.6.1**
- **PyTorch 2.0+**
- **NumPy**
- **Matplotlib**
- **Tqdm**

---

## 🧠 Como a IA aprende?

A IA usa **Deep Q-Learning**, onde:

- Observa o estado do jogo (distância do obstáculo, altura do Dino, velocidade, tipo do obstáculo, etc.)
- Escolhe ações:  
  **0 = correr**, **1 = pular**, **2 = abaixar**
- Recebe recompensas:
  - +1 para cada obstáculo evitado  
  - -10 se colidir  
- Atualiza os pesos da rede neural para melhorar suas decisões
- Possui:
  - Replay Buffer
  - Rede alvo (Target Network)
  - Epsilon Decay para explorar → explorar menos com o tempo