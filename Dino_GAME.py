import pygame
import os
import random
import numpy as np

# -----------------------------
# CONFIGURAÇÕES DE TELA E ASSETS
# -----------------------------
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100

# Carrega imagens do Dino (correndo)
RUNNING = [
    pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
    pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))
]

# Imagem do Dino pulando
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))

# Imagens do Dino abaixado
DUCKING = [
    pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
    pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))
]

# Cactos pequenos
SMALL_CACTUS = [
    pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
    pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
    pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))
]

# Cactos grandes
LARGE_CACTUS = [
    pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
    pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
    pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))
]

# Pássaro (obstáculo aéreo)
BIRD = [
    pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
    pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))
]

# Fundo da pista
BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


# ---------------------------------------
# CLASSE PRINCIPAL: DINOSAUR (O PERSONAGEM)
# ---------------------------------------
class Dinosaur:
    # Posições e constantes do personagem
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5

    def __init__(self):
        # Carrega animações
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        # Estados possíveis do Dino
        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        # Controle de animação
        self.step_index = 0
        self.vel_y = 0.0  # velocidade do pulo

        # Imagem inicial
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    # Atualiza o Dino com base na ação recebida da IA
    def update(self, action):

        # Evita pulo duplo
        if action == 1 and self.dino_jump:
            action = 0  

        # Ação: pular
        if action == 1 and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
            self.vel_y = self.JUMP_VEL

        # Ação: abaixar
        elif action == 2 and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False

        # Ação padrão: correr
        elif not self.dino_jump:
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

        # Chama animação apropriada
        if self.dino_duck: self.duck()
        if self.dino_run: self.run()
        if self.dino_jump: self.jump()

        # Reseta animação
        if self.step_index >= 10:
            self.step_index = 0

    # Animação de abaixar
    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    # Animação de correr
    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    # Física do pulo
    def jump(self):
        self.image = self.jump_img

        if self.dino_jump:
            self.dino_rect.y -= self.vel_y * 4
            self.vel_y -= 0.8  # gravidade

        # Limita altura máxima
        if self.dino_rect.y < 120:
            self.dino_rect.y = 120
            self.vel_y = -abs(self.vel_y)

        # Volta ao chão
        if self.vel_y < -self.JUMP_VEL:
            self.dino_jump = False
            self.vel_y = 0.0
            if self.dino_rect.y > self.Y_POS:
                self.dino_rect.y = self.Y_POS

    # Desenha o Dino
    def draw(self, screen):
        screen.blit(self.image, (self.dino_rect.x, self.dino_rect.y))


# -------------------------------
# CLASSE BASE PARA OBSTÁCULOS
# -------------------------------
class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH  # nasce fora da tela à direita

    def update(self, game_speed):
        self.rect.x -= game_speed  # movimento lateral

    def draw(self, screen):
        screen.blit(self.image[self.type], self.rect)


# Tipos específicos de obstáculos
class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 300


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = 250
        self.index = 0

    # Animação de bater asas
    def draw(self, screen):
        if self.index >= 9:
            self.index = 0
        screen.blit(self.image[self.index // 5], self.rect)
        self.index += 1


# --------------------------------------------------------
# AMBIENTE DE JOGO (SEMELHANTE AO GYM DO REINFORCEMENT LEARNING)
# --------------------------------------------------------
class DinoEnv:
    def __init__(self, render=False):
        pygame.init()
        self.render_mode = render

        # Configura a janela
        if render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Dino - IA")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = pygame.time.Clock()

        self.reset()

    # Reinicia o episódio (jogo)
    def reset(self):
        self.player = Dinosaur()
        self.obstacles = []
        self.game_speed = 20.0
        self.points = 0
        self.done = False
        self.frame_count = 0
        return self._get_state()

    # Geração aleatória de obstáculos
    def _create_obstacle(self):
        choice = random.randint(0, 2)
        if choice == 0: self.obstacles.append(SmallCactus(SMALL_CACTUS))
        elif choice == 1: self.obstacles.append(LargeCactus(LARGE_CACTUS))
        else: self.obstacles.append(Bird(BIRD))

    # Converte o estado do jogo em vetor para IA
    def _get_state(self):
        if len(self.obstacles) == 0:
            # Sem obstáculo → usa valores padrão
            dist = SCREEN_WIDTH
            ow = 0
            oh = 0
            obst_type = -1
            obs_y = SCREEN_HEIGHT
        else:
            # Pega primeiro obstáculo
            obs = self.obstacles[0]
            dist = max(0.0, obs.rect.x - self.player.dino_rect.x)
            ow = obs.rect.width
            oh = obs.rect.height
            obs_y = obs.rect.y

            # Identifica tipo (cacto pequeno, grande, pássaro)
            if isinstance(obs, Bird):
                obst_type = 2
            else:
                obst_type = 0 if oh < 360 else 1

        # Normalização dos valores
        dist_norm = dist / SCREEN_WIDTH
        obs_h_norm = oh / SCREEN_HEIGHT
        obs_w_norm = ow / SCREEN_WIDTH
        obs_type_norm = (obst_type + 1) / 4.0
        speed_norm = self.game_speed / 50.0
        on_ground = 1.0 if not self.player.dino_jump else 0.0
        dino_y_norm = self.player.dino_rect.y / SCREEN_HEIGHT
        dino_vel_norm = self.player.vel_y / (self.player.JUMP_VEL if self.player.JUMP_VEL != 0 else 1)

        return np.array([
            dist_norm,
            obs_h_norm,
            obs_w_norm,
            obs_type_norm,
            speed_norm,
            on_ground,
            dino_y_norm,
            dino_vel_norm
        ], dtype=np.float32)

    # Executa um passo do ambiente (step)
    def step(self, action):

        # Segurança: impede pulo no ar
        if action == 1 and self.player.dino_jump:
            action = 0

        reward = 0.0
        self.player.update(action)
        self.frame_count += 1

        # Controla frequência de spawn
        spawn_interval = max(20, 50 - (self.game_speed - 20) * 2)

        if self.frame_count % int(spawn_interval) == 0:
            self._create_obstacle()

        # Atualiza obstáculos
        for obs in list(self.obstacles):
            obs.update(self.game_speed)

            # Se passar da tela → IA ganha recompensa
            if obs.rect.x < -obs.rect.width:
                try:
                    self.obstacles.remove(obs)
                    reward += 1.0
                except:
                    pass

        # Detecção de colisão
        if len(self.obstacles) > 0:
            if self.player.dino_rect.colliderect(self.obstacles[0].rect):
                reward -= 10.0
                self.done = True

        # Se sair do mapa verticalmente
        if self.player.dino_rect.y > SCREEN_HEIGHT:
            reward -= 10.0
            self.done = True

        self.points += 1
        self.game_speed += 0.005  # aceleração gradual

        next_state = self._get_state()
        return next_state, reward, self.done, {"score": self.points}

    # Renderização visual
    def render(self):
        if not self.render_mode:
            return

        # Possibilidade de fechar a janela
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((255, 255, 255))
        self.player.draw(self.screen)

        for obs in self.obstacles:
            obs.draw(self.screen)

        # Exibe pontuação
        font = pygame.font.SysFont("Comic Sans MS", 36)
        text_render = font.render(str(int(self.points)), True, (0, 0, 0))
        self.screen.blit(text_render, (950, 20))

        # Desenha o chão
        self.screen.blit(BG, (0, 380))
        pygame.display.update()
        self.clock.tick(30)

    # Fecha o ambiente
    def close(self):
        if self.render_mode:
            pygame.quit()
