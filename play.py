import random
import sys

import torch
from game import Point, SnakeGame
import pygame

from network import DQN, EnsembleDQN

class PygameSnakeGame:
    def __init__(self, field_size=(10, 10), cell_size=25, wnd=7):
        pygame.init()
        
        self.field_size = field_size
        self.cell_size = cell_size
        self.game = SnakeGame(field_size, wnd=wnd)
        self.wnd = wnd

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.DARK_GREEN = (0, 150, 0)
        
        # Screen setup
        self.screen_width = field_size[0] * cell_size
        self.screen_height = field_size[1] * cell_size + 50  # Extra space for score
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Snake Game")
        
        # Font for score display
        self.font = pygame.font.Font(None, 36)
        
        # Game timing
        self.clock = pygame.time.Clock()
        self.game_speed = 8
        
        # Current direction to prevent immediate reverse
        self.current_direction = 1  # Start moving right

        device = torch.device('cpu')
        self.policy_net = DQN(6 + self.wnd*self.wnd, 4).to(device)
        self.policy_net.load_state_dict(torch.load('best.pt', weights_only=True))
        self.policy_net.eval()
        
    def handle_input(self):
        if True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and self.current_direction != 2:
                        self.current_direction = 0
                    elif event.key == pygame.K_RIGHT and self.current_direction != 3:
                        self.current_direction = 1
                    elif event.key == pygame.K_DOWN and self.current_direction != 0:
                        self.current_direction = 2
                    elif event.key == pygame.K_LEFT and self.current_direction != 1:
                        self.current_direction = 3
                    elif event.key == pygame.K_r:
                        self.game.reset()
                        self.policy_net.load_state_dict(torch.load('best.pt', weights_only=True))
                        self.current_direction = 1

        # try:
        #     input = torch.tensor(self.game.observation(), dtype=torch.float32)
        #     input = input.unsqueeze(0)
        #     pred = self.policy_net(input)
        #     self.current_direction = pred.max(1).indices.view(1, 1)
        # except Exception as e:
        #     print(f"Error during prediction: {e}")
        #     self.game.reset()
                    
        return True
    
    def draw(self):
        self.screen.fill(self.BLACK)
        
        # Draw game area border
        pygame.draw.rect(self.screen, self.WHITE, 
                        (0, 0, self.screen_width, self.screen_height - 50), 2)
        
        # Draw snake
        for i, body_part in enumerate(self.game.gamestate.body):
            x = body_part.x * self.cell_size
            y = body_part.y * self.cell_size
            
            if i == 0:  # Head
                pygame.draw.rect(self.screen, self.GREEN, 
                               (x + 2, y + 2, self.cell_size - 4, self.cell_size - 4))
                # Draw eyes
                pygame.draw.circle(self.screen, self.BLACK, 
                                 (x + 6, y + 6), 3)
                pygame.draw.circle(self.screen, self.BLACK, 
                                 (x + self.cell_size - 6, y + 6), 3)
            else:  # Body
                pygame.draw.rect(self.screen, self.DARK_GREEN, 
                               (x + 2, y + 2, self.cell_size - 4, self.cell_size - 4))
        
        # Draw food
        food_x = self.game.gamestate.food.x * self.cell_size
        food_y = self.game.gamestate.food.y * self.cell_size
        pygame.draw.circle(self.screen, self.RED, 
                          (food_x + self.cell_size // 2, food_y + self.cell_size // 2), 
                          self.cell_size // 2 - 2)
        
        # Draw score
        score_text = self.font.render(f"Score: {int(self.game.gamestate.reward)}", True, self.WHITE)
        self.screen.blit(score_text, (10, self.screen_height - 40))
        
        # Draw length
        length_text = self.font.render(f"Length: {len(self.game.gamestate.body)}", True, self.WHITE)
        self.screen.blit(length_text, (200, self.screen_height - 40))
        
        pygame.display.flip()
    
    def show_game_over(self):
        self.screen.fill(self.BLACK)
        
        game_over_text = self.font.render("Game Over!", True, self.RED)
        score_text = self.font.render(f"Final Score: {int(self.game.gamestate.reward)}", True, self.WHITE)
        restart_text = self.font.render("Press R to restart or ESC to quit", True, self.WHITE)
        
        # Center the text
        game_over_rect = game_over_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        score_rect = score_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        restart_rect = restart_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 50))
        
        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(score_text, score_rect)
        self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()


    
    def run(self):
        running = True
        game_over = False
        
        while running:
            # Handle input
            if not self.handle_input():
                break
            
            # Handle game over state
            if game_over:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_r]:
                    self.game.reset()
                    self.current_direction = 1
                    game_over = False
                elif keys[pygame.K_ESCAPE]:
                    break
                    
                self.show_game_over()
                self.clock.tick(60)

                # self.policy_net.load_state_dict(torch.load('best.pt', weights_only=True))
                # self.game.reset()
                # game_over = False
                continue
            
            # Update game
            observation, reward, collision = self.game.update(self.current_direction)
            
            if collision:
                game_over = True
            
            # Draw everything
            self.draw()
            
            # Control game speed
            self.clock.tick(self.game_speed)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = PygameSnakeGame(field_size=(20, 20), cell_size=25)
    game.run()