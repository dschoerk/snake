
from dataclasses import dataclass
import random
import operator
import fractions

import numpy as np

@dataclass
class Point:
    x: int
    y: int

    def _add(a, b):
        """a + b"""
        return Point(a.x + b.x, a.y + b.y)  # _operator_fallback has already coerced the types appropriately

    __add__, __radd__ = fractions.Fraction._operator_fallbacks(_add, operator.add)

@dataclass
class GameState:
    # the snake is defined by the list of snake sections
    # the head is always at position 0
    body : list[Point]
    
    # direction is [0,3] for up, right, down, left
    direction : Point

    # location of food item
    food: Point

    # score
    reward: float

    # playing field
    field_size : Point

    steps: int = 0
    last_reward_change: int = 0  # step when the last reward change happened

@dataclass
class AgentAction:
    direction : int
    
class SnakeGame:
    def __init__(self, field_size = (20, 20), wnd=3):
        self.field_size = field_size
        self.wnd = wnd
        self.reset()

    def reset(self, random_field_size=False):
        if random_field_size:
            self.field_size = (random.randint(10, 30), random.randint(10, 30))

        self.gamestate = GameState(
            body = [Point(5, 5)],
            direction = Point(1,0),
            food = Point(random.randint(0, self.field_size[0]-1), random.randint(0, self.field_size[1]-1)),
            field_size = Point(x=self.field_size[0],y=self.field_size[1]),
            reward=0,
            steps=0,
            last_reward_change=0
        )

    def observation(self):
        w = (self.wnd - 1) // 2

        map = np.zeros((self.gamestate.field_size.y+w*2, self.gamestate.field_size.x+w*2), dtype=np.int32)
        map[0, :] = 1
        map[-1, :] = 1
        map[:, 0] = 1
        map[:, -1] = 1
        
        # map[self.gamestate.food.y, self.gamestate.food.x] = 1
        
        for b in self.gamestate.body:
            map[b.y+w, b.x+w] = 1

        h = self.gamestate.body[0]

        local_map = map[h.y : h.y+1+2*w, h.x : h.x+1+2*w]
        
        # if h.x > 0 and h.y > 0 and h.x < map.shape[1] and h.y < map.shape[0]:
        #     map[h.y, h.x] = 3

        # map_padded = np.zeros((map.shape[0]+2, map.shape[1]+2))
        # map_padded[:,:] = -1
        # map_padded[1:-1, 1:-1] = map

        # direction observation
        go_up = self.gamestate.food.y < h.y
        go_down = self.gamestate.food.y > h.y
        go_left = self.gamestate.food.x < h.x
        go_right = self.gamestate.food.x > h.x


        return np.append(np.array([go_up, go_down, go_left, go_right, self.gamestate.reward, len(self.gamestate.body)], dtype=np.int32), local_map.flatten())
    
    def draw(self, observation=None):
        map = np.zeros((self.gamestate.field_size.y, self.gamestate.field_size.x), dtype=np.str_)
        for b in self.gamestate.body[1:]:
            map[b.y, b.x] = 'o'
        h = self.gamestate.body[0]
        map[h.y, h.x] = '+'

        map[self.gamestate.food.y, self.gamestate.food.x] = 'x'

        print(map)
        

    def update(self, action : AgentAction):
        
        match(action):
            case 0:
                self.gamestate.direction = Point(0, -1)
            case 1: 
                self.gamestate.direction = Point(1, 0)
            case 2:
                self.gamestate.direction = Point(0, 1)
            case 3:
                self.gamestate.direction = Point(-1, 0)

        next_head_position = self.gamestate.body[0]._add(self.gamestate.direction)

        distance_to_food_new = abs(self.gamestate.food.x - next_head_position.x) + abs(self.gamestate.food.y - next_head_position.y)
        distance_to_food = abs(self.gamestate.food.x - self.gamestate.body[0].x) + abs(self.gamestate.food.y - self.gamestate.body[0].y)

        # check for collision
        collision = False
        if next_head_position.x < 0 or next_head_position.y < 0 or next_head_position.x >= self.gamestate.field_size.x or next_head_position.y >= self.gamestate.field_size.y:
            collision = True

        # check if we ate food
        ate_food = self.gamestate.body[0] == self.gamestate.food
        if not ate_food:
            self.gamestate.body.pop()
        else:
            self.gamestate.food = Point(random.randint(0, self.field_size[0]-1), random.randint(0, self.field_size[1]-1))

        for b in self.gamestate.body:
            if b == next_head_position:
                collision = True

        

        # update the snake head
        self.gamestate.body.insert(0, next_head_position)

        observation = self.observation()

        # reward = 0
        # if not ate_food:
        #     reward -= 0.3 # / len(self.gamestate.body)
        # else:
        #     reward += 10

        # self.gamestate.reward += reward

        terminated = False
        if collision:
          terminated = True

        distance_to_food_improved = distance_to_food_new <= distance_to_food

        old_reward = self.gamestate.reward

        reward = 1.0 + ate_food * 2 + (0 if distance_to_food_improved else -1.0)

        if collision:
            reward = -1

        self.gamestate.reward += reward

        if self.gamestate.reward != old_reward:
            self.gamestate.last_reward_change = self.gamestate.steps

        if abs(self.gamestate.last_reward_change - self.gamestate.steps) > 50 * len(self.gamestate.body):
            terminated = True

        # if self.gamestate.reward < 0:
        #     terminated = True

        self.gamestate.steps += 1

        return observation, reward, terminated

