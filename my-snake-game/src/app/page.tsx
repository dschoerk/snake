'use client';


import React, { useState, useEffect, useCallback } from 'react';
import * as ort from 'onnxruntime-web';

const GRID_SIZE = 20;
const CANVAS_SIZE = 400;
const INITIAL_SNAKE = [{ x: 10, y: 10 }];
const INITIAL_FOOD = { x: 15, y: 15 };
const INITIAL_DIRECTION = { x: 0, y: 1 };

export default function SnakeGame() {
  const [snake, setSnake] = useState(INITIAL_SNAKE);
  const [food, setFood] = useState(INITIAL_FOOD);
  const [direction, setDirection] = useState(INITIAL_DIRECTION);
  const [gameOver, setGameOver] = useState(false);
  const [score, setScore] = useState(0);
  const [gameStarted, setGameStarted] = useState(false);
  const ortSessionRef = React.useRef(null);

  // Generate random food position not overlapping the snake
  const generateFood = useCallback((currentSnake: {x: number, y: number}[]) => {
    const gridW = CANVAS_SIZE / GRID_SIZE;
    const gridH = CANVAS_SIZE / GRID_SIZE;
    const occupied = new Set(currentSnake.map(s => `${s.x},${s.y}`));
    let x: number, y: number;
    do {
      x = Math.floor(Math.random() * gridW);
      y = Math.floor(Math.random() * gridH);
    } while (occupied.has(`${x},${y}`));
    return { x, y };
  }, []);

  // Check collision with walls or self
  const checkCollision = useCallback((head, snakeArray) => {
    // Wall collision
    if (head.x < 0 || head.x >= CANVAS_SIZE / GRID_SIZE || head.y < 0 || head.y >= CANVAS_SIZE / GRID_SIZE) {
      return true;
    }
    // Self collision
    for (let segment of snakeArray) {
      if (head.x === segment.x && head.y === segment.y) {
        return true;
      }
    }
    return false;
  }, []);

  // Game loop
  const moveSnake = useCallback(async () => {
    if (!gameStarted || gameOver) return;

    async function runModel(inputArray) {
      if (!ortSessionRef.current) {
        ortSessionRef.current = await ort.InferenceSession.create('/dqn_model.onnx');
      }
      const input = new ort.Tensor('float32', Float32Array.from(inputArray), [1, inputArray.length]);
      const feeds = { "observations": input };
      return ortSessionRef.current.run(feeds);
    }

    // Match Python game.py observation():
    // food(4) + danger(3) + direction(4) + body_len(1)
    const head = snake[0];
    const gridW = CANVAS_SIZE / GRID_SIZE;
    const gridH = CANVAS_SIZE / GRID_SIZE;

    const go_up    = food.y < head.y ? 1 : 0;
    const go_down  = food.y > head.y ? 1 : 0;
    const go_left  = food.x < head.x ? 1 : 0;
    const go_right = food.x > head.x ? 1 : 0;

    const bodySet = new Set(snake.map(s => `${s.x},${s.y}`));
    function isDanger(pt) {
      return pt.x < 0 || pt.y < 0 || pt.x >= gridW || pt.y >= gridH || bodySet.has(`${pt.x},${pt.y}`);
    }

    const d = direction;
    const danger_straight = isDanger({ x: head.x + d.x,  y: head.y + d.y  }) ? 1 : 0;
    const danger_left     = isDanger({ x: head.x - d.y,  y: head.y + d.x  }) ? 1 : 0;
    const danger_right    = isDanger({ x: head.x + d.y,  y: head.y - d.x  }) ? 1 : 0;

    const dir_right = d.x === 1  ? 1 : 0;
    const dir_left  = d.x === -1 ? 1 : 0;
    const dir_down  = d.y === 1  ? 1 : 0;
    const dir_up    = d.y === -1 ? 1 : 0;

    function argmax(array) {
      let maxIdx = 0;
      let maxVal = array[0];
      for (let i = 1; i < array.length; i++) {
        if (array[i] > maxVal) {
          maxVal = array[i];
          maxIdx = i;
        }
      }
      return maxIdx;
    }

    // 7x7 local grid centered on head: 1=wall/body, -1=food, 0=empty
    const localGrid = [];
    for (let row = -3; row <= 3; row++) {
      for (let col = -3; col <= 3; col++) {
        const cx = head.x + col;
        const cy = head.y + row;
        if (cx < 0 || cy < 0 || cx >= gridW || cy >= gridH) {
          localGrid.push(1.0);   // wall
        } else if (cx === food.x && cy === food.y) {
          localGrid.push(-1.0);  // food
        } else if (bodySet.has(`${cx},${cy}`)) {
          localGrid.push(1.0);   // body
        } else {
          localGrid.push(0.0);   // empty
        }
      }
    }

    var model_output = await runModel([
      go_up, go_down, go_left, go_right,
      danger_straight, danger_left, danger_right,
      dir_right, dir_left, dir_down, dir_up,
      snake.length / (gridW * gridH),
      ...localGrid
    ]);
    const qValues = model_output["q_values"] ?? Object.values(model_output)[0];
    const pred_dir = argmax(qValues.cpuData);
    console.log("Model output:", pred_dir);

    // Compute new direction locally to avoid stale closure in setSnake
    const dirMap: Record<number, {x: number, y: number}> = {
      0: { x: 0, y: -1 }, 1: { x: 1, y: 0 }, 2: { x: 0, y: 1 }, 3: { x: -1, y: 0 }
    };
    let newDirection = dirMap[pred_dir] ?? direction;
    // Prevent 180-degree reversal when snake has a body
    if (snake.length > 1 && newDirection.x + direction.x === 0 && newDirection.y + direction.y === 0) {
      newDirection = direction;
    }
    setDirection(newDirection);

    setSnake(currentSnake => {
      const newSnake = [...currentSnake];
      const head = { x: newSnake[0].x + newDirection.x, y: newSnake[0].y + newDirection.y };

      // Match Python game.py: remove tail before self-collision check (when not eating food).
      // This allows the head to legally move into the space the tail is vacating.
      const willEatFood = head.x === food.x && head.y === food.y;
      const bodyForCollision = willEatFood ? newSnake : newSnake.slice(0, -1);

      if (checkCollision(head, bodyForCollision)) {
        setGameOver(true);
        return currentSnake;
      }

      newSnake.unshift(head);

      // Check if food is eaten
      if (willEatFood) {
        setScore(s => s + 10);
        setFood(generateFood(newSnake));
      } else {
        newSnake.pop();
      }

      return newSnake;
    });
  }, [direction, snake, food, gameStarted, gameOver, checkCollision, generateFood]);

  // Handle keyboard input
  const handleKeyPress = useCallback(async e => {
    if (!gameStarted && e.key === ' ') {
      setGameStarted(true);
      return;
    }

    if (gameOver && e.key === ' ') {
      // Restart game
      setSnake(INITIAL_SNAKE);
      setFood(generateFood(INITIAL_SNAKE));
      setDirection(INITIAL_DIRECTION);
      setGameOver(false);
      setScore(0);
      setGameStarted(true);
      return;
    }

    if (!gameStarted || gameOver) return;

    switch (e.key) {
      case 'ArrowUp':
        if (direction.y === 0) setDirection({ x: 0, y: -1 });
        break;
      case 'ArrowDown':
        if (direction.y === 0) setDirection({ x: 0, y: 1 });
        break;
      case 'ArrowLeft':
        if (direction.x === 0) setDirection({ x: -1, y: 0 });
        break;
      case 'ArrowRight':
        if (direction.x === 0) setDirection({ x: 1, y: 0 });
        break;
    }
    
  }, [direction, gameStarted, gameOver]);

  // Set up game loop
  useEffect(() => {
    const gameInterval = setInterval(moveSnake, 150);
    return () => clearInterval(gameInterval);
  }, [moveSnake]);

  // Set up keyboard listener
  useEffect(() => {
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [handleKeyPress]);

  // Focus on mount to capture keyboard events
  useEffect(() => {
    window.focus();
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-4">
      <div className="text-center mb-6">
        <h1 className="text-4xl font-bold mb-2 text-green-400">Snake Game</h1>
        <p className="text-lg mb-2">Score: {score}</p>
        {!gameStarted && !gameOver && (
          <p className="text-sm text-gray-300">Press SPACE to start</p>
        )}
        {gameOver && (
          <div className="text-center">
            <p className="text-xl text-red-400 mb-2">Game Over!</p>
            <p className="text-sm text-gray-300">Press SPACE to restart</p>
          </div>
        )}
      </div>

      <div className="relative">
        <div 
          className="border-2 border-gray-600 bg-black relative"
          style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
        >
          {/* Snake */}
          {snake.map((segment, index) => (
            <div
              key={index}
              className={`absolute ${index === 0 ? 'bg-green-300' : 'bg-green-500'}`}
              style={{
                left: segment.x * GRID_SIZE,
                top: segment.y * GRID_SIZE,
                width: GRID_SIZE - 1,
                height: GRID_SIZE - 1,
              }}
            />
          ))}

          {/* Food */}
          <div
            className="absolute bg-red-500 rounded-full"
            style={{
              left: food.x * GRID_SIZE,
              top: food.y * GRID_SIZE,
              width: GRID_SIZE - 1,
              height: GRID_SIZE - 1,
            }}
          />
        </div>
      </div>

      <div className="mt-6 text-center">
        <p className="text-sm text-gray-300 mb-2">Use arrow keys to control the snake</p>
        <div className="grid grid-cols-3 gap-1 w-32 mx-auto">
          <div></div>
          <button 
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded text-xs"
            onClick={() => handleKeyPress({ key: 'ArrowUp' })}
          >
            ↑
          </button>
          <div></div>
          <button 
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded text-xs"
            onClick={() => handleKeyPress({ key: 'ArrowLeft' })}
          >
            ←
          </button>
          <div></div>
          <button 
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded text-xs"
            onClick={() => handleKeyPress({ key: 'ArrowRight' })}
          >
            →
          </button>
          <div></div>
          <button 
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded text-xs"
            onClick={() => handleKeyPress({ key: 'ArrowDown' })}
          >
            ↓
          </button>
          <div></div>
        </div>
        <p className="text-xs text-gray-400 mt-2">Or click the arrow buttons above</p>
      </div>
    </div>
  );
}