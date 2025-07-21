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

  // Generate random food position
  const generateFood = useCallback(() => {
    const x = Math.floor(Math.random() * (CANVAS_SIZE / GRID_SIZE));
    const y = Math.floor(Math.random() * (CANVAS_SIZE / GRID_SIZE));
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
      const session = await ort.InferenceSession.create('/dqn_model.onnx');
      const input = new ort.Tensor('float32', Float32Array.from(inputArray), [1, inputArray.length]);

      
      const feeds = { "onnx::Gemm_0": input }; // Replace 'input' with your model's input name
      const results = await session.run(feeds);
      // Access output: results[Object.keys(results)[0]]
      return results;
    }

    // [go_up, go_down, go_left, go_right, self.gamestate.reward, len(self.gamestate.body)]

    const go_up = snake[0].y < food.y ? 1 : 0;
    const go_down = snake[0].y > food.y ? 1 : 0;
    const go_left = snake[0].x < food.x ? 1 : 0;
    const go_right = snake[0].x > food.x ? 1 : 0

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


    var model_output = await runModel([go_up, go_down, go_left, go_right, snake.length]);
    const pred_dir = argmax(model_output[11].cpuData)
    console.log("Model output:", pred_dir);

    if(pred_dir === 0) {
      setDirection({ x: 0, y: -1 });
    } else if(pred_dir === 1) {
      setDirection({ x: 1, y: 0 });
    } else if(pred_dir === 2) {
      setDirection({ x: 0, y: 1 });
    } else if(pred_dir === 3) {
      setDirection({ x: -1, y: 0 });
    }

    setSnake(currentSnake => {
      const newSnake = [...currentSnake];
      const head = { x: newSnake[0].x + direction.x, y: newSnake[0].y + direction.y };

      // Check collision
      if (checkCollision(head, newSnake)) {
        setGameOver(true);
        return currentSnake;
      }

      newSnake.unshift(head);

      // Check if food is eaten
      if (head.x === food.x && head.y === food.y) {
        setScore(s => s + 10);
        setFood(generateFood());
      } else {
        newSnake.pop();
      }

      return newSnake;
    });
  }, [direction, food, gameStarted, gameOver, checkCollision, generateFood]);

  // Handle keyboard input
  const handleKeyPress = useCallback(async e => {
    if (!gameStarted && e.key === ' ') {
      setGameStarted(true);
      return;
    }

    if (gameOver && e.key === ' ') {
      // Restart game
      setSnake(INITIAL_SNAKE);
      setFood(INITIAL_FOOD);
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