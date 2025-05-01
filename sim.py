import cv2
import numpy as np
import random
import os
import uuid

def create_randomized_simulation(output_dir="simulations"):
    width, height = random.choice([(640, 480), (800, 600), (1280, 720)])
    fps = 30
    duration = random.randint(5, 15)
    num_frames = int(duration * fps)
    gravity = random.uniform(0.5, 1.2)
    damping = random.uniform(0.5, 0.65)
    num_balls = random.randint(1, 5)
    
    filename = f"{output_dir}/simulation_{uuid.uuid4().hex[:8]}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    def random_color():
        return tuple(random.randint(0, 255) for _ in range(3))

    balls = []
    for _ in range(num_balls):
        radius = random.randint(10, 40)
        ball = {
            'x': random.randint(radius, width - radius),
            'y': random.randint(radius, height - radius),
            'radius': radius,
            'speed_x': random.uniform(-5, 5),
            'color': random_color(),
            'speed_y': random.uniform(-5, 0),
            'bounces': 0
        }
        balls.append(ball)

    num_obstacles = random.randint(0, 3)
    obstacles = []
    for _ in range(num_obstacles):
        ow, oh = random.randint(80, 150), random.randint(30, 70)
        ox, oy = random.randint(0, width - ow), random.randint(0, height - oh)
        obstacles.append((ox, oy, ow, oh))

    for _ in range(num_frames):
        # Create a random background with gradients and noise
        # background = np.zeros((height, width, 3), dtype=np.uint8)
        # for i in range(0, height, 50):
        #     for j in range(0, width, 50):
        #         color = tuple(random.randint(50, 200) for _ in range(3))
        #         cv2.rectangle(background, (j, i), (j + 50, i + 50), color, -1)

        # Reduce the noise level
        # noise = np.random.normal(0, 1, (height, width, 3)).astype(np.uint8)
        # background = cv2.add(background, noise)

        frame = np.zeros((height, width, 3), dtype=np.uint8)

        for ox, oy, ow, oh in obstacles:
            cv2.rectangle(frame, (ox, oy), (ox + ow, oh + oy), (128, 128, 128), -1)

        for ball in balls:
            ball['x'] += ball['speed_x']
            ball['y'] += ball['speed_y']
            ball['speed_y'] += gravity

            if ball['x'] + ball['radius'] > width or ball['x'] - ball['radius'] < 0:
                ball['speed_x'] *= -1
            if ball['y'] + ball['radius'] > height:
                ball['y'] = height - ball['radius']
                ball['speed_y'] *= -damping
                ball['bounces'] += 1
            if ball['y'] - ball['radius'] < 0:
                ball['speed_y'] *= -1

            for ox, oy, ow, oh in obstacles:
                if (ox < ball['x'] < ox + ow) and (oy < ball['y'] + ball['radius'] < oy + oh):
                    ball['speed_y'] *= -damping
                    ball['bounces'] += 1

            # Create a gradient for the ball
            ball_mask = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.circle(ball_mask, (int(ball['x']), int(ball['y'])), ball['radius'], ball['color'], -1)
            gradient = cv2.GaussianBlur(ball_mask, (15, 15), 0)
            frame = cv2.addWeighted(frame, 1, gradient, 0.5, 0)

        out.write(frame)

    out.release()
    print(f"Randomized simulation saved to {filename}")

if __name__ == "__main__":
    os.makedirs("simulations", exist_ok=True)
    for _ in range(1):  
        create_randomized_simulation()