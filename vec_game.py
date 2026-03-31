"""Vectorized Snake environments running entirely on GPU via torch tensors."""

import torch

# Action mapping: 0=up, 1=right, 2=down, 3=left
DIR_X = torch.tensor([0, 1, 0, -1], dtype=torch.int32)
DIR_Y = torch.tensor([-1, 0, 1, 0], dtype=torch.int32)
OPPOSITE = torch.tensor([2, 3, 0, 1], dtype=torch.int32)

GRID_RADIUS = 3
GRID_SIZE = 2 * GRID_RADIUS + 1  # 7
N_SCALAR = 12
OBS_SIZE = N_SCALAR + GRID_SIZE * GRID_SIZE  # 61

# Precompute grid offsets (49,)
_row_offsets = torch.arange(-GRID_RADIUS, GRID_RADIUS + 1).repeat_interleave(GRID_SIZE)
_col_offsets = torch.arange(-GRID_RADIUS, GRID_RADIUS + 1).repeat(GRID_SIZE)


class VecSnakeGame:
    """N snake environments stepped in parallel on GPU."""

    def __init__(self, n_envs: int, device: torch.device, max_field: int = 30, min_field: int = 10):
        self.n = n_envs
        self.device = device
        self.max_field = max_field
        self.min_field = min_field

        # Move lookup tables to device
        self.DIR_X = DIR_X.to(device)
        self.DIR_Y = DIR_Y.to(device)
        self.OPPOSITE = OPPOSITE.to(device)
        self.ROW_OFFSETS = _row_offsets.to(device)
        self.COL_OFFSETS = _col_offsets.to(device)

        # Body tracked via visit-time grid: occupied if visit_grid[y,x] > step - body_len
        self.visit_grid = torch.zeros(n_envs, max_field, max_field, dtype=torch.int32, device=device)
        self.head_x = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self.head_y = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self.dir_action = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.food_x = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self.food_y = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self.body_len = torch.ones(n_envs, dtype=torch.int32, device=device)
        self.step_count = torch.ones(n_envs, dtype=torch.int32, device=device)
        self.last_food_step = torch.ones(n_envs, dtype=torch.int32, device=device)
        self.field_w = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self.field_h = torch.zeros(n_envs, dtype=torch.int32, device=device)
        self.reward_acc = torch.zeros(n_envs, dtype=torch.float32, device=device)

        self.reset_all()

    def reset_all(self):
        mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self._reset_envs(mask)

    def _reset_envs(self, mask):
        count = mask.sum().item()
        if count == 0:
            return
        self.field_w[mask] = torch.randint(self.min_field, self.max_field + 1, (count,), device=self.device, dtype=torch.int32)
        self.field_h[mask] = torch.randint(self.min_field, self.max_field + 1, (count,), device=self.device, dtype=torch.int32)
        self.head_x[mask] = 5
        self.head_y[mask] = 5
        self.dir_action[mask] = 1
        self.body_len[mask] = 1
        self.step_count[mask] = 1
        self.last_food_step[mask] = 1
        self.reward_acc[mask] = 0.0
        self.visit_grid[mask] = 0
        idx = torch.where(mask)[0]
        self.visit_grid[idx, 5, 5] = 1
        self.food_x[mask] = (torch.rand(count, device=self.device) * self.field_w[mask].float()).int()
        self.food_y[mask] = (torch.rand(count, device=self.device) * self.field_h[mask].float()).int()

    def step(self, actions):
        """Step all envs. actions: (n,) long tensor on device.
        Returns (obs, rewards, dones) all tensors on device."""
        n = self.n
        all_idx = torch.arange(n, device=self.device)

        # Prevent 180-degree reversal
        is_opposite = (actions == self.OPPOSITE[self.dir_action]) & (self.body_len > 1)
        actions = torch.where(is_opposite, self.dir_action, actions.int())
        self.dir_action = actions.long()

        dx = self.DIR_X[actions.long()]
        dy = self.DIR_Y[actions.long()]

        dist_before = torch.abs(self.food_x - self.head_x) + torch.abs(self.food_y - self.head_y)

        new_hx = self.head_x + dx
        new_hy = self.head_y + dy

        dist_after = torch.abs(self.food_x - new_hx) + torch.abs(self.food_y - new_hy)

        # Food check (old head == food, matching game.py)
        ate_food = (self.head_x == self.food_x) & (self.head_y == self.food_y)
        self.body_len += ate_food.int()

        # Respawn food
        n_eaters = ate_food.sum().item()
        if n_eaters > 0:
            self.food_x[ate_food] = (torch.rand(n_eaters, device=self.device) * self.field_w[ate_food].float()).int()
            self.food_y[ate_food] = (torch.rand(n_eaters, device=self.device) * self.field_h[ate_food].float()).int()
            self.last_food_step[ate_food] = self.step_count[ate_food]

        # Wall collision
        wall_hit = (new_hx < 0) | (new_hy < 0) | (new_hx >= self.field_w) | (new_hy >= self.field_h)

        # Body collision
        safe_hx = new_hx.clamp(0, self.max_field - 1)
        safe_hy = new_hy.clamp(0, self.max_field - 1)
        visit_time = self.visit_grid[all_idx, safe_hy, safe_hx]
        threshold = self.step_count - self.body_len
        body_hit = (visit_time > threshold) & ~wall_hit

        collision = wall_hit | body_hit
        self.step_count += 1

        # Update head for non-collided
        no_collision = ~collision
        nc_idx = torch.where(no_collision)[0]
        if nc_idx.numel() > 0:
            self.visit_grid[nc_idx, safe_hy[nc_idx], safe_hx[nc_idx]] = self.step_count[nc_idx]
        self.head_x = torch.where(collision, self.head_x, new_hx)
        self.head_y = torch.where(collision, self.head_y, new_hy)

        # Rewards
        rewards = torch.where(collision, -10.0,
                    torch.where(ate_food, 10.0,
                        torch.where(dist_after < dist_before, 1.0, -1.0)))
        self.reward_acc += rewards

        stagnant = (self.step_count - self.last_food_step) > (50 * self.body_len)
        dones = collision | stagnant

        obs = self._observations()

        # Capture body lengths before reset
        done_body_lens = self.body_len.clone()

        self._reset_envs(dones)

        return obs, rewards, dones, done_body_lens

    def _observations(self):
        """Fully vectorized observation on GPU. Returns (n, 61) float32 tensor."""
        n = self.n
        all_idx = torch.arange(n, device=self.device)
        hx, hy = self.head_x, self.head_y
        fx, fy = self.food_x, self.food_y
        da = self.dir_action
        fw, fh = self.field_w, self.field_h

        ddx = self.DIR_X[da]
        ddy = self.DIR_Y[da]

        # Scalars
        go_up = (fy < hy).float()
        go_down = (fy > hy).float()
        go_left = (fx < hx).float()
        go_right = (fx > hx).float()

        threshold = (self.step_count - self.body_len)  # (n,)

        # Danger helper
        def check_danger(px, py):
            oob = (px < 0) | (py < 0) | (px >= fw) | (py >= fh)
            cpx = px.clamp(0, self.max_field - 1)
            cpy = py.clamp(0, self.max_field - 1)
            vt = self.visit_grid[all_idx, cpy, cpx]
            body = (vt > threshold) & ~oob
            return (oob | body).float()

        danger_straight = check_danger(hx + ddx, hy + ddy)
        danger_left = check_danger(hx - ddy, hy + ddx)
        danger_right = check_danger(hx + ddy, hy - ddx)

        dir_right = (ddx == 1).float()
        dir_left = (ddx == -1).float()
        dir_down = (ddy == 1).float()
        dir_up = (ddy == -1).float()

        field_area = (fw * fh).float()
        body_norm = self.body_len.float() / field_area

        scalars = torch.stack([
            go_up, go_down, go_left, go_right,
            danger_straight, danger_left, danger_right,
            dir_right, dir_left, dir_down, dir_up,
            body_norm
        ], dim=1)  # (n, 12)

        # 7x7 local grid — fully vectorized
        grid_cx = hx[:, None] + self.COL_OFFSETS[None, :]  # (n, 49)
        grid_cy = hy[:, None] + self.ROW_OFFSETS[None, :]  # (n, 49)

        oob = (grid_cx < 0) | (grid_cy < 0) | (grid_cx >= fw[:, None]) | (grid_cy >= fh[:, None])

        clip_cx = grid_cx.clamp(0, self.max_field - 1)
        clip_cy = grid_cy.clamp(0, self.max_field - 1)
        env_idx_2d = all_idx[:, None].expand(-1, 49)
        visit_times = self.visit_grid[env_idx_2d, clip_cy, clip_cx]  # (n, 49)
        is_body = (visit_times > threshold[:, None]) & ~oob

        is_food = (grid_cx == fx[:, None]) & (grid_cy == fy[:, None]) & ~oob

        local_grid = torch.where(oob | is_body, 1.0, torch.where(is_food, -1.0, 0.0))

        return torch.cat([scalars, local_grid], dim=1)  # (n, 61)

    def observations(self):
        return self._observations()
