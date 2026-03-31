

wnd = 2 * 2 + 1
n_scalar = 12        # food(4) + danger(3) + direction(4) + body_len(1)
grid_size = 7
n_observations = n_scalar + grid_size * grid_size  # 12 + 49 = 61
n_actions = 4