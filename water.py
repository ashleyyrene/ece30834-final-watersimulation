import pygame
import numpy as np

from settings import (
    WIDTH, HEIGHT, GRID_W, GRID_H,
    DAMPING, WAVE_SPEED,
    RIPPLE_STRENGTH, RIPPLE_RADIUS,
    EDGE_DAMPING_WIDTH, EDGE_DAMPING_AMOUNT,
    LIGHT_DIR,
    AMBIENT_WATER_MOTION,
    AMBIENT_STRENGTH,
    AMBIENT_SPEED
)

LILY_PAD_COLOR = None
LILY_PAD_MASK = None
LILY_PAD_SHADOW = None


def create_lily_pads():
    global LILY_PAD_COLOR, LILY_PAD_MASK, LILY_PAD_SHADOW

    # boolean mask
    lily_pads = np.zeros((GRID_H, GRID_W), dtype=bool)
    lily_color = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)

    y, x = np.ogrid[:GRID_H, :GRID_W]

    # position of lily pads
    pads = [
        (0.25, 0.25, 0.08, 0.11, 0.3, -0.5),
        (0.30, 0.75, 0.07, 0.10, -0.6, 1.2),
        (0.70, 0.30, 0.10, 0.14, 0.8, -1.1),
        (0.75, 0.75, 0.09, 0.13, -0.4, 2.2)
    ]

    for cy_p, cx_p, ry_p, rx_p, angle, cut_angle in pads:
        cy = int(cy_p * GRID_H)
        cx = int(cx_p * GRID_W)
        ry = int(ry_p * GRID_H)
        rx = int(rx_p * GRID_W)

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        dx = x - cx
        dy = y - cy

        # rotation of lily pads
        xr = dx * cos_a + dy * sin_a
        yr = -dx * sin_a + dy * cos_a

        # ellipse equation
        radius = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
        theta = np.arctan2(yr, xr)

        ellipse = radius <= 1

        angle_diff = np.arctan2(
            np.sin(theta - cut_angle),
            np.cos(theta - cut_angle)
        )

        # cut in lily pad
        wedge = (np.abs(angle_diff) < 0.38) & (xr > 0)
        # boolean mask - array same size as grid
        mask = ellipse & ~wedge

        # mask is true where lily pads exist
        lily_pads[mask] = True

        # shading - center brightness
        center_light = np.clip(1.25 - radius, 0.55, 1.2)

        # light hitting from a direction
        directional = 0.9 + 0.25 * (
            -np.cos(theta) * LIGHT_DIR[0] - np.sin(theta) * LIGHT_DIR[1]
        )

        shade = center_light * directional

        veins = np.zeros((GRID_H, GRID_W), dtype=np.float32)

        for k in range(10):
            vein_angle = k * (2 * np.pi / 10)
            diff = np.abs(np.arctan2(
                np.sin(theta - vein_angle),
                np.cos(theta - vein_angle)
            ))

            veins += np.exp(-(diff ** 2) / 0.003) * (1 - radius)

        veins = np.clip(veins, 0, 0.35)

        texture = (
            0.08 * np.sin(0.5 * x + 1.1 * y)
            + 0.05 * np.sin(1.2 * x - 0.8 * y)
        )

        final = np.clip(shade + veins + texture, 0.45, 1.3)

        red = 28 * final
        green = 110 * final
        blue = 35 * final

        lily_color[mask, 0] = np.clip(red[mask], 0, 255).astype(np.uint8)
        lily_color[mask, 1] = np.clip(green[mask], 0, 255).astype(np.uint8)
        lily_color[mask, 2] = np.clip(blue[mask], 0, 255).astype(np.uint8)

        rim = mask & (radius > 0.88)
        lily_color[rim] = (lily_color[rim] * 0.55).astype(np.uint8)

        center = mask & (radius < 0.08)
        lily_color[center] = [145, 165, 45]

    # shadow around lily pads for realism
    shadow = np.roll(lily_pads, shift=(6, 7), axis=(0, 1)) & ~lily_pads

    LILY_PAD_COLOR = lily_color
    LILY_PAD_MASK = lily_pads
    LILY_PAD_SHADOW = shadow

    return lily_pads

def create_damping_map():
    damping_map = np.ones((GRID_H, GRID_W), dtype=np.float32) * DAMPING

    edge = EDGE_DAMPING_WIDTH
    damping_map[:edge, :] *= EDGE_DAMPING_AMOUNT
    damping_map[-edge:, :] *= EDGE_DAMPING_AMOUNT
    damping_map[:, :edge] *= EDGE_DAMPING_AMOUNT
    damping_map[:, -edge:] *= EDGE_DAMPING_AMOUNT

    return damping_map


def add_ripple(current, previous, x, y, strength=RIPPLE_STRENGTH):
    gx = int(x / WIDTH * GRID_W)
    gy = int(y / HEIGHT * GRID_H)

    for dy in range(-RIPPLE_RADIUS, RIPPLE_RADIUS + 1):
        for dx in range(-RIPPLE_RADIUS, RIPPLE_RADIUS + 1):
            nx = gx + dx
            ny = gy + dy

            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                if LILY_PAD_MASK is not None and LILY_PAD_MASK[ny, nx]:
                    continue

                distance = dx * dx + dy * dy

                if distance <= RIPPLE_RADIUS * RIPPLE_RADIUS:
                    falloff = 1 - distance / (RIPPLE_RADIUS * RIPPLE_RADIUS)

                    current[ny, nx] += strength * falloff
                    previous[ny, nx] -= strength * falloff


def update_waves(current, previous, damping_map, lily_pads):
    # wave equation
    wave_equation = (
        np.roll(current, 1, axis=0)     # wraps the grid around- opposite edges treated as neighbors
        + np.roll(current, -1, axis=0)
        + np.roll(current, 1, axis=1)
        + np.roll(current, -1, axis=1)
        - 4 * current
    )

    # numpy optimization to update whole grid at once
    new = 2 * current - previous + WAVE_SPEED * wave_equation
    new *= damping_map

    # grid cell where a lily pad exists is forced to have wave height of 0
    new[lily_pads] = 0
    current[lily_pads] = 0

    return new, current


def compute_normals(current):
    # surface normal computation
    dx = np.roll(current, -1, axis=1) - np.roll(current, 1, axis=1)
    dy = np.roll(current, -1, axis=0) - np.roll(current, 1, axis=0)

    # estimates slope of the water then builds normal vectors
    normal_x = -dx
    normal_y = -dy
    normal_z = np.ones_like(current)

    length = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)

    # builds 3D vector perpendicular to surface
    # each point has a unit normal vector
    normal_x /= length
    normal_y /= length
    normal_z /= length

    return normal_x, normal_y, normal_z

def phong_shading(normal_x, normal_y, normal_z):
    # normal vector 
    N = np.dstack((normal_x, normal_y, normal_z))

    # light direction 
    L = np.array(LIGHT_DIR, dtype=np.float32)
    L = L / np.linalg.norm(L)

    # view direction looking straight down at water
    V = np.array([0, 0, 1], dtype=np.float32)

    # ambient term
    ambient = 0.25

    # diffuse - light hitting surface
    diffuse = np.clip(
        normal_x * L[0] + normal_y * L[1] + normal_z * L[2],
        0,
        1
    )

    # reflection vector
    dot = diffuse
    R = 2 * dot[:, :, None] * N - L

    # normalize R
    R_length = np.sqrt(
        R[:, :, 0] ** 2 + R[:, :, 1] ** 2 + R[:, :, 2] ** 2
    )
    R[:, :, 0] /= R_length
    R[:, :, 1] /= R_length
    R[:, :, 2] /= R_length

    # specular - shine in water
    specular = np.clip(
        R[:, :, 0] * V[0] + R[:, :, 1] * V[1] + R[:, :, 2] * V[2],
        0,
        1
    ) ** 40

    return ambient, diffuse, specular


def render_water(screen, current, lily_pads, time):
    display_height = current.copy()

    if AMBIENT_WATER_MOTION:
        y, x = np.mgrid[0:GRID_H, 0:GRID_W]

    ambient = (
        0.45 * np.sin(x * 0.045 + time * 1.2)
        + 0.35 * np.sin(y * 0.052 + time * 1.0)
        + 0.25 * np.sin((x + y) * 0.035 + time * 0.8)
        + 0.15 * np.sin((x - y) * 0.025 + time * 0.6)
    )

    display_height += ambient * AMBIENT_STRENGTH

    normal_x, normal_y, normal_z = compute_normals(display_height)

    ambient, diffuse, specular = phong_shading(
        normal_x,
        normal_y,
        normal_z
    )  

    y, x = np.mgrid[0:GRID_H, 0:GRID_W]

    # light patterns in water - moving light
    caustics = (
        np.sin(x * 0.055 + time * 1.4)
        + np.sin(y * 0.065 + time * 1.1)
        + np.sin((x + y) * 0.04 + time * 0.9)
    )

    caustics = (caustics + 3) / 6
    caustics = caustics ** 4

    color = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)

    # base color of display
    base_red = 8
    base_green = 75
    base_blue = 135

    red = base_red * ambient + base_red * diffuse + specular * 220 + caustics * 10
    green = base_green * ambient + base_green * diffuse + specular * 230 + caustics * 25
    blue = base_blue * ambient + base_blue * diffuse + specular * 255 + caustics * 35

    color[:, :, 0] = np.clip(red, 0, 255).astype(np.uint8)
    color[:, :, 1] = np.clip(green, 0, 255).astype(np.uint8)
    color[:, :, 2] = np.clip(blue, 0, 255).astype(np.uint8)

    color[LILY_PAD_SHADOW] = (color[LILY_PAD_SHADOW] * 0.45).astype(np.uint8)
    color[LILY_PAD_MASK] = LILY_PAD_COLOR[LILY_PAD_MASK]

    surface = pygame.surfarray.make_surface(np.transpose(color, (1, 0, 2)))
    surface = pygame.transform.smoothscale(surface, (WIDTH, HEIGHT))

    screen.blit(surface, (0, 0))
