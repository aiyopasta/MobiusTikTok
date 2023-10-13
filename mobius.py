import copy
from svgpathtools import svg2paths
import numpy as np
np.set_printoptions(suppress=True)
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from playsound import playsound
from functools import partial

# Save the animation? TODO: Make sure you're saving to correct destination!!
save_anim = True

# Pygame + gameloop setup
width = 800
height = 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Parallel Transport Animations")
pygame.init()


# Coordinate Shift Functions
def A(val):
    return np.array([val[0] + width / 2, -val[1] + height / 2])


def A_inv(val):
    global width, height
    return np.array([val[0] - width / 2, -(val[1] - height / 2)])


def A_many(vals):
    return [A(v) for v in vals]   # Galaxy brain function


# 3D Camera parameters (rho = distance from origin, phi = angle from world's +z-axis, theta=angle from world's +x-axis)
rho, theta, phi = 1000., -np.pi/2, 0  # Rho is the distance from world origin to near clipping plane
v_rho, v_theta, v_phi = 0, 0, 0
focus = 1000000.  # Distance from near clipping plane to eye


# Normal perspective project
def world_to_plane(v):
    '''
        Converts from point in 3D to its 2D perspective projection, based on location of camera.

        v: vector in R^3 to convert.
        NOTE: Here, we do NOT "A" the final output.
    '''
    # Camera params
    global rho, theta, phi, focus

    # Radial distance to eye from world's origin.
    eye_rho = rho + focus

    # Vector math from geometric computation (worked out on white board, check iCloud for possible picture)
    eye_to_origin = -np.array([eye_rho * np.sin(phi) * np.cos(theta),
                               eye_rho * np.sin(phi) * np.sin(theta), eye_rho * np.cos(phi)])

    eye_to_ei = eye_to_origin + v
    origin_to_P = np.array(
        [rho * np.sin(phi) * np.cos(theta), rho * np.sin(phi) * np.sin(theta), rho * np.cos(phi)])

    # Formula for intersecting t: t = (n•(a-b)) / (n•v)
    tau = np.dot(eye_to_origin, origin_to_P - v) / np.dot(eye_to_origin, eye_to_ei)
    r_t = v + (tau * eye_to_ei)

    # Location of image coords in terms of world coordinates.
    tile_center_world = -origin_to_P + r_t

    # Spherical basis vectors
    theta_hat = np.array([-np.sin(theta), np.cos(theta), 0])
    phi_hat = -np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), -np.sin(phi)])

    # Actual transformed 2D coords
    tile_center = np.array([np.dot(tile_center_world, theta_hat), np.dot(tile_center_world, phi_hat)])

    return tile_center


def world_to_plane_many(pts):
    return np.array([world_to_plane(v) for v in pts])  # galaxy brain function


# Keyframe / timing params
FPS = 60
t = 0
dt = 0.01    # i.e., 1 frame corresponds to +0.01 in parameter space = 0.01 * FPS = +0.6 per second (assuming 60 FPS)

keys = [0,      # Keyframe 0. Straight line move
        4.]

# keys.extend([keys[-1] + 10, keys[-1] + (10 * 2)])  # Placeholder + done


# Helper functions
def lerp(t_, start, fin):
    return ((1. - t_) * start) + (t_ * fin)


def squash(t_, intervals=None):
    global keys
    if intervals is None:
        intervals = keys
    for i in range(len(intervals) - 1):
        if intervals[i] <= t_ < intervals[i + 1]:
            return (t_ - intervals[i]) / (intervals[i + 1] - intervals[i]), i

    return intervals[-1], len(intervals) - 2


# Specific case of the squash. We squash t into equally sized intervals.
def squash2(t_, n_intervals=1):
    intervals = [float(i) / n_intervals for i in range(n_intervals + 1)]
    return squash(t_, intervals)


# Squeeze actual interpolation to be within [new_start, new_end], and make it 0 and 1 outside this range
def slash(t_, new_start=0.0, new_end=0.5):
    if t_ < new_start:
        return 0.0

    if t_ > new_end:
        return 1.0

    return (t_ - new_start) / (new_end - new_start)


def rot_mat(radians):
    M = np.array([[np.cos(radians), -np.sin(radians), 0],
                  [np.sin(radians), np.cos(radians), 0],
                  [0, 0, 1]])
    return M


# Easing functions.
# TODO: Add more!
def ease_inout(t_):
    return np.power(t_, 2) / (np.power(t_, 2) + np.power(1 - t_, 2))


def ease_inout2(t_, beta, center=0.5):
    r = -np.log(2) / np.log(center)
    power = np.power(t_, r)
    return 1.0 / (1.0 + np.power(power / (1 - power), -beta)) if t_ not in [0., 1., 0, 1] else t_


def ease_out(t_):
    return 1.0 - np.power(1.0 - t_, 2.0)


def blink_bump(t_, p):
    assert 0.05 <= p <= 0.95
    return np.exp(-100. * np.power(t_ - p, 2.0) / np.power(0.1, 2.0))


# Inverse of easing functions
# TODO: Add more!
def ease_inout_inverse(t_):
    return (t_ - np.sqrt((1 - t_) * t_)) / ((2 * t_) - 1)


# From ChatGPT
def rgb_to_hex(rgb):
    return '#' + ''.join(['%02x' % int(val) for val in rgb])


# Parametric shapes
def line(u, start, fin):
    return ((1 - u) * start) + (u * fin)


def ellipse3d(u, width_, height_, center_x=0, center_y=0, center_z=0):
    '''
        Parametric ellipse in 3D, parallel to the xy-plane (ground).
    '''
    tau = 2 * np.pi * u
    return np.array([width_ * np.cos(tau), height_ * np.sin(tau), 0.]) + np.array([center_x, center_y, center_z])


def circle3d(u, radius, center_x=0, center_y=0, center_z=0):
    '''
        Parametric circle in 3D, parallel to the xy-plane (ground).
    '''
    ellipse3d(u, radius, radius, center_x, center_y, center_z)


def Scurve3d(u, startpt, endpt, steepness=1.0):
    '''
        Returns an S shaped curve in 3D connecting the two points, in the xy-plane
    '''
    return np.array([lerp(ease_inout2(u, beta=steepness), startpt[0], endpt[0]), lerp(u, startpt[1], endpt[1]), 0.])


# Shape sampling functions
def get_ellipse3d_pts(u, width_, height_, center_x, center_y, center_z, du=0.001):
    pts = []
    for u_ in np.arange(0, u + du, du):
        pts.append(ellipse3d(u_, width_, height_, center_x, center_y, center_z))
    return pts


def get_circle3d_pts(u, radius, center_x, center_y, center_z, du=0.001):
    return get_ellipse3d_pts(u, radius, radius, center_x, center_y, center_z, du)


def get_Scurve3d_pts(u, startpt, endpt, steepness=1.0, du=0.001):
    pts = []
    for u_ in np.arange(0, u + du, du):
        pts.append(Scurve3d(u_, startpt, endpt, steepness=steepness))
    return pts


# TODO: Implement for circles not necessarily aligned with the xy-plane.
def get_gradient_circle3d(radius, center_x, center_y, center_z, start_col, end_col, sharpness=5.0):
    '''
        'sharpness': How sharp the gradient is.

        Yields, one by one, tuples of the form (linepts, color), where 'linepts' contains the start and end points
        of one of the many lines in question (forming the gradient) and the 'color' is the color of that line.

        NOTE: The gradient is in the x-direction.
    '''

    center_x, center_y, center_z, radius = int(center_x), int(center_y), int(center_z), int(radius)
    for x in range(center_x - radius, center_x + radius + 1, 1):
        u_ = (x + radius) / (2 * radius)
        u_ = ease_inout2(u_, beta=sharpness)
        sqrt = np.sqrt(np.power(radius, 2.0) - np.power(x - center_x, 2.0))
        ptop, pbot = np.array([x, center_y + sqrt, center_z]), np.array([x, center_y - sqrt, center_z])
        yield ptop, pbot, lerp(u_, start_col, end_col)


# Keyhandling
def handle_keys(keys_pressed):
    global rho, theta, phi, focus
    m = 300
    drho, dphi, dtheta, dfocus = 10, np.pi/m, np.pi/m, 10

    if keys_pressed[pygame.K_w]:
        phi -= dphi
    if keys_pressed[pygame.K_a]:
        theta -= dtheta
    if keys_pressed[pygame.K_s]:
        phi += dphi
    if keys_pressed[pygame.K_d]:
        theta += dtheta
    if keys_pressed[pygame.K_p]:
        rho -= drho
    if keys_pressed[pygame.K_o]:
        rho += drho
    if keys_pressed[pygame.K_k]:
        focus -= dfocus
    if keys_pressed[pygame.K_l]:
        focus += dfocus


# From ChatGPT (because text in pygame is stupid as shit)
def blit_rotated_text(screen, text_surface, pos, angle):
    # Create a new Surface with the same size as the original text_surface
    rotated_surface = pygame.transform.rotate(text_surface, angle)

    # Create a new Rect object with the same center as the original text_surface
    rotated_rect = rotated_surface.get_rect(center=text_surface.get_rect(center=pos).center)

    # Draw the rotated image to the screen at the specified position
    screen.blit(rotated_surface, rotated_rect)


# Additional vars / knobs
play_music = False
colors = {
    'white': np.array([255., 255., 255.]),
    'black': np.array([0., 0., 0.]),
    'red': np.array([255, 66, 48]),
    'blue': np.array([30, 5, 252]),
    'fullred': np.array([255, 0, 0]),
    'fullblue': np.array([0, 0, 255]),
    'START': np.array([255, 255, 255])
}


def main():
    global t, dt, keys, FPS, rho, phi, theta, focus, save_anim, play_music, colors

    # Pre-animation setup
    clock = pygame.time.Clock()
    run = True

    # Animation saving setup
    path_to_save = '/Users/adityaabhyankar/Desktop/Programming/MobiusTikTok/output'
    if save_anim:
        for filename in os.listdir(path_to_save):
            # Check if the file name follows the required format
            b1 = filename.startswith("frame") and filename.endswith(".png")
            b2 = filename.startswith("output.mp4")
            if b1 or b2:
                os.remove(os.path.join(path_to_save, filename))
                print('Deleted frame ' + filename)

    # Text data (Not necessary, we're using SVG data anyway. Just kept it here in case.)
    # font = pygame.font.SysFont("Avenir Next", 70)
    # text = font.render("START", True, (252, 3, 90))

    # Load letter paths (using svg image)
    paths, _ = svg2paths("knuth.svg")
    letters_map = ['A', 'C', 'G', 'B', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'Q', 'S', 'P',
                   'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'f', 'b', 'd', 'h', 'k', 'l', 'i1', 'j1', 'c',
                   'g', 's', 'e', 'i2', 'j2', 'm', 'n', 'o', 'p', 'q', 'r', '4', '7', '0', '1', '2', '3', '5',
                   '6', '8', 't', 'u', 'v', 'w', 'x', 'y', 'z', '[', ']', '\\', '/', '$', '!1', '@', '\'', '#',
                   '~', '9', '`', ';1', '=1', '-', '=2', ';2', ',', '.', '!2', '%1', '*', '(', ')', '{', '}',
                   '|', '&', '?1', '"1', '"2', '^', '+', '<', '>', ':1', '%2', ':2', '?2', '_']
    STAR_paths = {
        'S': paths[letters_map.index('S')],
        'T': paths[letters_map.index('T')],
        'A': paths[letters_map.index('A')],
        'R': paths[letters_map.index('R')]
    }

    # Parameters for the "fake" strip + Mobius strip
    strip_width = 600.
    straight_length = 5000.  # This is the length of ONE CYCLE of the "fake" straight-line portion, NOT full strip.
    y_sampling_interval = 75.  # for the straight line specifically
    x_sampling_interval = 75.  # for the straight line specifically
    straight_length = y_sampling_interval * np.round(straight_length / y_sampling_interval)  # make a multiple of interval
    strip_width = x_sampling_interval * np.round(strip_width / x_sampling_interval)  # make a multiple of interval

    # Parameters for the traveler guy
    radius = 100.
    blink_times = [0.1, 0.5, 0.6, 0.8]
    eye_params = {
        'ypos': 0.6,  # center of eyeball +n% of the way BODY
        'width': radius * 0.4,
        'height': radius * 0.35,
        'iris_pos': 0.35,  # center of iris +n% of the way EYEBALL
        'iris_width': radius * 0.25,
        'iris_height': radius * 0.25
    }

    # Game loop
    count = 0
    while run:
        # Reset stuff
        window.fill((0, 0, 0))

        # Animation!
        u, frame = squash(t)
        # Keyframe 0 — Straight Line Move
        if frame == 0:
            yOffset = -ease_out(slash(u, 0.2, 1.0)) * straight_length

            # Draw 1 cycle of "fake" straight-line strip. It starts at y=0 (so that Mobius strip ends up
            # being centered at the origin; remember the starting of this "fake" strip = middle of the straight-line
            # portion of the Mobius strip.) TODO: Extend the strip to be > 1 cycle (for "infinite" feel.)
            # (a) Horizontal lines
            xleft, xright = -strip_width / 2., +strip_width / 2.
            for y in np.arange(0, straight_length + y_sampling_interval, y_sampling_interval):
                leftPt, rightPt = world_to_plane(np.array([xleft, y+yOffset, 0])), world_to_plane(np.array([xright, y+yOffset, 0]))
                pygame.draw.line(window, colors['white'], *A_many([leftPt, rightPt]), width=2)
            # (b) Vertical Lines
            for x in np.arange(xleft, xright + x_sampling_interval, x_sampling_interval):
                botPt, topPt = world_to_plane(np.array([x, yOffset, 0])), world_to_plane(np.array([x, straight_length+yOffset, 0]))
                pygame.draw.line(window, colors['white'], *A_many([botPt, topPt]), width=2)

            # Draw the red / blue wires (just for effect of following along)
            for k in range(2):
                mult = -1 if k==1 else 1
                col = colors['blue'] if k==1 else colors['red']
                startpt, endpt = np.array([mult * -strip_width / 4., 0., 0.]), np.array([mult * strip_width / 4., straight_length, 0.])
                startpt[1] += yOffset; endpt[1] += yOffset
                pts = world_to_plane_many(get_Scurve3d_pts(1, startpt, endpt, steepness=2.0, du=0.001))
                pygame.draw.lines(window, col, False, A_many(pts), width=30)
                pygame.draw.lines(window, lerp(0.3, col, colors['white']), False, A_many(pts), width=10)

            # Draw the red + blue boxes for starting line (TODO: Draw it again for the ending of the strip)
            boxheight = y_sampling_interval * 6.
            cols = [colors['red'], colors['blue']]
            for y in [0, straight_length]:
                yoff = y + yOffset
                for i, leftX in enumerate([xleft, 0]):
                    halfwidth = strip_width / 2.
                    topright = world_to_plane(np.array([leftX + halfwidth, boxheight / 2. + yoff, 0]))
                    topleft = world_to_plane(np.array([leftX, boxheight / 2. + yoff, 0]))
                    botleft = world_to_plane(np.array([leftX, -boxheight / 2. + yoff, 0]))
                    botright = world_to_plane(np.array([leftX + halfwidth, -boxheight / 2. + yoff, 0]))
                    pygame.draw.polygon(window, cols[i], A_many([topright, topleft, botleft, botright]), 0)
                    pygame.draw.polygon(window, colors['white'], A_many([topright, topleft, botleft, botright]), 2)
                cols[0], cols[1] = cols[1], cols[0]

            # Draw "START" Text via SVG
            scale = [width * 0.13, width * 0.2]
            spacing = 113
            for k in range(2):
                yoff = yOffset + (straight_length if k == 1 else 0)
                mult = -1 if k == 1 else 1

                for j, char in enumerate('START'):
                    path = STAR_paths[char]
                    points = [path.point(t).conjugate() for t in np.linspace(0, 1, 50)]

                    # Renormalize coordinates
                    min_x, max_x = min(point.real for point in points), max(point.real for point in points)
                    avg_x = (min_x + max_x) / 2.
                    min_y, max_y = min(point.imag for point in points), max(point.imag for point in points)
                    avg_y = (min_y + max_y) / 2.
                    for i, point in enumerate(points):
                        normalized_x = (point.real - avg_x) / (max_x - min_x)
                        normalized_y = -(point.imag - avg_y) / (max_y - min_y)
                        points[i] = world_to_plane(np.array([mult * ((normalized_x * scale[0]) - strip_width / 2.7 + (j * spacing)),
                                                             yoff + normalized_y * scale[1] + (0.5 * boxheight / 2),
                                                             0.0]))
                    # Draw the path
                    if len(points) > 1:
                        pygame.draw.polygon(window, colors['START'], A_many(points))  # Fill the shape

            # Draw the traveler
            amp, n_cycles = 10, 5
            xOffset = amp * np.sin(2 * np.pi * slash(u, 0.2, 1.0) * n_cycles) * np.sin(np.pi * slash(u, 0.2, 1.0))
            # (a) Draw main body
            traveler_pos = np.array([xOffset, -boxheight/2 + radius*1.3, 0.])
            pts = get_circle3d_pts(1.0, radius, *traveler_pos, du=0.01)
            start_col, end_col = lerp(0.6, colors['red'], colors['fullred']), lerp(0.6, colors['blue'], colors['fullblue'])
            for ptop, pbot, col in get_gradient_circle3d(radius, *traveler_pos, start_col, end_col, sharpness=2.0):
                pygame.draw.line(window, col, *A_many([world_to_plane(ptop), world_to_plane(pbot)]), width=3)
            # pygame.draw.polygon(window, colors['black'], A_many(world_to_plane_many(pts)), 5)
            # ***
            # For blinking
            blink_mult = sum([blink_bump(u, p) for p in blink_times])
            eye_height = lerp(blink_mult, eye_params['height'], 0.0)
            iris_height = lerp(blink_mult, eye_params['iris_height'], 0.0)

            # ***
            # (b) Draw white eyeball
            eyeball_pos = traveler_pos + np.array([0., eye_params['ypos'] * radius, 0.])
            pts = get_ellipse3d_pts(1.0, eye_params['width'], eye_height, *eyeball_pos, du=0.01)
            pygame.draw.polygon(window, colors['white'], A_many(world_to_plane_many(pts)), 0)
            pygame.draw.polygon(window, colors['black'], A_many(world_to_plane_many(pts)), 3)
            # (c) Draw black iris
            iris_pos = eyeball_pos + np.array([0., eye_params['iris_pos'] * eye_height, 0.])
            pts = get_ellipse3d_pts(1.0, eye_params['iris_width'], iris_height, *iris_pos, du=0.01)
            pygame.draw.polygon(window, colors['black'], A_many(world_to_plane_many(pts)), 0)






        # We handle keys pressed inside the gameloop in PyGame
        keys_pressed = pygame.key.get_pressed()
        handle_keys(keys_pressed)


        # End run (1. Tick clock, 2. Save the frame, and 3. Detect if window closed)
        pygame.display.update()
        t += dt
        clock.tick(FPS)
        count += 1
        if save_anim:
            pygame.image.save(window, path_to_save+'/frame'+str(count)+'.png')
            print('Saved frame '+str(count))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()


    # Post game-loop stuff
    # Do more stuff...
    # Use ffmpeg to combine the PNG images into a video
    if save_anim:
        input_files = path_to_save + '/frame%d.png'
        output_file = path_to_save + '/output.mp4'
        ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
        os.system(f'{ffmpeg_path} -r 60 -i {input_files} -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf "eq=brightness=0.00:saturation=1.3" {output_file} > /dev/null 2>&1')
        print('Saved video to ' + output_file)


if __name__ == "__main__":
    main()
