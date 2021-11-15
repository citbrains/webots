# Copyright 1996-2021 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gamestate import GameState
from field import Field
from forceful_contact_matrix import ForcefulContactMatrix

from controller import Supervisor, AnsiCodes, Node

import copy
import json
import math
import numpy as np
import os
import random
import socket
import sys
import time

from scipy.spatial import ConvexHull

from types import SimpleNamespace

import mmap

SIMULATED_TIME_INTERRUPTION_PHASE_0 = 5   # waiting time of 5 simulated seconds in phase 0 of interruption
SIMULATED_TIME_BEFORE_PLAY_STATE = 5      # wait 5 simulated seconds in SET state before sending the PLAY state
SIMULATED_TIME_SET_PENALTY_SHOOTOUT = 15  # wait 15 simulated seconds in SET state before sending the PLAY state
HALF_TIME_BREAK_REAL_TIME_DURATION = 15   # the half-time break lasts 15 real seconds
REAL_TIME_BEFORE_FIRST_READY_STATE = 120  # wait 2 real minutes before sending the first READY state
IN_PLAY_TIMEOUT = 10                      # time after which the ball is considered in play even if it was not kicked
GOALKEEPER_BALL_HOLDING_TIMEOUT = 6       # a goalkeeper may hold the ball up to 6 seconds on the ground
PLAYERS_BALL_HOLDING_TIMEOUT = 1          # field players may hold the ball up to 1 second
END_OF_GAME_TIMEOUT = 5                   # Once the game is finished, let the referee run for 5 seconds before closing game
FOUL_PUSHING_TIME = 1                     # 1 second
FOUL_PUSHING_PERIOD = 2                   # 2 seconds
GOAL_WIDTH = 2.6                          # width of the goal
RED_COLOR = 0xd62929                      # red team color used for the display
BLUE_COLOR = 0x2943d6                     # blue team color used for the display
WHITE_COLOR = 0xffffff                    # white color used for the display
BLACK_COLOR = 0x000000                    # black color used for the display
STATIC_SPEED_EPS = 1e-2                   # The speed below which an object is considered as static [m/s]
DROPPED_BALL_TEAM_ID = 128                # The team id used for dropped ball
BALL_DIST_PERIOD = 1                      # seconds. The period at which distance to the ball is checked
BALL_HOLDING_RATIO = 1.0/3                # The ratio of the radius used to compute minimal distance to the convex hull
GAME_INTERRUPTION_PLACEMENT_NB_STEPS = 5  # The maximal number of steps allowed when moving ball or player away
STATUS_PRINT_PERIOD = 20                  # Real time between two status updates in seconds
DISABLE_ACTUATORS_MIN_DURATION = 1.0      # The minimal simulated time [s] until enabling actuators again after a reset

# game interruptions requiring a free kick procedure
GAME_INTERRUPTIONS = {
    'DIRECT_FREEKICK': 'direct free kick',
    'INDIRECT_FREEKICK': 'indirect free kick',
    'PENALTYKICK': 'penalty kick',
    'CORNERKICK': 'corner kick',
    'GOALKICK': 'goal kick',
    'THROWIN': 'throw in'}

GOAL_HALF_WIDTH = GOAL_WIDTH / 2

global supervisor, game, red_team, blue_team, log_file, time_count, time_step#, game_controller_udp_filter


def log(message, msg_type, force_flush=True):
    if type(message) is list:
        for m in message:
            log(m, msg_type, False)
        if log_file and force_flush:
            log_file.flush()
        return
    if msg_type == 'Warning':
        console_message = f'{AnsiCodes.YELLOW_FOREGROUND}{AnsiCodes.BOLD}{message}{AnsiCodes.RESET}'
    elif msg_type == 'Error':
        console_message = f'{AnsiCodes.RED_FOREGROUND}{AnsiCodes.BOLD}{message}{AnsiCodes.RESET}'
    else:
        console_message = message
    print(console_message, file=sys.stderr if msg_type == 'Error' else sys.stdout)
    if log_file:
        real_time = int(1000 * (time.time() - log.real_time)) / 1000
        log_file.write(f'[{real_time:08.3f}|{time_count / 1000:08.3f}] {msg_type}: {message}\n')  # log real and virtual times
        if force_flush:
            log_file.flush()


log.real_time = time.time()


def announce_final_score():
    if not hasattr(game, "state"):
        return
    red_team_idx = team_index('red')
    blue_team_idx = team_index('blue')
    red_score = game.state.teams[red_team_idx].score
    blue_score = game.state.teams[blue_team_idx].score
    # TODO: store and print score before penalty shootouts
    info(f"FINAL SCORE: {red_score}-{blue_score}")


def clean_exit():
    """Save logs and clean all subprocesses"""
    announce_final_score()
    if hasattr(game, 'over') and game.over:
        info("Game is over")
        if hasattr(game, 'press_a_key_to_terminate') and game.press_a_key_to_terminate:
            print('Press a key to terminate')
            keyboard = supervisor.getKeyboard()
            keyboard.enable(time_step)
            while supervisor.step(time_step) != -1:
                if keyboard.getKey() != -1:
                    break
        else:
            waiting_steps = END_OF_GAME_TIMEOUT * 1000 / time_step
            info(f"Waiting {waiting_steps} simulation steps before exiting")
            while waiting_steps > 0:
                supervisor.step(time_step)
                waiting_steps -= 1
            info("Finished waiting")
    if hasattr(game, 'record_simulation'):
        if game.record_simulation.endswith(".html"):
            info("Stopping animation recording")
            supervisor.animationStopRecording()
        elif game.record_simulation.endswith(".mp4"):
            info("Starting encoding")
            supervisor.movieStopRecording()
            while not supervisor.movieIsReady():
                supervisor.step(time_step)
            info("Encoding finished")
    info("Exiting webots properly")

    if log_file:
        log_file.close()

    # Note: If supervisor.step is not called before the 'simulationQuit', information is not shown
    supervisor.step(time_step)
    supervisor.simulationQuit(0)


def info(message):
    log(message, 'Info')


def warning(message):
    log(message, 'Warning')


def error(message, fatal=False):
    log(message, 'Error')
    if fatal:
        clean_exit()


def perform_status_update():
    now = time.time()
    if not hasattr(game, "last_real_time"):
        game.last_real_time = now
        game.last_time_count = time_count
    elif now - game.last_real_time > STATUS_PRINT_PERIOD:
        elapsed_real = now - game.last_real_time
        elapsed_simulation = (time_count - game.last_time_count) / 1000
        speed_factor = elapsed_simulation / elapsed_real
        messages = [f"Avg speed factor: {speed_factor:.3f} (over last {elapsed_real:.2f} seconds)"]
        if game.state is None:
            messages.append("No messages received from GameController yet")
        else:
            messages.append(f"state: {game.state.game_state}, remaining time: {game.state.seconds_remaining}")
            if game.state.secondary_state in GAME_INTERRUPTIONS:
                messages.append(f"  sec_state: {game.state.secondary_state} phase: {game.state.secondary_state_info[1]}")
        messages = [f"STATUS: {m}" for m in messages]
        info(messages)
        game.last_real_time = now
        game.last_time_count = time_count


def toss_a_coin_if_needed(attribute):  # attribute should be either "side_left" or "kickoff"
    # If game.json contains such an attribute, use it to determine field side and kick-off
    # Supported values are "red", "blue" and "random". Default value is "random".
    if hasattr(game, attribute):
        if getattr(game, attribute) == 'red':
            setattr(game, attribute, game.red.id)
        elif getattr(game, attribute) == 'blue':
            setattr(game, attribute, game.blue.id)
        elif getattr(game, attribute) != 'random':
            error(f'Unsupported value for "{attribute}" in game.json file: {getattr(game, attribute)}, using "random".')
            setattr(game, attribute, 'random')
    else:
        setattr(game, attribute, 'random')
    if getattr(game, attribute) == 'random':  # toss a coin to determine a random team
        setattr(game, attribute, game.red.id if bool(random.getrandbits(1)) else game.blue.id)


def spawn_team(team, red_on_right, children):
    color = team['color']
    nb_players = len(team['players'])
    for number in team['players']:
        player = team['players'][number]
        model = player['proto']
        n = int(number) - 1
        port = game.red.ports[n] if color == 'red' else game.blue.ports[n]
        if red_on_right:  # symmetry with respect to the central line of the field
            flip_poses(player)
        defname = color.upper() + '_PLAYER_' + number
        halfTimeStartingTranslation = player['halfTimeStartingPose']['translation']
        halfTimeStartingRotation = player['halfTimeStartingPose']['rotation']
        string = f'DEF {defname} {model}{{name "{color} player {number}" translation ' + \
            f'{halfTimeStartingTranslation[0]} {halfTimeStartingTranslation[1]} {halfTimeStartingTranslation[2]} rotation ' + \
            f'{halfTimeStartingRotation[0]} {halfTimeStartingRotation[1]} {halfTimeStartingRotation[2]} ' + \
            f'{halfTimeStartingRotation[3]} controllerArgs ["{port}" "{nb_players}"'
        hosts = game.red.hosts if color == 'red' else game.blue.hosts
        for h in hosts:
            string += f', "{h}"'
        string += '] }}'
        children.importMFNodeFromString(-1, string)
        player['robot'] = supervisor.getFromDef(defname)
        player['position'] = player['robot'].getCenterOfMass()
        info(f'Spawned {defname} {model} on port {port} at halfTimeStartingPose: translation (' +
             f'{halfTimeStartingTranslation[0]} {halfTimeStartingTranslation[1]} {halfTimeStartingTranslation[2]}), ' +
             f'rotation ({halfTimeStartingRotation[0]} {halfTimeStartingRotation[1]} {halfTimeStartingRotation[2]} ' +
             f'{halfTimeStartingRotation[3]}).')


def format_time(s):
    seconds = str(s % 60)
    minutes = str(int(s / 60))
    if len(minutes) == 1:
        minutes = '0' + minutes
    if len(seconds) == 1:
        seconds = '0' + seconds
    return minutes + ':' + seconds


def update_time_display():
    if game.state:
        s = game.state.seconds_remaining
        if s < 0:
            s = -s
            sign = '-'
        else:
            sign = ' '
        value = format_time(s)
    else:
        sign = ' '
        value = '--:--'
    supervisor.setLabel(6, sign + value, 0, 0, game.font_size, 0x000000, 0.2, game.font)


def update_state_display():
    if game.state:
        state = game.state.game_state[6:]
        if state == 'READY' or state == 'SET':  # kickoff
            color = RED_COLOR if game.kickoff == game.red.id else BLUE_COLOR
        else:
            color = 0x000000
    else:
        state = ''
        color = 0x000000
    supervisor.setLabel(7, ' ' * 41 + state, 0, 0, game.font_size, color, 0.2, game.font)
    update_details_display()


def update_score_display():
    if game.state:
        red = 0 if game.state.teams[0].team_color == 'RED' else 1
        blue = 1 if red == 0 else 0
        red_score = str(game.state.teams[red].score)
        blue_score = str(game.state.teams[blue].score)
    else:
        red_score = '0'
        blue_score = '0'
    if game.side_left == game.blue.id:
        offset = 21 if len(blue_score) == 2 else 22
        score = ' ' * offset + blue_score + '-' + red_score
    else:
        offset = 21 if len(red_score) == 2 else 22
        score = ' ' * offset + red_score + '-' + blue_score
    supervisor.setLabel(5, score, 0, 0, game.font_size, BLACK_COLOR, 0.2, game.font)


def update_team_details_display(team, side, strings):
    for n in range(len(team['players'])):
        robot_info = game.state.teams[side].players[n]
        strings.background += '█  '
        if robot_info.number_of_warnings > 0:  # a robot can have both a warning and a yellow card
            strings.warning += '■  '
            strings.yellow_card += ' ■ ' if robot_info.number_of_yellow_cards > 0 else '   '
        else:
            strings.warning += '   '
            strings.yellow_card += '■  ' if robot_info.number_of_yellow_cards > 0 else '   '
        strings.red_card += '■  ' if robot_info.number_of_red_cards > 0 else '   '
        strings.white += str(n + 1) + '██'
        strings.foreground += f'{robot_info.secs_till_unpenalized:02d} ' if robot_info.secs_till_unpenalized != 0 else '   '


def update_details_display():
    if not game.state:
        return
    red = 0 if game.state.teams[0].team_color == 'RED' else 1
    blue = 1 if red == 0 else 0
    if game.side_left == game.red.id:
        left = red
        right = blue
        left_team = red_team
        right_team = blue_team
        left_color = RED_COLOR
        right_color = BLUE_COLOR
    else:
        left = blue
        right = red
        left_team = blue_team
        right_team = red_team
        left_color = BLUE_COLOR
        right_color = RED_COLOR

    class StringObject:
        pass

    strings = StringObject()
    strings.foreground = ' ' + format_time(game.state.secondary_seconds_remaining) + '  ' \
                         if game.state.secondary_seconds_remaining > 0 else ' ' * 8
    strings.background = ' ' * 7
    strings.warning = strings.background
    strings.yellow_card = strings.background
    strings.red_card = strings.background
    strings.white = '█' * 7
    update_team_details_display(left_team, left, strings)
    strings.left_background = strings.background
    strings.background = ' ' * 28
    space = 21 - len(left_team['players']) * 3
    strings.white += '█' * space
    strings.warning += ' ' * space
    strings.yellow_card += ' ' * space
    strings.red_card += ' ' * space
    strings.foreground += ' ' * space
    update_team_details_display(right_team, right, strings)
    strings.right_background = strings.background
    del strings.background
    space = 12 - 3 * len(right_team['players'])
    strings.white += '█' * (22 + space)
    secondary_state = ' ' * 41 + game.state.secondary_state[6:]
    sr = IN_PLAY_TIMEOUT - game.interruption_seconds + game.state.seconds_remaining \
        if game.interruption_seconds is not None else 0
    if sr > 0:
        secondary_state += ' ' + format_time(sr)
    if game.state.secondary_state[6:] != 'NORMAL' or game.state.secondary_state_info[1] != 0:
        secondary_state += ' [' + str(game.state.secondary_state_info[1]) + ']'
    if game.interruption_team is not None:  # interruption
        secondary_state_color = RED_COLOR if game.interruption_team == game.red.id else BLUE_COLOR
    else:
        secondary_state_color = BLACK_COLOR
    y = 0.0465  # vertical position of the second line
    supervisor.setLabel(10, strings.left_background, 0, y, game.font_size, left_color, 0.2, game.font)
    supervisor.setLabel(11, strings.right_background, 0, y, game.font_size, right_color, 0.2, game.font)
    supervisor.setLabel(12, strings.white, 0, y, game.font_size, WHITE_COLOR, 0.2, game.font)
    supervisor.setLabel(13, strings.warning, 0, 2 * y, game.font_size, 0x0000ff, 0.2, game.font)
    supervisor.setLabel(14, strings.yellow_card, 0, 2 * y, game.font_size, 0xffff00, 0.2, game.font)
    supervisor.setLabel(15, strings.red_card, 0, 2 * y, game.font_size, 0xff0000, 0.2, game.font)
    supervisor.setLabel(16, strings.foreground, 0, y, game.font_size, BLACK_COLOR, 0.2, game.font)
    supervisor.setLabel(17, secondary_state, 0, y, game.font_size, secondary_state_color, 0.2, game.font)


def update_team_display():
    # red and blue backgrounds
    left_color = RED_COLOR if game.side_left == game.red.id else BLUE_COLOR
    right_color = BLUE_COLOR if game.side_left == game.red.id else RED_COLOR
    supervisor.setLabel(2, ' ' * 7 + '█' * 14, 0, 0, game.font_size, left_color, 0.2, game.font)
    supervisor.setLabel(3, ' ' * 26 + '█' * 14, 0, 0, game.font_size, right_color, 0.2, game.font)
    # white background and names
    left_team = red_team if game.side_left == game.red.id else blue_team
    right_team = red_team if game.side_left == game.blue.id else blue_team
    team_names = 7 * '█' + (13 - len(left_team['name'])) * ' ' + left_team['name'] + \
        ' █████ ' + right_team['name'] + ' ' * (13 - len(right_team['name'])) + '█' * 22
    supervisor.setLabel(4, team_names, 0, 0, game.font_size, WHITE_COLOR, 0.2, game.font)
    update_score_display()


def setup_display():
    update_team_display()
    update_time_display()
    update_state_display()


def team_index(color):
    if color not in ['red', 'blue']:
        raise RuntimeError(f'Wrong color passed to team_index(): \'{color}\'.')
    id = game.red.id if color == 'red' else game.blue.id
    index = 0 if game.state.teams[0].team_number == id else 1
    if game.state.teams[index].team_number != id:
        raise RuntimeError(f'Wrong team number set in team_index(): {id} != {game.state.teams[index].team_number}')
    return index

def distance2(v1, v2):
    return math.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)


def append_solid(solid, solids, tagged_solids, active_tag=None):  # we list only the hands and feet
    name_field = solid.getField('name')
    if name_field:
        name = name_field.getSFString()
        tag_start = name.rfind('[')
        tag_end = name.rfind(']')
        if tag_start != -1 and tag_end != -1:
            active_tag = name[tag_start+1:tag_end]
        if name.endswith("[hand]") or name.endswith("[foot]"):
            solids.append(solid)
        if active_tag is not None:
            tagged_solids[name] = active_tag
    children = solid.getProtoField('children') if solid.isProto() else solid.getField('children')
    for i in range(children.getCount()):
        child = children.getMFNode(i)
        if child.getType() in [Node.ROBOT, Node.SOLID, Node.GROUP, Node.TRANSFORM, Node.ACCELEROMETER, Node.CAMERA, Node.GYRO,
                               Node.TOUCH_SENSOR]:
            append_solid(child, solids, tagged_solids, active_tag)
            continue
        if child.getType() in [Node.HINGE_JOINT, Node.HINGE_2_JOINT, Node.SLIDER_JOINT, Node.BALL_JOINT]:
            endPoint = child.getProtoField('endPoint') if child.isProto() else child.getField('endPoint')
            solid = endPoint.getSFNode()
            if solid.getType() == Node.NO_NODE or solid.getType() == Node.SOLID_REFERENCE:
                continue
            append_solid(solid, solids, tagged_solids, None)  # active tag is reset after a joint


def list_player_solids(player, color, number):
    robot = player['robot']
    player['solids'] = []
    player['tagged_solids'] = {}  # Keys: name of solid, Values: name of tag
    solids = player['solids']
    append_solid(robot, solids, player['tagged_solids'])
    if len(solids) != 4:
        info(f"Tagged solids: {player['tagged_solids']}")
        error(f'{color} player {number}: invalid number of [hand]+[foot], received {len(solids)}, expected 4.',
              fatal=True)


def list_team_solids(team):
    for number in team['players']:
        list_player_solids(team['players'][number], team['color'], number)


def list_solids():
    list_team_solids(red_team)
    list_team_solids(blue_team)


def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def show_polygon(vertices):
    polygon = supervisor.getFromDef('POLYGON')
    if polygon:
        polygon.remove()
    material = 'Material { diffuseColor 1 1 0 }'
    appearance = f'Appearance {{ material {material} }}'
    point = 'point ['
    for vertex in vertices:
        point += ' ' + str(vertex[0]) + ' ' + str(vertex[1]) + f' {game.field.turf_depth + 0.001},'  # 1 mm above turf
    point = point[:-1]
    point += ' ]'
    coord = f'Coordinate {{ {point} }}'
    coordIndex = '['
    for i in range(len(vertices)):
        coordIndex += ' ' + str(i)
    coordIndex += ' -1 ]'
    geometry = f'IndexedFaceSet {{ coord {coord} coordIndex {coordIndex} }}'
    shape = f'DEF POLYGON Shape {{ appearance {appearance} geometry {geometry} castShadows FALSE isPickable FALSE }}'
    children = supervisor.getRoot().getField('children')
    children.importMFNodeFromString(-1, shape)


def init_team(team):
    # check validity of team files
    # the players IDs should be "1", "2", "3", "4" for four players, "1", "2", "3" for three players, etc.
    count = 1
    for number in team['players']:
        count += 1
        player = team['players'][number]
        player['outside_circle'] = True
        player['outside_field'] = True
        player['inside_field'] = False
        player['on_outer_line'] = False
        player['inside_own_side'] = False
        player['outside_goal_area'] = True
        player['outside_penalty_area'] = True
        player['left_turf_time'] = None
        # Stores tuples of with (time_count[int], dic) at a 1Hz frequency
        player['history'] = []
        window_size = int(1000 / time_step)  # one second window size
        player['velocity_buffer'] = [[0] * 6] * window_size
        player['ball_handling_start'] = None
        player['ball_handling_last'] = None
        player['contact_points'] = []


def set_ball_touched(team_color, player_number):
    game.ball_previous_touch_team = game.ball_last_touch_team
    game.ball_previous_touch_player_number = game.ball_last_touch_player_number
    game.ball_last_touch_team = team_color
    game.ball_last_touch_player_number = player_number
    game.dropped_ball = False


def reset_ball_touched():
    game.ball_previous_touch_team = 'blue'
    game.ball_previous_touch_player_number = 1
    game.ball_last_touch_team = 'blue'
    game.ball_last_touch_player_number = 1


def flip_pose(pose):
    pose['translation'][0] = -pose['translation'][0]
    pose['rotation'][3] = math.pi - pose['rotation'][3]


def flip_poses(player):
    flip_pose(player['halfTimeStartingPose'])
    flip_pose(player['reentryStartingPose'])
    flip_pose(player['shootoutStartingPose'])
    flip_pose(player['goalKeeperStartingPose'])


def interruption(interruption_type, team=None, location=None, is_goalkeeper_ball_manipulation=False):
    if location is not None:
        game.ball_kick_translation[:2] = location[:2]
    if interruption_type == 'FREEKICK':
        own_side = (game.side_left == team) ^ (game.ball_position[0] < 0)
        inside_penalty_area = game.field.circle_fully_inside_penalty_area(game.ball_position, game.ball_radius)
        if inside_penalty_area and own_side:
            if is_goalkeeper_ball_manipulation:
                # move the ball on the penalty line parallel to the goal line
                dx = game.field.size_x - game.field.penalty_area_length
                dy = game.field.penalty_area_width / 2
                moved = False
                if abs(location[0]) > dx:
                    game.ball_kick_translation[0] = dx * (-1 if location[0] < 0 else 1)
                    moved = True
                if abs(location[1]) > dy:
                    game.ball_kick_translation[1] = dy * (-1 if location[1] < 0 else 1)
                    moved = True
                if moved:
                    info(f'Moved the ball on the penalty line at {game.ball_kick_translation}')
                interruption_type = 'INDIRECT_FREEKICK'
            else:
                interruption_type = 'PENALTYKICK'
                ball_reset_location = [game.field.penalty_mark_x, 0]
                if location[0] < 0:
                    ball_reset_location[0] *= -1
        else:
            interruption_type = 'DIRECT_FREEKICK'
        game.can_score = interruption_type != 'INDIRECT_FREEKICK'
    game.in_play = None
    game.can_score_own = False
    game.ball_set_kick = True
    game.interruption = interruption_type
    game.phase = interruption_type
    game.ball_first_touch_time = 0
    game.interruption_countdown = SIMULATED_TIME_INTERRUPTION_PHASE_0
    info(f'Interruption countdown set to {game.interruption_countdown}')
    if not team:
        game.interruption_team = game.red.id if game.ball_last_touch_team == 'blue' else game.blue.id
    else:
        game.interruption_team = team
    game.ball_must_kick_team = 'red' if game.interruption_team == game.red.id else 'blue'
    reset_ball_touched()
    info(f'Ball not in play, will be kicked by a player from the {game.ball_must_kick_team} team.')
    color = 'red' if game.interruption_team == game.red.id else 'blue'
    info(f'{GAME_INTERRUPTIONS[interruption_type].capitalize()} awarded to {color} team.')
    #game_controller_send(f'{game.interruption}:{game.interruption_team}')


def move_ball_away():
    """Places ball far away from field for phases where the referee is supposed to hold it in it's hand"""
    target_location = [100, 100, game.ball_radius + 0.05]
    game.ball.resetPhysics()
    game.ball_translation.setSFVec3f(target_location)
    info("Moved ball out of the field temporarily")


def kickoff():
    color = 'red' if game.kickoff == game.red.id else 'blue'
    info(f'Kick-off is {color}.')
    game.phase = 'KICKOFF'
    game.ball_kick_translation[0] = 0
    game.ball_kick_translation[1] = 0
    game.ball_set_kick = True
    game.ball_first_touch_time = 0
    game.in_play = None
    game.ball_must_kick_team = color
    reset_ball_touched()
    game.ball_left_circle = None  # one can score only after ball went out of the circle
    game.can_score = False        # or was touched by another player
    game.can_score_own = False
    game.kicking_player_number = None
    move_ball_away()
    info(f'Ball not in play, will be kicked by a player from the {game.ball_must_kick_team} team.')


def read_team(json_path):
    team = None
    try:
        with open(json_path) as json_file:
            team = json.load(json_file)
            for field_name in ["name", "players"]:
                if field_name not in team:
                    raise RuntimeError(f"Missing field {field_name}")
            if len(team['players']) == 0:
                warning(f"No players found for team {team['name']}")
            count = 1
            for p_key, p in team['players'].items():
                if int(p_key) != count:
                    raise RuntimeError(f'Wrong team player number: expecting "{count}", found "{p_key}".')
                for field_name in ['proto', 'halfTimeStartingPose', 'reentryStartingPose', 'shootoutStartingPose',
                                   'goalKeeperStartingPose']:
                    if field_name not in p:
                        raise RuntimeError(f"Missing field {field_name} in player {p_key}")
                count += 1
    except Exception:
        error(f"Failed to read file {json_path} with the following error:\n{traceback.format_exc()}", fatal=True)
    return team

def set_positions_shared_memory():
    with open("/tmp/position.txt", "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        pos = game.ball_position
        rot = game.ball.getField('rotation').getSFRotation()
        data = "ball " + format(pos[0], '.3f') + " " + format(pos[1], '.3f') + " " + format(pos[2], '.3f') + " " + format(rot[0], '.3f')  + " " + format(rot[1], '.3f')  + " " + format(rot[2], '.3f')  + " " + format(rot[3], '.3f') + " \r\n"
        for team in [blue_team, red_team]:
            for number in team['players']:
                player = team['players'][number]
                if player['robot'] is None:
                    continue
                pos = player['position']
                rot = player['robot'].getField('rotation').getSFRotation()
                data += team['color'] + str(number) + " " + format(pos[0], '.3f') + " " + format(pos[1], '.3f') + " " + format(pos[2], '.3f') + " " + format(rot[0], '.3f')  + " " + format(rot[1], '.3f')  + " " + format(rot[2], '.3f')  + " " + format(rot[3], '.3f') + " \r\n"
        mm[0:1000] = (' '*1000).encode()
        mm[0:len(data)] = data.encode()
        mm.close()

with open("/tmp/position.txt", "w") as f:
    f.write(str(' '*1000))

is_simple = False
if len(sys.argv) > 1:
    is_simple = True if sys.argv[1] == "simple" else False

# start the webots supervisor
supervisor = Supervisor()
time_step = int(supervisor.getBasicTimeStep())
time_count = 0

log_file = open('log.txt', 'w')

# determine configuration file name
game_config_file = os.environ['WEBOTS_ROBOCUP_GAME'] if 'WEBOTS_ROBOCUP_GAME' in os.environ \
    else os.path.join(os.getcwd(), 'game.json')

if is_simple:
    game_config_file = os.path.join(os.getcwd(), 'game_simple.json')

if not os.path.isfile(game_config_file):
    error(f'Cannot read {game_config_file} game config file.', fatal=True)

# read configuration files
with open(game_config_file) as json_file:
    game = json.loads(json_file.read(), object_hook=lambda d: SimpleNamespace(**d))
red_team = read_team(game.red.config)
blue_team = read_team(game.blue.config)
# if the game.json file is malformed with ids defined as string instead of int, we need to convert them to int:
if not isinstance(game.red.id, int):
    game.red.id = int(game.red.id)
if not isinstance(game.blue.id, int):
    game.blue.id = int(game.blue.id)

# finalize the game object
if not hasattr(game, 'minimum_real_time_factor'):
    game.minimum_real_time_factor = 3  # we garantee that each time step lasts at least 3x simulated time
if game.minimum_real_time_factor == 0:  # speed up non-real time tests
    REAL_TIME_BEFORE_FIRST_READY_STATE = 5
    HALF_TIME_BREAK_REAL_TIME_DURATION = 2
if not hasattr(game, 'press_a_key_to_terminate'):
    game.press_a_key_to_terminate = False
if game.type not in ['NORMAL', 'KNOCKOUT', 'PENALTY']:
    error(f'Unsupported game type: {game.type}.', fatal=True)
game.penalty_shootout = game.type == 'PENALTY'
info(f'Minimum real time factor is set to {game.minimum_real_time_factor}.')
if game.minimum_real_time_factor == 0:
    info('Simulation will run as fast as possible, real time waiting times will be minimal.')
else:
    info(f'Simulation will guarantee a maximum {1 / game.minimum_real_time_factor:.2f}x speed for each time step.')
field_size = getattr(game, 'class').lower()
game.field = Field(field_size)


red_team['color'] = 'red'
blue_team['color'] = 'blue'
init_team(red_team)
init_team(blue_team)

# check team name length (should be at most 12 characters long, trim them if too long)
if len(red_team['name']) > 12:
    red_team['name'] = red_team['name'][:12]
if len(blue_team['name']) > 12:
    blue_team['name'] = blue_team['name'][:12]

# check if the host parameter of the game.json file correspond to the actual host
host = socket.gethostbyname(socket.gethostname())
if host != '127.0.0.1' and host != game.host:
    warning(f'Host is not correctly defined in game.json file, it should be {host} instead of {game.host}.')

toss_a_coin_if_needed('side_left')
toss_a_coin_if_needed('kickoff')

children = supervisor.getRoot().getField('children')
if is_simple:
    children.importMFNodeFromString(-1, f'RobocupSoccerField_simple {{ size "{field_size}" }}')
else:
    children.importMFNodeFromString(-1, f'RobocupSoccerField {{ size "{field_size}" }}')

ball_size = 1 if field_size == 'kid' else 5
# the ball is initially very far away from the field
children.importMFNodeFromString(-1, f'DEF BALL RobocupSoccerBall {{ translation 100 100 0.5 size {ball_size} }}')

game.state = None
game.font_size = 0.096
game.font = 'Lucida Console'
spawn_team(red_team, game.side_left == game.blue.id, children)
spawn_team(blue_team, game.side_left == game.red.id, children)
setup_display()

SIMULATED_TIME_INTERRUPTION_PHASE_0 = int(SIMULATED_TIME_INTERRUPTION_PHASE_0 * 1000 / time_step)
SIMULATED_TIME_BEFORE_PLAY_STATE = int(SIMULATED_TIME_BEFORE_PLAY_STATE * 1000 / time_step)
SIMULATED_TIME_SET_PENALTY_SHOOTOUT = int(SIMULATED_TIME_SET_PENALTY_SHOOTOUT * 1000 / time_step)
players_ball_holding_time_window_size = int(1000 * PLAYERS_BALL_HOLDING_TIMEOUT / time_step)
goalkeeper_ball_holding_time_window_size = int(1000 * GOALKEEPER_BALL_HOLDING_TIMEOUT / time_step)
red_team['players_holding_time_window'] = np.zeros(players_ball_holding_time_window_size, dtype=bool)
red_team['goalkeeper_holding_time_window'] = np.zeros(goalkeeper_ball_holding_time_window_size, dtype=bool)
blue_team['players_holding_time_window'] = np.zeros(players_ball_holding_time_window_size, dtype=bool)
blue_team['goalkeeper_holding_time_window'] = np.zeros(goalkeeper_ball_holding_time_window_size, dtype=bool)


list_solids()  # prepare lists of solids to monitor in each robot to compute the convex hulls

game.penalty_shootout_count = 0
game.penalty_shootout_goal = False
game.penalty_shootout_time_to_score = [None, None, None, None, None, None, None, None, None, None]
game.penalty_shootout_time_to_reach_goal_area = [None, None, None, None, None, None, None, None, None, None]
game.penalty_shootout_time_to_touch_ball = [None, None, None, None, None, None, None, None, None, None]
game.ball = supervisor.getFromDef('BALL')
game.ball_radius = 0.07 if field_size == 'kid' else 0.1125
game.ball_kick_translation = [0, 0, game.ball_radius + game.field.turf_depth]  # initial position of ball before kick
game.ball_translation = supervisor.getFromDef('BALL').getField('translation')
game.ball_exit_translation = None
reset_ball_touched()
game.ball_last_touch_time = 0
game.ball_first_touch_time = 0
game.ball_last_touch_time_for_display = 0
game.ball_position = [0, 0, 0]
game.ball_last_move = 0
game.real_time_multiplier = 1000 / (game.minimum_real_time_factor * time_step) if game.minimum_real_time_factor > 0 else 10
game.interruption = None
game.interruption_countdown = 0
game.interruption_step = None
game.interruption_step_time = 0
game.interruption_team = None
game.interruption_seconds = None
game.dropped_ball = False
game.overtime = False
game.finished_overtime = False
game.ready_countdown = 0  # simulated time countdown before ready state (used in kick-off after goal and dropped ball)
game.play_countdown = 0
game.in_play = None
game.throw_in = False  # True while throwing in to allow ball handling
game.throw_in_ball_was_lifted = False  # True if the throwing-in player lifted the ball
game.over = False
game.wait_for_state = 'INITIAL'
game.wait_for_sec_state = None
game.wait_for_sec_phase = None
game.forceful_contact_matrix = ForcefulContactMatrix(len(red_team['players']), len(blue_team['players']),
                                                     FOUL_PUSHING_PERIOD, FOUL_PUSHING_TIME, time_step)

previous_seconds_remaining = 0

try:
    update_state_display()
    info(f'Game type is {game.type}.')
    info(f'Red team is "{red_team["name"]}", playing on {"left" if game.side_left == game.red.id else "right"} side.')
    info(f'Blue team is "{blue_team["name"]}", playing on {"left" if game.side_left == game.blue.id else "right"} side.')

    if hasattr(game, 'supervisor'):  # optional supervisor used for CI tests
        children.importMFNodeFromString(-1, f'DEF TEST_SUPERVISOR Robot {{ supervisor TRUE controller "{game.supervisor}" }}')

    if game.penalty_shootout:
        info(f'{"Red" if game.kickoff == game.red.id else "Blue"} team will start the penalty shoot-out.')
        game.phase = 'PENALTY-SHOOTOUT'
        game.ready_real_time = None
        info(f'Penalty start: Waiting {REAL_TIME_BEFORE_FIRST_READY_STATE} seconds (real-time) before going to SET')
        game.set_real_time = time.time() + REAL_TIME_BEFORE_FIRST_READY_STATE  # real time for set state (penalty-shootout)
    else:
        info(f'Regular start: Waiting {REAL_TIME_BEFORE_FIRST_READY_STATE} seconds (real-time) before going to READY')
        game.ready_real_time = time.time() + REAL_TIME_BEFORE_FIRST_READY_STATE  # real time for ready state (initial kick-off)
        kickoff()
except Exception:
    error(f"Failed setting initial state: {traceback.format_exc()}", fatal=True)

try:
    previous_real_time = time.time()
    game.ball_translation.setSFVec3f(game.ball_position)
    while supervisor.step(time_step) != -1 and not game.over:
        set_positions_shared_memory()
        perform_status_update()
        send_play_state_after_penalties = False
        previous_position = copy.deepcopy(game.ball_position)
        game.ball_position = game.ball_translation.getSFVec3f()
        if game.ball_position != previous_position:
            game.ball_last_move = time_count
        for team in [blue_team, red_team]:
            for number in team['players']:
                player = team['players'][number]
                if player['robot'] is None:
                    continue
                player['position'] = player['robot'].getCenterOfMass()
        if True:

                game.ball_exit_translation = game.ball_position
                scoring_team = None
                if game.ball_position[0] + game.ball_radius < -game.field.size_x:
                    if game.ball_position[1] < GOAL_HALF_WIDTH and \
                       game.ball_position[1] > -GOAL_HALF_WIDTH and \
                       game.ball_position[2] < game.field.goal_height:
                        # goal
                        scoring_team = game.red.id if game.blue.id == game.side_left else game.blue.id
                        info(f'Goal')

except Exception:
    error(f"Unexpected exception in main referee loop: {traceback.format_exc()}", fatal=True)

clean_exit()
