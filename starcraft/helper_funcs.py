import pickle
import os
import torch
import numpy as np
import random
from torch.autograd import Variable
from sc2.constants import *


label_names_indexed_sort_of = ['BUILD_NEXUS', 'BUILD_PYLON', 'BUILD_ASSIMILATOR', 'BUILD_GATEWAY', 'MORPH_WARPGATE',
                               'BUILD_FORGE', 'BUILD_CYBERNETICSCORE', 'BUILD_PHOTONCANNON', 'BUILD_SHIELDBATTERY',
                               'BUILD_ROBOTICSFACILITY', 'BUILD_STARGATE', 'BUILD_TWILIGHTCOUNCIL', 'BUILD_ROBOTICSBAY',
                               'BUILD_FLEETBEACON', 'BUILD_TEMPLARARCHIVE', 'BUILD_DARKSHRINE', 'TRAIN_PROBE',
                               'TRAIN_ZEALOT', 'TRAIN_STALKER', 'TRAIN_SENTRY', 'TRAIN_ADEPT',
                               'TRAIN_HIGHTEMPLAR', 'TRAIN_DARKTEMPLAR', 'TRAIN_OBSERVER', 'TRAIN_WARPPRISM',
                               'TRAIN_IMMORTAL', 'TRAIN_COLOSSUS', 'TRAIN_DISRUPTOR', 'TRAIN_PHOENIX',
                               'TRAIN_VOIDRAY', 'TRAIN_ORACLE', 'TRAIN_TEMPEST', 'TRAIN_CARRIER',
                               'TRAINMOTHERSHIP']

unit_list = ['PROBE', 'ZEALOT', 'STALKER', 'SENTRY', 'ADEPT', 'HIGHTEMPLAR', 'DARKTEMPLAR', 'OBSERVER',
                     'WARPPRISM', 'IMMORTAL', 'COLOSSUS', 'DISRUPTOR', 'PHOENIX', 'VOIDRAY', 'ORACLE',
                     'TEMPEST', 'CARRIER', 'INTERCEPTOR', 'MOTHERSHIP', 'NEXUS', 'PYLON', 'ASSIMILATOR',
                     'GATEWAY', 'WARPGATE', 'FORGE', 'CYBERNETICSCORE', 'PHOTONCANNON', 'SHIELDBATTERY',
                     'ROBOTICSFACILITY', 'STARGATE', 'TWILIGHTCOUNCIL', 'ROBOTICSBAY', 'FLEETBEACON',
                     'TEMPLARARCHIVE', 'DARKSHRINE', 'ORACLESTASISTRAP']




def get_games_and_game_length(DATA_GAME_DIR):
    games_array_done = []
    c = 0
    for game_dir in os.listdir(DATA_GAME_DIR):
        # choose a random game from directory
        games_array_done.append(game_dir)
        c += 1
    print('Total # of games in directory is ' + str(c))
    return c, games_array_done

def does_game_exist_for_player(current_dir, player):
    game_dir_exists = os.path.exists(os.path.join(current_dir, player + 'visibility.pkl')) and \
                      os.path.exists(os.path.join(current_dir, player + 'my_counts.pkl')) and \
                      os.path.exists(os.path.join(current_dir, player + 'enemy_counts.pkl')) and \
                      os.path.exists(os.path.join(current_dir, player + 'state.pkl')) and \
                      os.path.exists(os.path.join(current_dir, player + 'actions.pkl')) and \
                      os.path.exists(os.path.join(current_dir, player + 'placement_grid.pkl')) and \
                      os.path.exists(os.path.join(current_dir, player + 'unit_info.pkl'))
    return game_dir_exists

def load_in_all_but_placement(current_dir, player):
    vis_maps = (pickle.load(open(os.path.join(current_dir, player + 'visibility.pkl'), 'rb')))
    my_counts = (pickle.load(open(os.path.join(current_dir, player + 'my_counts.pkl'), 'rb')))
    enemy_counts = (pickle.load(open(os.path.join(current_dir, player + 'enemy_counts.pkl'), 'rb')))
    my_states = (pickle.load(open(os.path.join(current_dir, player + 'state.pkl'), 'rb')))
    actions_all_game = pickle.load(open(os.path.join(current_dir, player + 'actions.pkl'), 'rb'))
    units_all_game = pickle.load(open(os.path.join(current_dir, player + 'unit_info.pkl'), 'rb'))
    return actions_all_game, enemy_counts, my_counts, my_states, units_all_game, vis_maps


def get_random_frames(length_of_game, num_frames):
    f = []
    for i in range(1,num_frames):
        frame = random.randint(1, length_of_game - 2)
        f.append(frame)
    return f


def parse_action_into_list(ATTACK, BUILD, HARVEST, SCOUT, action, correct_list, frame, maxIndex_labels, units_all_game):
    option_found = False
    for ability_string in label_names_indexed_sort_of:
        if ability_string in AbilityId(action[1]).name:
            # check if they are in the big set
            correct_list.add(label_names_indexed_sort_of.index(ability_string))
            option_found = True
            break
    if not option_found:
        if 'ATTACK' in AbilityId(action[1]).name:
            correct_list.add(maxIndex_labels + ATTACK)
        elif 'BUILD' in AbilityId(action[1]).name:
            correct_list.add(maxIndex_labels + BUILD)
        elif 'TRAIN' in AbilityId(action[1]).name:
            correct_list.add(maxIndex_labels + BUILD)
        elif 'HARVEST' in AbilityId(action[1]).name:
            correct_list.add(maxIndex_labels + HARVEST)
        elif 'SMART' in AbilityId(action[1]).name:
            # check unit
            unit_tag = action[0]
            frame_units = np.array(units_all_game[frame])
            tag_indexes = frame_units[:, 1].tolist()
            tag_indexes = [int(x) for x in tag_indexes]
            my_probes = frame_units[frame_units[:, 0] == unit_list.index('PROBE')]
            for unit in units_all_game[frame]:
                if unit[16] == unit_tag[0]:  # if you find which unit of yours this is
                    thisUnit = unit
                    unitType = unit_list[thisUnit[0]]  # Find its type
                    dist_to_likely_base = 1000
                    for probe in my_probes:
                        # Assuming bases are full of probes, estimate distance to -
                        # - home by distance to nearest probe. This will _not_ catch worker rushes
                        if probe[-1] == thisUnit[-1]:  # If I'm a probe, don't count myself
                            continue
                        how_far_am_i = dist_bw(probe, thisUnit)
                        if how_far_am_i < dist_to_likely_base:
                            dist_to_likely_base = how_far_am_i
                    if dist_to_likely_base > 20:
                        if unitType in ['OBSERVER', 'PROBE']:
                            correct_list.add(maxIndex_labels + SCOUT)
                            # Observer or probe not home, probably off scouting
                        else:
                            correct_list.add(maxIndex_labels + ATTACK)


                    elif unitType == 'PROBE':
                        # Close to home and smart action, probably clicked a mineral patch or something
                        correct_list.add(maxIndex_labels + HARVEST)
            if len(correct_list) < 1:
                correct_list.add(39)

    return correct_list

def get_frame_data(enemy_counts, frame, my_counts, my_states, placement_grids, vis_maps):
    current_vis = vis_maps[frame]
    current_psi = placement_grids[frame][:, :, 0] / 3
    current_enemy_loc = placement_grids[frame][:, :, 1]
    current_unit_loc = placement_grids[frame][:, :, 2]
    current_build_grid = placement_grids[frame][:, :, 3] / 255
    current_unit_count = my_counts[frame]
    current_enemy_count = enemy_counts[frame]
    current_player_state = my_states[frame]
    return current_build_grid, current_enemy_count, current_enemy_loc, current_player_state, current_psi, current_unit_count, current_unit_loc, current_vis

def get_torch_variables(current_build_grid, current_enemy_count, current_enemy_loc, current_player_state, current_psi,
                        current_unit_count, current_unit_loc, current_vis, frame_actions):
    if torch.cuda.is_available():
        # TRANSFORM TO TORCH VARIABLES
        vector_player_state = Variable(torch.Tensor(np.asarray(current_player_state).reshape(1, 9)).cuda())
        vector_player_count = Variable(torch.Tensor(np.asarray(current_unit_count).reshape(1, 36)).cuda())
        vector_enemy_count = Variable(torch.Tensor(np.asarray(current_enemy_count).reshape(1, 112)).cuda())
        fu_action = Variable(torch.Tensor(np.asarray(frame_actions).reshape(1, 40)).cuda())

        vis6464 = Variable(torch.Tensor(current_vis.reshape(1, 1, 64, 64)).cuda())

        psionic = Variable(torch.Tensor(current_psi.reshape(1, 1, 180, 200)).cuda())
        enemy_loc = Variable(torch.Tensor(current_enemy_loc.reshape(1, 1, 180, 200)).cuda())
        our_loc = Variable(torch.Tensor(current_unit_loc.reshape(1, 1, 180, 200)).cuda())
        legal_placement = Variable(torch.Tensor(current_build_grid.reshape(1, 1, 180, 200)).cuda())

        label = fu_action
    else:
        vector_player_state = Variable(torch.Tensor(np.asarray(current_player_state).reshape(1, 9)))
        vector_player_count = Variable(torch.Tensor(np.asarray(current_unit_count).reshape(1, 36)))
        vector_enemy_count = Variable(torch.Tensor(np.asarray(current_enemy_count).reshape(1, 112)))
        fu_action = Variable(torch.Tensor(np.asarray(frame_actions).reshape(1, 40)))

        vis6464 = Variable(torch.Tensor(current_vis.reshape(1, 1, 64, 64)))

        psionic = Variable(torch.Tensor(current_psi.reshape(1, 1, 180, 200)))
        enemy_loc = Variable(torch.Tensor(current_enemy_loc.reshape(1, 1, 180, 200)))
        our_loc = Variable(torch.Tensor(current_unit_loc.reshape(1, 1, 180, 200)))
        legal_placement = Variable(torch.Tensor(current_build_grid.reshape(1, 1, 180, 200)))

        label = Variable(torch.Tensor(fu_action.reshape(1, 40)))
    return enemy_loc, label, legal_placement, our_loc, psionic, \
           vector_enemy_count, vector_player_count, vector_player_state, vis6464

def load_in_embedding(NeuralNet, embedding_list, player_id):
    curr_embedding = embedding_list[player_id]
    curr_dict = NeuralNet.state_dict()
    curr_dict['EmbeddingList.0.embedding'] = curr_embedding.clone()
    NeuralNet.load_state_dict(curr_dict)

def batch_load_in_embedding(NeuralNet, embedding_list, player_id,i):
    curr_embedding = embedding_list[player_id]
    curr_dict = NeuralNet.state_dict()
    curr_dict['EmbeddingList.' + str(i) + '.embedding'] = curr_embedding
    NeuralNet.load_state_dict(curr_dict)


def store_embedding_back(NeuralNet, embedding_list, player_id, DEBUG = False):
    curr_dict = NeuralNet.state_dict()
    new_embedding = curr_dict['EmbeddingList.0.embedding']
    curr_embedding = embedding_list[player_id]
    if DEBUG:
        print(curr_embedding)
        print(new_embedding)
    embedding_list[player_id] = new_embedding.clone()

def batch_store_embedding_back(NeuralNet, embedding_list, player_id, DEBUG = False):
    curr_dict = NeuralNet.state_dict()
    new_embedding = curr_dict['EmbeddingList.0.embedding']
    curr_embedding = embedding_list[player_id]
    if DEBUG:
        print(curr_embedding)
        print(new_embedding)
    embedding_list[player_id] = new_embedding.clone()

def dist_bw(unit1, unit2):
    return np.sqrt((unit1[1] - unit2[1]) ** 2 + (unit1[2] - unit2[2]) ** 2)
