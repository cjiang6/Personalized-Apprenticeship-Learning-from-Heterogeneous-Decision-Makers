"""
Run through a set of schedules to generate a dataset in a naive and pairwise fashion
"""
import sys
sys.path.insert(0, '/home/ghost/PycharmProjects/bayesian_prolo')

from scheduling_env.create_scheduling_data.world import *
from utils.global_utils import save_pickle


def main(str_for_naive, str_for_pickle):
    """
    entry point of the file
    :return:
    """
    total_schedule = []
    total_schedule_pairwise = []
    n = 0
    num_scheds = 1500
    test = False
    if test:
        np.random.seed(50)
        random.seed(50)
    else:
        np.random.seed(150)
        random.seed(150)
    while n < num_scheds:
        world = World(num_scheds,n)

        # 2 cases this breaks, either t > max duration, or 19 tasks are scheduled.
        while not world.data_done_generating:
            is_feasible = world.update_floyd_warshall_and_all_vectors()
            world.print_all_features()
            if is_feasible == False:
                print('constraint was infeasible')
                break
            for counter, agent in enumerate(world.agents):
                world.compute_task_to_schedule(counter)

                is_feasible = world.update_floyd_warshall_and_all_vectors()

            world.add_constraints_based_on_task()
            world.check_if_schedule_finished()
            print('current finish times', world.agents[0].getFinishTime()
                  , world.agents[1].getFinishTime())



            # Next Time Step
            world.t += 1
            print('current time is ',world.t)
            world.update_based_on_time()

        if world.did_schedule_fail:
            pass
        else:
            # temp = []
            # temp_pairwise = []
            # with open(world.filepath, 'r') as csvfile:
            #     data = (csv.reader(csvfile))
            #     for row in data:
            #         temp.append(row)
            #
            # with open(world.second_file_path, 'r') as csvfile:
            #     data = (csv.reader(csvfile))
            #     for row in data:
            #         temp_pairwise.append(row)

            total_schedule.append(world.naive_total_data)
            total_schedule_pairwise.append(world.pairwise_total_data)
            n += 1
            print('on schedule ', n)
            if n == 100:
                save_pickle(file_location='/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env', file=total_schedule,
                            special_string=str(100) + str_for_naive + '.pkl')
                save_pickle(file_location='/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env', file=total_schedule_pairwise,
                            special_string=str(100) + str_for_pickle + '.pkl')

            if n == 250:
                save_pickle(file_location='/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env', file=total_schedule,
                            special_string=str(250) + str_for_naive + '.pkl')
                save_pickle(file_location='/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env', file=total_schedule_pairwise,
                            special_string=str(250) + str_for_pickle + '.pkl')

            if n == 500:
                save_pickle(file_location='/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env', file=total_schedule,
                            special_string=str(500) + str_for_naive + '.pkl')
                save_pickle(file_location='/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env', file=total_schedule_pairwise,
                            special_string=str(500) + str_for_pickle + '.pkl')

            if n == 1500:
                save_pickle(file_location='/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env', file=total_schedule,
                            special_string=str(1500) + str_for_naive + '.pkl')
                save_pickle(file_location='/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env', file=total_schedule_pairwise,
                            special_string=str(1500) + str_for_pickle + '.pkl')



    save_pickle(file_location='/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env', file=total_schedule, special_string=str(num_scheds) + str_for_naive + '.pkl')
    save_pickle(file_location='/home/ghost/PycharmProjects/bayesian_prolo/scheduling_env', file=total_schedule_pairwise, special_string=str(num_scheds) + str_for_pickle + '.pkl')


if __name__ == main('dist_early_hili_naive','dist_early_hili_pairwise'):
    main('dist_early_hili_naive','dist_early_hili_pairwise')
