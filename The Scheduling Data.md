## The Scheduling Data

---------------

### Features in the Scheduling Data

- Pairwise
  - Agent Idle: Is the agent currently working on any tasks? (IS IRRELEVANT)
  - Task Finished
  - Task Enabled: Tasks that the current task is dependent on have been started
  - Task Alive: Tasks that the current task is dependent on have been completed
  - Task Travel Satisfied: When the agent gets here, can it start?
  - Task Location Occupied 
  - Distance to Task (Continuous)
  - Orientation (Continuous)
  - Deadline (Continuous)
  - In Progress
  - \# of tasks at that location
  - orientation * distance (even though this is redundant, this says we are giving the network everything)
  - Alongside these usable features, I store the timestep, embedding values, and task scheduled (which is the output)

**At each timestep, N rows of data are recorded (for N number of tasks). Note, only timesteps where a task was scheduled is utilized. This was due to an issue with (90% of the data being schedule null task)**. In total, you can expect to have `N * num_total_timesteps_in_schedule * num_agents rows of data`. Out of this, only `N * N` will be useful data.

- Naive
  - The data is the same but now everything is vectorized
  - Each feature above basically turns into a vector of size 20 (except for is agent idle)
  - The total number of data rows is now `num_total_timesteps_in_schedule * num_agents`

A final note is that you can create new data fairly quickly using a file called create_data if you would like to try different sizes. 

-------

### Going through `nn_only_task.py`

1. Load in the .pkl file corresponding to the number of schedules you would like to train on, using `pickle.load(open(self.load_directory, "rb"))`
2. `self.create_new_data()`
   - so this is the function that parses through the data and turns it into a nice list of data 
   - This function steps through each schedule (finding the start and end of a schedule by looking at the timesteps)
   - I remove all timesteps where the task scheduled was null and I record the start and end of this schedule
   - There is a small helper function to append to X properly and grab the output from the .pkl file in `rebuild_input_output_from_pickle` but that function is pretty self explanatory
3. At the end of this, you have an array of all features for each timestep where things were scheduled, and another array called self.schedule_array that tells you when schedules start and end within that array
4. Finally, I just keep another helper list that holds the integer values where each timestep starts and completes

--------------------

### Training Procedure

```
For a random timestep within a random schedule:
	task scheduled = Find the task that was scheduled
	for each task not including task scheduled:
		diff = task_scheduled_features - each_task_not_scheduled_features
		output = NN.forward_pass(diff)
		Expected_Output is 1
		# NOTE NOW THE OUTPUT IS [1 0]
    for each task not including task scheduled:
		diff = -(task_scheduled_features - each_task_not_scheduled_features)
		output = BNN.forward_pass(diff)
		Expected_Output is 0
		# NOTE NOW THE OUTPUT IS [0 1]
	
```

### Testing Procedure

```
For each task:
	For each other task:
		if each_task = each_other_task:
			continue
		else: 
			diff = each_task_features - each_other_task_features
			preference = NN.forward_pass(diff)
			store preference in matrix of size total_tasks by total_tasks
		
Sum over columns of matrix
Task to schedule = argmax

Then, update parameters using training procedure
```

