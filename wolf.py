import numpy as np



r1_nash_equilibrium_dict = {'0':'wait', '1':'pick','2':'deliver'}
r2_nash_equilibrium_dict = {'0':'wait', '1':'pick','2':'deliver'}

def q_function_initial(states, actions):

    return np.zeros((states,actions))

def policy_function(states, actions):

    probablity = 1/3

    return np.full((states,actions),probablity)


def average_policy_function(states, actions):

    policy_average_1 = np.array([[1,0,0],
                               [1,0,0],
                               [0,0,1],
                               [0,0,1],
                               [0,0.5,0],
                              [0,0.5,0],
                              [0,0,1],
                              [0,0,1]])

    policy_average_2 = np.array([[1, 0, 0],
                                 [0, 0, 1],
                                 [1, 0, 0],
                                 [0, 0, 1],
                                 [1, 0, 0],
                                 [0, 0, 1],
                                 [0, 0.5, 0],
                                 [0, 0, 1]])

    return policy_average_1,policy_average_2

def state_c_function(states):

    return np.zeros((states,1))

def rewards_robot_1():
    rewards = np.array([[0, -1,- 1],
                         [0, -1, -1],
                         [0, -1, 3],
                         [0, -1, 3],
                         [0, -1,  -1],
                         [0, -1, -1],
                         [0, -1, 3],
                         [0, -1, 3]])
    return rewards

def rewards_robot_2():
    rewards = np.array([[0, -1,- 1],
                         [0, -1, 3],
                         [0, -1, -1],
                         [0, -1, 3],
                         [0, -1,  -1],
                         [0, -1, 3],
                         [0, -1, -1],
                         [0, -1, 3]])
    return rewards

if __name__ == "__main__":

    alpha_loose = 0.09
    alpha_win = 0.035
    discount_factor = 0.9
    alpha = 0.001


    #robot 1 parameters

    q_initial_robot_1= q_function_initial(8, 3)
    policy_initial_robot_1 = policy_function(8 ,3)
    average_policy_robot_1 = average_policy_function(8, 3)[0]
    c_function_robot_1 = state_c_function(8)
    rewards_robot_1 = rewards_robot_1()


    #robot 2 parameters

    q_initial_robot_2 = q_function_initial(8, 3)
    policy_initial_robot_2 = policy_function(8, 3)
    average_policy_robot_2 = average_policy_function(8, 3)[1]
    c_function_robot_2 = state_c_function(8)
    rewards_robot_2 = rewards_robot_2()




    # calculation for robot 1

    for iterations in range(2000):

        for i in range(0,len(q_initial_robot_1)):

            c_function_robot_1[i] = c_function_robot_1[i] + 1
            # temp_action_list = action_list

            for j in range(0,3):


                if i is 7:

                    q_initial_robot_1[i][j] = (1 - alpha) * q_initial_robot_1[i][j] + alpha * (rewards_robot_1[i][j] + discount_factor * max(q_initial_robot_1[0]))

                    average_policy_robot_1[i][j] = average_policy_robot_1[i][j] + ((1/c_function_robot_1[i]) * (policy_initial_robot_1[i][j] - average_policy_robot_1[i][j]))

                    average_policy_product = average_policy_robot_1[i]
                    policy_product = policy_initial_robot_1[i]
                    q_values_product = np.transpose(q_initial_robot_1[i])

                    product = np.dot(policy_product, q_values_product)
                    product_average = np.dot(average_policy_product, q_values_product)

                    if product > product_average:

                        delta = alpha_win

                        if j is max(q_initial_robot_1[i]):

                            policy_initial_robot_1[i][j] = policy_initial_robot_1[i][j] + delta

                        else:

                            policy_initial_robot_1[i][j] = policy_initial_robot_1[i][j] + (- delta / 2)

                    else:

                        delta = alpha_loose

                        if j is max(q_initial_robot_1[i]):

                            policy_initial_robot_1[i][j] = policy_initial_robot_1[i][j] + delta

                        else:

                            policy_initial_robot_1[i][j] = policy_initial_robot_1[i][j] + (-delta / 2)


                else:

                    q_initial_robot_1[i][j] = (1 - alpha) * q_initial_robot_1[i][j] + alpha * (rewards_robot_1[i][j] + discount_factor * max(q_initial_robot_1[i + 1]))

                    average_policy_robot_1[i][j] = average_policy_robot_1[i][j] + ((1 / c_function_robot_1[i]) * (policy_initial_robot_1[i][j] - average_policy_robot_1[i][j]))

                    average_policy_product = average_policy_robot_1[i]
                    policy_product = policy_initial_robot_1[i]
                    q_values_product = np.transpose(q_initial_robot_1[i])

                    product = np.dot(policy_product,q_values_product)
                    product_average = np.dot(average_policy_product, q_values_product)

                    if product > product_average:

                        delta = alpha_win

                        if j is max(q_initial_robot_1[i]):

                          policy_initial_robot_1[i][j] = policy_initial_robot_1[i][j] + delta

                        else:

                            policy_initial_robot_1[i][j] = policy_initial_robot_1[i][j] + (-delta / 2)

                    else:

                        delta = alpha_loose

                        if j is max(q_initial_robot_1[i]):

                            policy_initial_robot_1[i][j] = policy_initial_robot_1[i][j] + delta

                        else:

                            policy_initial_robot_1[i][j] = policy_initial_robot_1[i][j] + (-delta / 2)



    # print(policy_initial_robot_1)

    # print("the policy matrix of robot 2")

    # calculation for robot 2

    for iterations in range(2000):

        for i in range(0,len(q_initial_robot_2)):

            c_function_robot_2[i] = c_function_robot_2[i] + 1

            for j in range(0,3):


                if i is 7:

                    q_initial_robot_2[i][j] = (1 - alpha) * q_initial_robot_2[i][j] + alpha * (rewards_robot_2[i][j] + discount_factor * max(q_initial_robot_2[0]))

                    average_policy_robot_2[i][j] = average_policy_robot_2[i][j] + ((1/c_function_robot_2[i]) * (policy_initial_robot_2[i][j] - average_policy_robot_2[i][j]))

                    average_policy_product = average_policy_robot_2[i]
                    policy_product = policy_initial_robot_2[i]
                    q_values_product = np.transpose(q_initial_robot_2[i])

                    product = np.dot(policy_product, q_values_product)
                    product_average = np.dot(average_policy_product, q_values_product)

                    if product > product_average:

                        delta = alpha_win

                        if j is max(q_initial_robot_2[i]):

                            policy_initial_robot_2[i][j] = policy_initial_robot_2[i][j] + delta

                        else:

                            policy_initial_robot_2[i][j] = policy_initial_robot_2[i][j] + (- delta / 2)

                    else:

                        delta = alpha_loose

                        if j is max(q_initial_robot_2[i]):

                            policy_initial_robot_2[i][j] = policy_initial_robot_2[i][j] + delta

                        else:

                            policy_initial_robot_2[i][j] = policy_initial_robot_2[i][j] + (-delta / 2)


                else:

                    q_initial_robot_2[i][j] = (1 - alpha) * q_initial_robot_2[i][j] + alpha * (rewards_robot_2[i][j] +  discount_factor * max(q_initial_robot_2[i + 1]))

                    average_policy_robot_2[i][j] = average_policy_robot_2[i][j] + ((1 / c_function_robot_2[i]) * (policy_initial_robot_2[i][j] - average_policy_robot_2[i][j]))

                    average_policy_product = average_policy_robot_2[i]
                    policy_product = policy_initial_robot_2[i]
                    q_values_product = np.transpose(q_initial_robot_2[i])

                    product = np.dot(policy_product,q_values_product)
                    product_average = np.dot(average_policy_product, q_values_product)

                    if product > product_average:

                        delta = alpha_win

                        if j is max(q_initial_robot_2[i]):

                          policy_initial_robot_2[i][j] = policy_initial_robot_2[i][j] + delta

                        else:

                            policy_initial_robot_2[i][j] = policy_initial_robot_2[i][j] + (-delta / 2)

                    else:

                        delta = alpha_loose

                        if j is max(q_initial_robot_2[i]):

                            policy_initial_robot_2[i][j] = policy_initial_robot_2[i][j] + delta

                        else:

                            policy_initial_robot_2[i][j] = policy_initial_robot_2[i][j] + (-delta / 2)



    print("Final Q values for Robot 1\n")
    print(q_initial_robot_1)
    print("\n")

    print("Final Q values for Robot 2\n")
    print(q_initial_robot_2)
    print("\n")

    print("Final Policy matrix for Robot 1\n")
    print(policy_initial_robot_1)
    print("\n")

    print("Final Policy matrix for Robot 2\n")
    print(policy_initial_robot_2)
    print("\n")


    # print(policy_initial_robot_2)

    max_value_index_r1 = []
    for ind in policy_initial_robot_1:

        max_value_index_r1.append(list(ind).index(max(list(ind))))


    max_value_index_r2 = []
    for ind in policy_initial_robot_2:
        max_value_index_r2.append(list(ind).index(max(list(ind))))

    # Nash Equilibrium of each state
    for i in range(8):

        temp_1 = ''
        temp_2 = ''

        print("Nash Equilibrium for State {}".format(i))


        for key, value in r1_nash_equilibrium_dict.items():

            if int(key) == max_value_index_r1[i]:
                temp_1 = value


        for key, value in r2_nash_equilibrium_dict.items():

            if int(key) == max_value_index_r2[i]:
                temp_2 = value

        print('{} , {}'.format(temp_1, temp_2))
        print("")
