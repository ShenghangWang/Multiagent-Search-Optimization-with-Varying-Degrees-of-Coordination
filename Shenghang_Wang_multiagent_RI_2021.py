import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from matplotlib import cm

def main():
    exercise = 3
    run(exercise)


def run(exercise):
    if exercise == 1:
        exercise1()
    elif exercise == 2:
        exercise2()
    elif exercise == 3:
        exercise3()


def bivariate_gaussian(size, mu, Sigma):
    x = np.arange(size[0])
    y = np.arange(size[1])
    X, Y = np.meshgrid(x, y)  # generate grid
    Sigma_1 = np.linalg.inv(Sigma)  # Compute the inverse Sigma^-1
    p = np.zeros(size[0] * size[1])  # probability map
    d = np.array([X.ravel() - mu[0], Y.ravel() - mu[1]])  # distances (x- \mu)
    for i in range(0, size[0] * size[1]):
        p[i] = np.exp(- 0.5 * d[:, i].dot(Sigma_1).dot(d[:, i].transpose()))
    p = p / np.sum(p)  # normalize to sum to 1
    return p


# Exercise 1
def exercise1():
    class Agent:

        # Constructor
        def __init__(self, i, state=np.array([0., 0.]), map_size=np.array([40, 40]), bk=np.ones(1600) * (1. / 1600)):

            self.id = i  # Create an instance variable
            self.map_size = map_size

            ## Environment. Tip it would be efficient to compute the grid into one vector as we did in the previous notebook

            self.bk = bk  # belief at instant k

            ## Agent params
            self.x = state
            self.track = self.x.T  # Stores all positions. Aux variable for visualization of the agent path
            self.height_plot = 0.1

            # add extra agents params
            self.width = map_size[0]
            self.height = map_size[1]
            self.X, self.Y = np.meshgrid(np.arange(self.width), np.arange(self.height))

            ## Sensor params

        # get id
        def get_id(self):
            return self.id

        # compute discrete forward states 1 step ahead
        def forward(self):
            fs = np.zeros((0, 2))
            w_min = np.array([0, 0])
            w_max = np.array([self.width, self.height])
            x_move = np.array([-0.5, 0, 0.5])
            y_move = x_move
            for i in x_move:
                for j in y_move:
                    move = np.sum([self.x, [i, j]], axis=0)
                    if (np.all(np.append(move >= w_min, move <= w_max))):
                        fs = np.append(fs, [move], axis=0)
            # make sure the agent won't repeat its history path
            fs = [agent for agent in fs if agent.tolist() not in self.track.tolist()]
            return fs

        # computes utility of forward states
        def nodetect_observation(self, x):
            I = self.X.flatten()  # Convert matrix into a single vector useful for operating
            J = self.Y.flatten()
            d_square = np.sqrt(np.square(I - x[0]) + np.square(J - x[1]))
            expo = - sigma * ((d_square / dmax) * (d_square / dmax))
            pnd = 1 - Pdmax * np.exp(expo)
            return pnd

        # computes utility of states
        def utility(self, fs):
            # compute cost funtion J of all potential forward states (brute force method)
            J = np.zeros(0)
            for x in fs:
                pnd = self.nodetect_observation(x)
                prod = np.multiply(pnd, self.bk)
                J = np.append(J, np.sum(prod))
            return J

        # find the next best state by utility values
        def next_best_state(self, fs):
            J = self.utility(fs)
            return fs[np.argmin(J)]

        # simulate agent next state
        def next(self, state):
            self.x = state
            self.track = np.vstack((self.track, self.x))

        # update belief with observation at state self.x
        def update_belief(self):
            # update belief with the new observation (non-detection)
            pnd = self.nodetect_observation(self.x)
            belief = np.multiply(pnd, self.bk)
            # normalize belief
            self.bk = belief / np.sum(belief)

        def set_belief(self, bk):
            self.bk = bk

        def plot(self, ax):
            # Reshape belief for plotting
            bkx = self.bk.reshape((self.map_size[1], self.map_size[0]))

            # plot contour of the P(\tau) -> self.bk
            cset = ax.contourf(self.X, self.Y, bkx, zdir='z', offset=-0.002, cmap=cm.viridis)

            # plot agent trajectory, self.track
            ax.plot(self.track[:, 0], self.track[:, 1], np.ones(self.track.shape[0]) * self.height_plot, 'r-',
                    linewidth=2);
            ax.plot([self.track[-1, 0], self.track[-1, 0]], [self.track[-1, 1], self.track[-1, 1]],
                    [self.height_plot, 0], 'ko-', linewidth=2);

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('P')

            # Adjust the limits, ticks and view angle
            ax.set_zlim(0, 0.12)
            ax.view_init(27, -21)

    class Environment:

        # Constructor
        def __init__(self):
            size = np.array([40, 40])
            mu = np.array([size[0] / 2., size[1] / 2.])
            Sigma = np.array([[40, 0], [0, 60]])
            b_0 = bivariate_gaussian(size, mu, Sigma)

        def plot(self, agents):
            for a in agents:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                a.plot(ax)

    print('-------------------------------------------------\n');
    print('> M-Agents search 2D (1 n-step ahead no comm)\n')

    nagents = 4
    # Create environment
    e = Environment()
    size = np.array([40, 40])

    mu = np.array([size[0] / 2., size[1] / 2.])
    Sigma = np.array([[40, 0], [0, 60]])
    b_0 = bivariate_gaussian(size, mu, Sigma)

    # Create agents
    a1 = Agent(i=0, state=np.array([5, 5]), map_size=size, bk=b_0)
    a1.next(np.array([5, 5]))

    a2 = Agent(i=1, state=np.array([20, 20]), map_size=size, bk=b_0)
    a2.next(np.array([20, 20]))

    a3 = Agent(i=2, state=np.array([35, 15]), map_size=size, bk=b_0)
    a3.next(np.array([35, 15]))

    a4 = Agent(i=3, state=np.array([35, 35]), map_size=size, bk=b_0)
    a4.next(np.array([35, 35]))

    agents = []
    agents.append(a1)  # add agent
    agents.append(a2)
    agents.append(a3)
    agents.append(a4)

    # Global plot for animations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax1 = fig.add_subplot(111, projection='3d')
    ax2 = fig.add_subplot(111, projection='3d')
    ax3 = fig.add_subplot(111, projection='3d')
    plt.ion()

    # Start algorithm
    ite = 0  # iteration count
    nite = 30  # number of iterations
    found = 0  # target found

    ## start search
    while not found and ite < nite:
        ax.cla()
        ax1.cla()
        ax2.cla()
        ax3.cla()# clear axis plot
        for a in agents:
            fs = a.forward()
            x_next = a.next_best_state(fs)
            a.next(x_next)
            a.update_belief()
        for a in agents:
            a.plot(ax)
            a.plot(ax1)
            a.plot(ax2)
            a.plot(ax3)

        plt.draw()
        plt.pause(0.1)
        # iteration count
        ite += 1



# Exercise 2
def exercise2():
    class Agent:

        # Constructor
        def __init__(self, i, state=np.array([0., 0.]), map_size=np.array([40, 40]), bk=np.ones(1600) * (1. / 1600)):

            self.id = i  # Create an instance variable
            self.map_size = map_size

            ## Environment. Tip it would be efficient to compute the grid into one vector as we did in the previous notebook

            self.bk = bk  # belief at instant k

            ## Agent params
            self.x = state
            self.track = self.x.T  # Stores all positions. Aux variable for visualization of the agent path
            self.height_plot = 0.1

            # add extra agents params
            self.width = map_size[0]
            self.height = map_size[1]
            self.X, self.Y = np.meshgrid(np.arange(self.width), np.arange(self.height))

            ## Sensor params

        # get id
        def get_id(self):
            return self.id

        # compute discrete forward states 1 step ahead
        def forward(self):
            fs = np.zeros((0, 2))
            w_min = np.array([0, 0])
            w_max = np.array([self.width, self.height])
            x_move = np.array([-0.5, 0, 0.5])
            y_move = x_move
            for i in x_move:
                for j in y_move:
                    move = np.sum([self.x, [i, j]], axis=0)
                    if (np.all(np.append(move >= w_min, move <= w_max))):
                        fs = np.append(fs, [move], axis=0)
            fs = [agent for agent in fs if agent.tolist() not in self.x.tolist()]
            return fs

        # computes utility of forward states
        def nodetect_observation(self, x):
            I = self.X.flatten()  # Convert matrix into a single vector useful for operating
            J = self.Y.flatten()
            d_square = np.sqrt(np.square(I - x[0]) + np.square(J - x[1]))
            expo = - sigma * ((d_square / dmax) * (d_square / dmax))
            pnd = 1 - Pdmax * np.exp(expo)
            return pnd

        # computes utility of states
        def utility(self, fs):
            # compute cost funtion J of all potential forward states (brute force method)
            J = np.zeros(0)
            for x in fs:
                pnd = self.nodetect_observation(x)
                prod = np.multiply(pnd, self.bk)
                J = np.append(J, np.sum(prod))
            return J

        # find the next best state by utility values
        def next_best_state(self, fs):
            J = self.utility(fs)
            return fs[np.argmin(J)]

        # simulate agent next state
        def next(self, state):
            self.x = state
            self.track = np.vstack((self.track, self.x))

        def set_belief(self, bk):
            self.bk = bk

        def plot(self, ax):
            # Reshape belief for plotting
            bkx = self.bk.reshape((self.map_size[1], self.map_size[0]))

            # plot contour of the P(\tau) -> self.bk
            cset = ax.contourf(self.X, self.Y, bkx, zdir='z', offset=-0.002, cmap=cm.viridis)

            # plot agent trajectory, self.track
            ax.plot(self.track[:, 0], self.track[:, 1], np.ones(self.track.shape[0]) * self.height_plot, 'r-',
                    linewidth=2);
            ax.plot([self.track[-1, 0], self.track[-1, 0]], [self.track[-1, 1], self.track[-1, 1]],
                    [self.height_plot, 0], 'ko-', linewidth=2);

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('P')

            # Adjust the limits, ticks and view angle
            ax.set_zlim(0, 0.12)
            ax.view_init(27, -21)

    class Environment:
        # Constructor
        def __init__(self):
            size = np.array([40, 40])
            mu = np.array([size[0] / 2., size[1] / 2.])
            Sigma = np.array([[40, 0], [0, 60]])
            self.bk = bivariate_gaussian(size, mu, Sigma)

        #Update common belief for all agents given the state of the agent and the current common belief
        def update_belief(self, agents):
            for a in agents:
                # update belief with the new observation (non-detection)
                pnd = a.nodetect_observation(a.x)
                belief = np.multiply(pnd, self.bk)
                # normalize belief
                self.bk = belief / np.sum(belief)
                a.set_belief(self.bk)

        def get_belief(self):
            return self.bk

    print('-------------------------------------------------\n');
    print('> M-Agents search 2D (1 n-step ahead no comm)\n')

    nagents = 2
    # Create environment
    e = Environment()
    size = np.array([40, 40])

    # Create agents
    a1 = Agent(i=1, state=np.array([25, 5]), map_size=size, bk=e.get_belief())
    a1.next(np.array([25, 5]))

    a2 = Agent(i=2, state=np.array([30, 30]), map_size=size, bk=e.get_belief())
    a2.next(np.array([30, 30]))

    agents = []
    agents.append(a1)  # add agent
    agents.append(a2)

    # Global plot for animations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax1 = fig.add_subplot(111, projection='3d')
    plt.ion()

    # Start algorithm
    ite = 0  # iteration count
    nite = 2000  # number of iterations
    found = 0  # target found

    ## start search
    while not found and ite < nite:
        ax.cla()
        ax1.cla() # clear axis plot
        for a in agents:
            fs = a.forward()
            x_next = a.next_best_state(fs)
            a.next(x_next)
        # set new belief
        e.update_belief(agents)
        for a in agents:
            a.plot(ax)
            a.plot(ax1)

        plt.draw()
        plt.pause(0.1)
        # iteration count
        ite += 1


def exercise3():
    class Environment:
        # Constructor
        def __init__(self):
            size = np.array([40, 40])
            mu = np.array([size[0] / 2., size[1] / 2.])
            Sigma = np.array([[40, 0], [0, 60]])
            self.bk = bivariate_gaussian(size, mu, Sigma)

        # update belief with observation of agents and self.bx
        def update_belief(self, agents):
            for a in agents:
                # update belief with the new observation (non-detection)
                pnd = a.nodetect_observation(a.x)
                belief = np.multiply(pnd, self.bk)
                # normalize belief
                self.bk = belief / np.sum(belief)
                a.set_belief(self.bk)

        # Return the current belief
        def get_belief(self):
            return self.bk

    class Optimizer:

        def __init__(self):
            self.method = 'trust-constr'  # Optimization method
            self.jac = "2-point"  # Automatic Jacobian finite differenciation
            self.hess = opt.SR1()  # opt.BFGS()
            self.ul = np.pi / 4  # Max turn constraint for our problem (action limits)

        def optimize(self, fun, x0, agents, N, bk):
            # write your optimization call using scipy.optimize.minimize
            n = x0.shape[0]
            # Define the bounds of the variables in our case the limits of the actions variables
            bounds = opt.Bounds(np.ones(n) * (-self.ul), np.ones(n) * self.ul)
            # minimize the cost function. Note that I added the as arguments the extra variables needed for the function.
            res = opt.minimize(fun, x0, args=(agents, N, bk), method=self.method, jac=self.jac, hess=self.hess,
                               bounds=bounds)
            #  options={'verbose': 1})
            return res

    class AgentContinuos():
        # Constructor
        def __init__(self, i, state=np.array([0., 0.]), map_size=np.array([40, 40]), bk=np.ones(1600) * (1. / 1600)):
            self.id = i  # Create an instance variable
            self.map_size = map_size

            self.bk = bk  # belief at instant k

            ## Agent params
            self.x = state
            self.track = self.x.T  # Stores all positions. Aux variable for visualization of the agent path
            self.height_plot = 0.1

            # add extra agents params
            self.width = map_size[0] #Width of the Environment
            self.height = map_size[1] #Height of the Environment
            self.X, self.Y = np.meshgrid(np.arange(self.width), np.arange(self.height))

            self.V = 2  # Velocity of the agent
            self.dt = 1  # Interval for euler integration (continuous case)
            self.max_turn_change = 0.2  # Max angle turn (action bounds)

            self.x = np.array([5, 5, 0.2])
            self.track = self.x.T  # Stores all positions

        # set next state
        def next(self, vk):
            # singular case u = 0 -> the integral changes
            if vk == 0:
                self.x[0] = self.x[0] + self.dt * self.V * np.cos(self.x[2])
                self.x[1] = self.x[1] + self.dt * self.V * np.sin(self.x[2])
                self.x[2] = self.x[2]
            else:
                desp = self.V / vk
                if np.isinf(desp) or np.isnan(desp):
                    print('forwardstates:V/u -> error');
                self.x[0] = self.x[0] + desp * (np.sin(self.x[2] + vk * self.dt) - np.sin(self.x[2]))
                self.x[1] = self.x[1] + desp * (-np.cos(self.x[2] + vk * self.dt) + np.cos(self.x[2]))
                self.x[2] = self.x[2] + vk * self.dt

            self.track = np.vstack((self.track, self.x))

        # computes utility of forward states
        def nodetect_observation(self, x):
            I = self.X.flatten()  # Convert matrix into a single vector useful for operating
            J = self.Y.flatten()
            d_square = np.sqrt(np.square(I - x[0]) + np.square(J - x[1]))
            expo = - sigma * ((d_square / dmax) * (d_square / dmax))
            pnd = 1 - Pdmax * np.exp(expo)
            return pnd

        #sets the belief of the agent
        def set_belief(self, bk):
            self.bk = bk

        #Allows for plotting of each agent and their belief states
        def plot(self, ax):
            # Reshape belief for plotting
            bkx = self.bk.reshape((self.map_size[1], self.map_size[0]))

            # plot contour of the P(\tau) -> self.bk
            cset = ax.contourf(self.X, self.Y, bkx, zdir='z', offset=-0.002, cmap=cm.viridis)

            # plot agent trajectory, self.track
            ax.plot(self.track[:, 0], self.track[:, 1], np.ones(self.track.shape[0]) * self.height_plot, 'r-',
                    linewidth=2);
            ax.plot([self.track[-1, 0], self.track[-1, 0]], [self.track[-1, 1], self.track[-1, 1]],
                    [self.height_plot, 0], 'ko-', linewidth=2);

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('P')

            # Adjust the limits, ticks and view angle
            ax.set_zlim(0, 0.12)
            ax.view_init(27, -21)

    #Compute the utility of an agent given velocity at N steps ahead
    def multi_utility(uk, agents, N, bk):
        value = 0
        copied = []
        uk = uk.reshape((N, len(agents)))
        J = np.zeros(0)
        for a in agents:
            copied.append(copy.deepcopy(a)) #Create a copy of the agent to move about the environment without changing the current agents
        for i, a in enumerate(copied):
            for j in range(0, N):
                a.next(uk[j, i])
                pnd = a.nodetect_observation(a.x)
                prod = np.multiply(pnd, bk)
                value = np.append(value, np.sum(prod))
        return np.sum(value)

    # Start

    # Global plot for animations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax1 = fig.add_subplot(111, projection='3d')
    ax2 = fig.add_subplot(111, projection='3d')
    plt.ion()

    #steps ahead we will look
    N = 4

    e = Environment()

    #Initialization
    size = np.array([40, 40])
    mu = np.array([size[0] / 2., size[1] / 2.])
    Sigma = np.array([[40, 0], [0, 60]])
    b_0 = bivariate_gaussian(size, mu, Sigma)
    # Create agents
    a1 = AgentContinuos(i=0, state=np.array([5, 5]), map_size=size, bk=b_0)
    a2 = AgentContinuos(i=1, state=np.array([20, 20]), map_size=size, bk=b_0)
    a3 = AgentContinuos(i=2, state=np.array([35, 15]), map_size=size, bk=b_0)

    agents = []
    agents.append(a1)  # add agent
    agents.append(a2)
    agents.append(a3)
    M = len(agents)

    #initialize a set of starting velocities to be optimized
    uk = np.ones(shape=(M, N))
    uk = uk * 0.000001

    # Optimizer
    optimizer = Optimizer()
    # Start algorithm
    ite = 0  # iteration count
    nite = 40  # number of iterations
    found = 0  # target found

    # Start search
    while not found and ite < nite:
        ax.cla()
        ax1.cla()  # clear axis plot
        ax2.cla()
        res = optimizer.optimize(multi_utility, uk.reshape((N * len(agents))), agents, N, e.get_belief()) #Find the most optimal move to make given N steps ahead
        start = 0
        for a in agents:
            a.next(res.x[start]) #move the agent to the next position based on the first N returned ignoring allow N's beyond 0
            start += N
        e.update_belief(agents)

        #plot the agents in the environment
        for a in agents:
            a.plot(ax)
            a.plot(ax1)
            a.plot(ax2)

        #Play animation
        plt.draw()
        plt.pause(0.1)
        # iteration count
        ite += 1

    class Environment:
        # Constructor
        def __init__(self):
            size = np.array([40, 40])
            mu = np.array([size[0] / 2., size[1] / 2.])
            Sigma = np.array([[40, 0], [0, 60]])
            self.bk = bivariate_gaussian(size, mu, Sigma)

        def update_belief(self, agents):
            for a in agents:
                # update belief with the new observation (non-detection)
                pnd = a.nodetect_observation(a.x)
                belief = np.multiply(pnd, self.bk)
                # normalize belief
                self.bk = belief / np.sum(belief)
                a.set_belief(self.bk)

        def get_belief(self):
            return self.bk

    print('-------------------------------------------------\n');
    print('> M-Agents search 2D (1 n-step ahead no comm)\n')

    nagents = 2
    # Create environment
    e = Environment()
    size = np.array([40, 40])

    # Create agents
    a1 = Agent(i=1, state=np.array([25, 5]), map_size=size, bk=e.get_belief())
    a1.next(np.array([25, 5]))

    a2 = Agent(i=2, state=np.array([30, 30]), map_size=size, bk=e.get_belief())
    a2.next(np.array([30, 30]))

    agents = []
    agents.append(a1)  # add agent
    agents.append(a2)

    # Global plot for animations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax1 = fig.add_subplot(111, projection='3d')
    plt.ion()

    # Start algorithm
    ite = 0  # iteration count
    nite = 2000  # number of iterations
    found = 0  # target found

    ## start search
    while not found and ite < nite:
        ax.cla()
        ax1.cla() # clear axis plot
        for a in agents:
            fs = a.forward()
            x_next = a.next_best_state(fs)
            a.next(x_next)
        # set new belief
        e.update_belief(agents)
        for a in agents:
            a.plot(ax)
            a.plot(ax1)

        plt.draw()
        plt.pause(0.1)
        # iteration count
        ite += 1


Pdmax = 0.8  # Max range sensor
dmax = 4  # Max distance
sigma = 0.7  # Sensor spread (standard deviation)

main()