from __future__ import unicode_literals
from casadi import *   
import numpy as np
import time as tm

def nmpc_cas_ipopt(NDT, T, currentState, F, cost_F, P, terminal_alpha, epsilon, lbx, ubx, lbu, ubu):
    # Start with an empty NLP
    assert len(lbu) == len(ubu), 'The size of control constraints is not equal!!!'
    assert len(lbx) == len(ubx), 'The size of state constraints is not equal!!!'
    sizeX = len(lbx)
    sizeU = len(lbu)
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    J_terminal = 0
    g=[]
    lbg = []
    ubg = []
    # "Lift" initial conditions
    X0 = MX.sym('X0', sizeX)
    w += [X0]
    lbw += currentState
    ubw += currentState
    w0 += currentState
    # Formulate the NLP
    N_integral = int(round(float(T)/NDT)) # integral amount
    Xk = MX(currentState) # initial states
    # multiple shooting
    for k in range(N_integral):
        # New NLP variable for the control
        Uk = MX.sym('U' + str(k), sizeU)
        w   += [Uk]
        lbw += lbu
        ubw += ubu
        w0  += [0.0*i for i in range(sizeU)]
        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=Uk)
        cost_Fk = cost_F(x0=Xk, p=Uk)
        Xk_end = Fk['xf']
        J=J+cost_Fk['qf']
        # New NLP variable for state at end of interval
        Xk = MX.sym('X' + str(k+1), sizeX)
        w   += [Xk]
        lbw += lbx
        ubw += ubx
        w0  += [0.0*i for i in range(sizeX)]
        # Add terminal cost
        J_terminal = mtimes([ Xk.T, P, Xk ])
        # Add equality constraint 
        g   += [Xk_end-Xk]
        lbg += [0.0*i for i in range(sizeX)]
        ubg += [0.0*i for i in range(sizeX)]
        # Add robust inequality constraints
        # g += [J_terminal]
        # lbg += [0]
        # ubg += [0.03*T*terminal_alpha/((k+1)*NDT)]
    # Create an NLP solver
    # # Add terminal constraint 
    g += [J_terminal]
    lbg += [0]
    ubg += [epsilon*terminal_alpha]
    # Add terminal cost
    J = J + J_terminal
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    opts = {'ipopt.linear_solver':'ma57', 'ipopt.tol':1e-6}
    solver = nlpsol('solver', 'ipopt', prob, opts)
    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    # print type(sol), sol.keys()
    w_opt = sol['x'].full().flatten()

    return w_opt

def intgral_etmpc_tracking(NDT, T, simulationTime, Q, Q_value, R, R_value, P, \
P_value, terminal_alpha, epsilon, lbx, ubx, lbu, ubu, beta, delta, K_feedback, \
currentState, w_r, v_r):
    DTS = int(round(float(T)/NDT))
    ####################################################
    ################# system dynamics ##################
    ####################################################
    # Declare model variables
    x_e = SX.sym('x_e')
    y_e = SX.sym('y_e')
    theta_e = SX.sym('theta_e')
    x = vertcat(x_e, y_e, theta_e)
    u1 = SX.sym('u1')
    u2 = SX.sym('u2')
    v_r = SX(v_r)
    w_r = SX(w_r)
    u = vertcat(u1, u2)
    # Disturbance
    d = SX.sym('d')
    # Nominal Non-holonomic Model -v+v_r*cos(theta_e)
    xdot = vertcat((w_r-u2)*y_e+u1, -(w_r-u2)*x_e+v_r*sin(theta_e), u2)
    # Real Non-holonomic Model
    xdot_real = vertcat((w_r-u2)*y_e+u1+d, -(w_r-u2)*x_e+v_r*sin(theta_e)+d, u2+d)
    # Objective term
    L = mtimes([ x.T, Q, x ]) + mtimes([ u.T, R, u ])
    cost_f = Function('cost_f', [x, u], [L])
    # Formulate discrete time dynamics
    # Nominal dynamics
    f = Function('f', [x, u], [xdot])
    # Real dynamics
    f_real = Function('f_real', [x, u, d], [xdot_real])
        # Fixed step Runge-Kutta 4 integrator
    M = 4 # RK4 steps per interval
    DT = float(NDT)/M
    ###################### nominal RK4 #########################
    X0 = MX.sym('X0', 3)
    U = MX.sym('U', 2)
    X = X0
    for j in range(M):
        k1 = f(X, U)
        k2 = f(X + DT/2 * k1, U)
        k3 = f(X + DT/2 * k2, U)
        k4 = f(X + DT * k3, U)
        X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
    F = Function('F', [X0, U], [X],['x0','p'],['xf'])
    ######################## real RK4 ###########################
    D= MX.sym('D') 
    X = X0
    for j in range(M):
        k1_r = f_real(X, U, D)
        k2_r = f_real(X + DT/2 * k1_r, U, D)
        k3_r = f_real(X + DT/2 * k2_r, U, D)
        k4_r = f_real(X + DT * k3_r, U, D)
        X=X+DT/6*(k1_r +2*k2_r +2*k3_r +k4_r)
    F_real = Function('F_real', [X0, U, D], [X],['x0','p', 'd'],['xf'])
    ####################### cost RK4 ############################
    Q = 0
    X = X0
    for j in range(M):
        k1 = cost_f(X, U)
        k2 = cost_f(X + DT/2 * k1, U)
        k3 = cost_f(X + DT/2 * k2, U)
        k4 = cost_f(X + DT * k3, U)
        Q = Q + DT/6*(k1 + 2*k2 + 2*k3 + k4)
    cost_F = Function('cost_F', [X0, U], [Q],['x0','p'],['qf'])
    ####################################################
    ############### end system dynamics ################
    ####################################################
    ######### ~~~~~~~~~~~start simulation~~~~~~~~~~##########  
    # initial variables
    states_x1 = []
    states_x2 = []
    states_x3 = []
    u1_seq = []
    u2_seq = []
    u_seq = []
    stagecost_seq = []
    event_triggered_instants = []
    event_sampling_intervals = []
    event_sampling_instants = [0]
    states_x1.append(currentState[0])
    states_x2.append(currentState[1])
    states_x3.append(currentState[2])
    # add the initial stage cost without u
    stagecost_seq.append(np.dot(np.dot(np.array(currentState).T,Q_value),np.array(currentState)))
    # main loop of event-triggered MPC
    clk = 0
    t = 0
    # sign that transform to feedback control
    clk1 = 0
    file = open("record.txt", 'a')
    file.write(tm.ctime()+'\n')
    file.close()
    while (t < simulationTime):
        w_opt = nmpc_cas_ipopt(NDT, T, currentState, F, cost_F, P, terminal_alpha, epsilon, lbx, ubx, lbu, ubu)
        # Untie the solution block
        x_opt = np.array([ w_opt[0::5], w_opt[1::5], w_opt[2::5] ])
        u_opt = np.array([ w_opt[3::5], w_opt[4::5] ])
        # file = open("optimal_u.txt", 'a')
        # file.write(str(len(optimal_u_seq))+str(optimal_u_seq[100])+str(optimal_u_seq) + "\n")
        # file.close()
        # discrete update steps 
        # Xk = np.array(currentState)+0.9*beta*np.random.rand()
        Xk = np.array(currentState)

        clk_innerloop = 0
        # initializes every detection
        checkingPeriod = 0
        samplingSign = 0
        # initializes integral-type event-triggering condition
        int_err = 0
        temp_err = np.zeros(2)

        while (samplingSign == 0):
            # hat_Xk = np.array([optimal_x1[clk_innerloop],optimal_x2[clk_innerloop]])
            hat_Xk = x_opt.T[clk_innerloop]
            # print hat_Xk, type(hat_Xk), np.size(hat_Xk)
            # error with respect to event-triggered condition
            err = Xk-hat_Xk
            # print (hat_Xk.T.dot(P_value)*hat_Xk.T).sum(axis=0), type(hat_Xk.T.dot(P_value)*hat_Xk.T)
            # # dual mode
            # temp_quad = np.sqrt(np.dot(np.dot(Xk.T,P_value),Xk))
            # if (np.sqrt(np.dot(np.dot(Xk.T,P_value),Xk)) < terminal_alpha*0.03) or (clk1 <> 0):
            #     clk1 += 1
            #     # optimal_u = np.dot(K_feedback, hat_Xk)
            #     optimal_u = np.dot(K_feedback, Xk)
            # else:
            #     optimal_u = u_opt.T[clk_innerloop]
            # no dual mode
            optimal_u = u_opt.T[clk_innerloop]
            # optimal_u = optimal_u_seq[0]
            # real system dynamics
            Fk_r = F_real(x0=Xk,p=optimal_u,d=beta*(np.random.rand()))
            # Fk_r = F(x0=Xk,p=optimal_u)
            Xk = Fk_r['xf'].full().flatten()
            # store the envolving states and optimal inputs, and stage cost
            states_x1.append(Xk[0])
            states_x2.append(Xk[1])
            states_x3.append(Xk[2]) 
            u_seq.append(optimal_u[0])
            # stagecost_seq.append(np.dot(np.dot(Xk.T,Q_value),Xk)+R_value*optimal_u**2)
            stagecost_seq.append(np.dot(np.dot(Xk.T,Q_value),Xk))           
            # print clk_innerloop, type(optimal_x1[clk_innerloop]), optimal_x1[clk_innerloop], err, err.shape, type(err)
            # print np.dot(np.dot(err.T,P_value),err)
            # before sampling, check the sampling Sign. If triggered then break the while loop
            # periodic detecting with period = 10*NDT
            # record the time steps passed since last sampling
            clk_innerloop += 1
            clk += 1
            ####################################################################
            ### integral-type event-triggering condition
            # int_err = int_err + np.sqrt(np.dot(np.dot(err.T,P_value),err))*NDT
            # int_temp_check = int_err/(clk_innerloop*NDT)
            ### non-integral-type event-triggering condition
            int_temp_check = np.sqrt(np.dot(np.dot(err.T,P_value),err))
            #####################################################################
            if int_temp_check > delta or clk_innerloop >= DTS:
                samplingSign = 1
                print np.sqrt(np.dot(np.dot(err.T,P_value),err)), int_err, delta
                file = open("record.txt", 'a')
                file.write(str(int_err) + "   " + str(delta) + "\n")
                file.close()
                # if triggering condtion is satisfied, then go to the next sampling 
            else:
                # otherwise, starting another detection
                checkingPeriod = 0
        # sampling the state for the MPC controller
        currentState = Xk.tolist()
        # record every sampling interval
        t += clk_innerloop*NDT
        event_triggered_instants += [t]
        event_sampling_intervals += [clk_innerloop*NDT] 
        event_sampling_instants += [int(t/NDT)-1]

    # cut the useful information from initial time to the simulationTime    
    event_sampling_intervals = event_sampling_intervals[0:-1]
    event_sampling_instants = event_sampling_instants[0:-1]
    simulationSteps = int(round(float(simulationTime)/NDT))
    states_x1 = states_x1[0:simulationSteps] 
    states_x2 = states_x2[0:simulationSteps]
    states_x3 = states_x3[0:simulationSteps]
    
    u_seq = u_seq[0:simulationSteps-1]
    stagecost_seq = stagecost_seq[0:simulationSteps]

    return (states_x1, states_x2, states_x3, u_seq, stagecost_seq, event_triggered_instants, event_sampling_intervals, event_sampling_instants)

def main():
    starttime = tm.time()
    # np.random.seed(0)
    # approximate time scale
    # approximate time scale
    NDT = 0.1
    T = 4.0 # Time horizon
    DTS = int(round(float(T)/NDT))
    # samplingInterval = 0.1 #10Hz
    simulationTime = 20.0 # s
    # initial error state construction
    initialState_real = [4, -3.2, 0] # initial x, y, theta
    theta = initialState_real[2]
    initialState_ref = [1, -0.2, 3.14/2] # initial x_r, y_r, theta_r
    circle_matrix = np.array([ [np.cos(theta), np.sin(theta), 0.0], 
        [-np.sin(theta), np.cos(theta), 0.0], 
        [0.0, 0.0, 1] ])
    initialState = np.dot(circle_matrix,\
    (np.array(initialState_ref)-np.array(initialState_real))) # initial x_e, y_e, theta_e
    initialState = initialState.tolist()
    ## circle reference
    # reference w and v
    w_r = 0.4
    v_r = 1.5
    # reference x and y
    simulation_instants = np.array([NDT*i for i in range(200)])+NDT
    theta_r = w_r*simulation_instants
    x_r = v_r*np.cos(theta_r)
    y_r = v_r*np.sin(theta_r)
    
    # Q matrix and R matrix
    Q_value = np.array([ [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5] ])
    Q = MX(DM(Q_value))
    R_value = np.array([ [0.2, 0], [0, 0.2] ])
    R = MX(DM(R_value))
    # terminal cost matrix
    # P_value = np.array([ [0.0214, -0.0064, -0.0104], [-0.0064, 0.0679, 0.0766], [-0.0104, 0.0766, 0.1153] ])
    # P_value = np.array([ [0.0880,-0.0131,-0.0202], [ -0.0131,0.2059,0.1716], [-0.0202,0.1716,0.2841] ])
    P_value = np.array([ [0.9000,-1.4000,0], [-1.4000,5.7000,0], [0,0,0.4750] ])
    P = MX(DM(P_value))
    # terminal constraints
    terminal_alpha = 0.8
    alpha = 0.8
    epsilon = 0.03
    # state constraints & input constraints
    lbx = [-2, -2, -2]
    ubx = [3, 3, 3]
    lbu = [-2, -2]
    ubu = [2, 2]
    # disturbance
    beta = 0.00
    # beta = 0.1
    # beta = [0.1,0.1]
    # event-triggered conditions
    delta = 0.000207
    # K_feedback_original = [ [1.0800, -0.4057, -0.7129], [-0.2446, 2.9808, 3.9200] ]
    # K_feedback = [ [0.5*y for y in x] for x in K_feedback_original]
    K_feedback = [ [2.0000,-2.0000,0], [0,0,1.5000] ]

    # start the integral-type ETMPC...
    for i in range(1):
        (states_x1, states_x2, states_x3, u_seq, stagecost_seq, \
        event_triggered_instants, event_sampling_intervals, \
        event_sampling_instants) = intgral_etmpc_tracking(NDT, T, simulationTime, \
        Q, Q_value, R, R_value, P, P_value, terminal_alpha, epsilon, lbx, ubx, lbu, \
        ubu, beta, delta, K_feedback, initialState, w_r, v_r)

        # restore the real states from the error states
        positions_e = np.array([ states_x1, states_x2, states_x3 ])
        angles_e = states_x3

        states_x1_real = []
        states_x2_real = []
        states_x3_real = []

        for i in range(len(angles_e)):
            temp_ref = np.array([ x_r[i], y_r[i], theta_r[i]])
            temp_state = temp_ref-np.dot(np.linalg.inv(circle_matrix), positions_e.T[i])
            states_x1_real.append(temp_state[0])
            states_x2_real.append(temp_state[1])
            states_x3_real.append(temp_state[2])

        # file = open("INTETMPC_intervals.txt", 'a')
        # file.write('INTETMPC-TEST #' + str(i+1) + ' ---> ' + 'SampledTimes: ' + str(len(event_sampling_intervals)) + ' EventIntervals: ' + str(event_sampling_intervals) + "\n")
        # file.close()

        # file = open("ETMPC_intervals.txt", 'a')
        # file.write('ETMPC-TEST #' + str(i+1) + ' ---> ' + 'SampledTimes: ' + str(len(event_sampling_intervals)) + ' EventIntervals: ' + str(event_sampling_intervals) + "\n")
        # file.close()
        

    # record the finish time
    endtime = tm.time()
    print 'Running time: ', (endtime - starttime)*1000 , 'ms'
    
    # for testing or debugging
    print 'Length of input, states: ', len(u_seq), len(states_x1)
    print 'SizeX: ', len(states_x1), 'SizeU: ', len(u_seq), 'StagecostSize: ', len(stagecost_seq)
    print 'Event sampling instants: ', event_sampling_instants, len(event_sampling_instants)

    print 'Event sampling intervals: ', event_sampling_intervals

    # # save to files 
    # intetcfile = open('intetcfile.npz', 'w+')
    # np.savez('intetcfile.npz', x1=np.array(states_x1), x2=np.array(states_x2), u=np.array(u_seq))
    
    # # start plot everything...  
    # ######### ~~~~~~~~~~~start plotting~~~~~~~~~~##########
    # paintingstep = int(round(float(simulationTime)/NDT))
    # tgrid = [k for k in range(paintingstep)]
    # fig1 = plt.figure(1)
    # ax1 = fig1.add_subplot(111)
    # ax1.plot(tgrid, states_x1, '-.', linewidth=3)
    # ax1.plot(tgrid, states_x2, '--', linewidth=3)
    # ax1.plot(tgrid, states_x3, '--', linewidth=3)
    # ax1.set_xlabel(r'$t(s)$')
    # ax1.set_xlim([0,paintingstep])
    # xt1 = np.rint(ax1.get_xticks()*NDT)
    # ax1.set_xticklabels(xt1.astype(int))
    # ax1.legend([r'$x_1$',r'$x_2$',r'$\theta$'])
    # ax1.grid(color='k', linestyle=':')

    # if os.name == 'posix':  
    #     fig1.savefig('system_trajectory.pdf')
    # elif os.name == 'nt':
    #     fig1.show()
    #     pass

    # fig2 = plt.figure(2)
    # ax2 = fig2.add_subplot(111)
    # ax2.plot(tgrid, vertcat(DM.nan(1), u_seq), '-', lw=0.8)
    # # plt.step(tgrid, vertcat(DM.nan(1), u_seq), '-', lw=1.0)
    # u_full = np.array(u_seq)
    # ax2.plot(np.array(event_sampling_instants), u_full[event_sampling_instants], 'x', markersize=8, markeredgewidth=1.2, markerfacecolor='none', markeredgecolor='r')
    # # plt.plot(tgrid, vertcat(DM.nan(1), u_seq), '-')
    # ax2.set_xlabel(r'$t(s)$', fontsize=14)
    # paintingstep = int(round(float(simulationTime)/NDT))
    # ax2.set_xlim([0,paintingstep])
    # ax2.set_ylim([-0.2,1.2])
    # xt2 = np.rint(ax2.get_xticks()*NDT)
    # ax2.set_xticklabels(xt2.astype(int))
    # ax2.legend([r'$u$', r'$Triggered\,\,instants$'], fontsize=14)
    # ax2.grid(color='k', linestyle=':')
    
    # if os.name == 'posix':  
    #     fig2.savefig('control_inputANDevent_triggering_instants.pdf')
    # elif os.name == 'nt':
    #     fig2.show()
    #     pass
    
    # fig3 = plt.figure(3)
    # ax3 = fig3.add_subplot(111)
    # ax3.plot(tgrid, stagecost_seq, '-', linewidth=3)
    # ax3.plot(np.array(event_sampling_instants), np.array(stagecost_seq)[event_sampling_instants], 's', markersize=6, markeredgewidth=0.5, markerfacecolor='k', markeredgecolor='k')
    # ax3.set_xlabel(r'$t(s)$', fontsize=16)
    # ax3.set_ylabel(r'$J$', fontsize=16)
    # paintingstep = int(round(float(simulationTime)/NDT))
    # ax3.set_xlim([0,paintingstep])
    # xt3 = np.rint(ax3.get_xticks()*NDT)
    # ax3.set_xticklabels(xt3.astype(int))
    # ax3.legend(['Stage Cost'])
    # ax3.grid(color='k', linestyle=':')
    
    # if os.name == 'posix':  
    #     fig3.savefig('cost_trajectory.pdf')
    # elif os.name == 'nt':
    #     fig3.show()
    #     pass
     
    # fig4 = plt.figure(4)
    # ax4 = fig4.add_subplot(111, aspect='equal')
    # ax4.plot(states_x1_real, states_x2_real, '-', linewidth=3)
    # ax4.plot(np.array(states_x1_real)[event_sampling_instants], np.array(states_x2_real)[event_sampling_instants], 's', markersize=6, markeredgewidth=0.5, markerfacecolor='k', markeredgecolor='k')
    # # t4 = np.arange(-4,4,0.1)
    # # x4 = y4 = np.arange(-0.5, 0.5, 0.01)
    # # x4, y4 = np.meshgrid(x4,y4)
    # # ax4.contour(x4, y4, x4**2*P_value[0,0]+y4**2*P_value[1,1]+2*x4*y4*P_value[0,1]-epsilon, [0], colors='magenta', linestyles='dotted')
    # # ax4.contour(x4, y4, x4**2*P_value[0,0]+y4**2*P_value[1,1]+2*x4*y4*P_value[0,1]-terminal_alpha*epsilon, [0], colors='green', linestyles='dotted')
    # ax4.set_xlabel(r'$x_1$', fontsize=16)
    # ax4.set_ylabel(r'$x_2$', fontsize=16)
    # ax4.legend([r'State'])
    # # ax4.annotate(r'$\Omega(\epsilon)$',
    # #      xy=(0.22, 0.25), xycoords='data',
    # #      xytext=(50, 20), textcoords='offset points', fontsize=16,
    # #      arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    # # ax4.annotate(r'$\Omega(\alpha\epsilon)$',
    # #      xy=(0.075, -0.12), xycoords='data',
    # #      xytext=(50, -20), textcoords='offset points', fontsize=16,
    # #      arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"))
    # ax4.grid(color='k', linestyle=':')
    
    # if os.name == 'posix':  
    #     fig4.savefig('2d_trajectory.pdf')
    # elif os.name == 'nt':
    #     fig4.show()
    #     pass

    # plt.show()


if __name__ == '__main__':
    main()
