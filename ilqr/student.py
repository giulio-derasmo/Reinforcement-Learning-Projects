import numpy as np
from scipy import linalg
from numpy import sin, cos

class CartPole:
    def __init__(self, env, x=None,max_linear_velocity=2, max_angular_velocity=np.pi/3):
        if x is None:
            x = np.zeros(2)
        self.env = env
        
    def getA(self, u, x=None):
        if x is None:
            x = self.x
        x, x_dot, theta, theta_dot = x
        polemass_length = self.env.polemass_length
        gravity = self.env.gravity
        masspole = self.env.masspole
        total_mass = self.env.total_mass
        length = self.env.length
        force = u
        dt = self.env.tau
        A = np.eye(4)

        
        # Auxiliary variable in order to make code more clear
        AUX = (force + polemass_length*theta_dot**2*sin(theta))/total_mass 
        DEN = length * (4/3 - masspole*cos(theta)**2/total_mass) 
        # theta_dot2 variable using the two aux variable above
        # here thetha_dot2 represent the second derivative of theta
        theta_dot2 = (gravity*sin(theta) - cos(theta)*AUX)/DEN
        
        # Hand-made derivative (all pieces neeed)
        # dAUX/dtheta
        dAUX_dtheta = polemass_length*theta_dot**2*cos(theta)/total_mass
        # dAUX/dtheta_dot
        dAUX_dtheta_dot = polemass_length*sin(theta)*2*theta_dot/total_mass
        # dDEN/dtheta
        dDEN_dtheta = length*masspole*2*sin(theta)*cos(theta)/total_mass
        # dtheta_dot2/d_theta
        dtheta_dot2_dtheta = ( (gravity*cos(theta)+sin(theta)*AUX-cos(theta)*dAUX_dtheta)*DEN\
                                - (gravity*sin(theta) - cos(theta)*AUX)*dDEN_dtheta)\
                              /DEN**2
        # dtheta_dot2/d_theta_dot
        dtheta_dot2_dtheta_dot = -cos(theta)*dAUX_dtheta_dot/DEN 
        # dx_dot2/dtheta
        dx_dot2_dtheta = (theta_dot**2*cos(theta)\
                          + theta_dot2*sin(theta)\
                          - cos(theta)*dtheta_dot2_dtheta)*polemass_length/total_mass
        # dx_dot2/dtheta_dot
        dx_dot2_dtheta_dot = (2*theta_dot*sin(theta)-cos(theta)*dtheta_dot2_dtheta_dot)*polemass_length/total_mass
        
        # Building the matrix with the right pieces
        A[0, :] = [1, dt, 0, 0]
        A[1, :] = [0, 1, dx_dot2_dtheta*dt, dx_dot2_dtheta_dot*dt]
        A[2, :] = [0, 0, 1, dt]
        A[3, :] = [0, 0, dtheta_dot2_dtheta*dt, 1 + dtheta_dot2_dtheta_dot*dt]
       
        
        return A
        
    def getB(self, x=None):
        if x is None:
            x = self.x
        
        x, x_dot, theta, theta_dot = x
        polemass_length = self.env.polemass_length
        gravity = self.env.gravity
        masspole = self.env.masspole
        total_mass = self.env.total_mass
        length = self.env.length
        dt = self.env.tau

        B = np.zeros((4,1))

        # Denominator of theta_dot2 (which is the second derivative of theta)
        DEN = length * (4/3 - (masspole*cos(theta)**2)/total_mass)
        
        # Hand-made derivative 
        # dtheta_dot2/du
        dtheta_dot2_du = -cos(theta)/(DEN*total_mass)
        # dx_dot2/du
        dx_dot2_du = (1-polemass_length*cos(theta)*dtheta_dot2_du) /total_mass
        
        # Build the matrix B
        B[0] = 0
        B[1] = dx_dot2_du*dt
        B[2] = 0
        B[3] = dtheta_dot2_du*dt
       
        return B



def lqr(A, B, T=100):
    K = np.zeros((1,4))
    
    # Initialize matrices
    V = np.zeros((4,4))
    Q = np.zeros((5,5))
    C = np.eye(5)
    F = np.hstack([A,B])
    
    
    for t in range(T):
        Q = C + F.T @ V @ F
        Q_xx = Q[:4,:4]
        Q_uu = Q[4:,4:]
        Q_ux = Q[4:,:4]
        Q_xu = Q[:4,4:]
        K = -np.linalg.inv(Q_uu) @ Q_ux
        V = Q_xx + Q_xu @ K + K.T @ Q_ux + K.T @ Q_uu @ K
    
    return K