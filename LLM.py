import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd
import os
from itertools import cycle 

os.chdir(os.path.dirname(__file__))

#---------------------Blade Discretisation---------------------
def BladeSegment(root_pos_R, tip_pos_R, pitch, nodes, seg_type='lin'):
    if seg_type=='lin':
        r_R = np.linspace(root_pos_R, tip_pos_R, nodes)

    if seg_type=='cos':
        theta = np.linspace(0,np.pi, nodes)
        r_R = (1/2)*(tip_pos_R-root_pos_R)*(1+np.cos(theta)) + root_pos_R
        r_R = np.flip(r_R)
        
    chord_dist = 0.18 - 0.06*(r_R)                  # normalised chord distribution
    twist_dist = -50*(r_R) + 35 + pitch             # twist distribution [deg]

    return r_R, chord_dist, twist_dist

#---------------------Blade Element Method---------------------
def PrandtlCorrections(Nb, r, R, TSR, a, root_pos_R, tip_pos_R):
    F_tip = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((tip_pos_R-(r/R))/(r/R))*(np.sqrt(1 + ((TSR*(r/R))**2)/((1-a)**2))))))
    F_root = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((r/R)-(root_pos_R))/(r/R))*(np.sqrt(1 + ((TSR*(r/R))**2)/((1-a)**2)))))  
    F_tot = F_tip*F_root
    
    if(F_tot == 0) or (F_tot is np.nan) or (F_tot is np.inf):
        # handle exceptional cases for 0/NaN/inf value of F_tot
        # print("F total is 0/NaN/inf.")
        F_tot = 0.00001

    return F_tot, F_tip, F_root

def BladeElementMethod(Vinf, TSR, n, rho, b, r, root_pos_R, tip_pos_R, dr, Omega, Nb, a, a_tan, twist, chord, polar_alfa, polar_cl, polar_cd, tol, P_up):
    flag = 0
    while True and (flag<1000):
            V_ax = Vinf*(1+a)       # axial velocity at the propeller blade
            V_tan = Omega*r*(1-a_tan)   # tangential veloity at the propeller blade
            V_loc = np.sqrt(V_ax**2 + V_tan**2)

            phi = np.arctan(V_ax/V_tan)     # inflow angle [rad]
            alfa = twist - np.rad2deg(phi)  # local angle of attack [deg]
            
            Cl = np.interp(alfa, polar_alfa, polar_cl)
            Cd = np.interp(alfa, polar_alfa, polar_cd)
            
            C_ax = Cl*np.cos(phi) - Cd*np.sin(phi)      # axial force coefficient
            F_ax = (0.5*rho*V_loc**2)*C_ax*chord        # axial force [N/m]

            C_tan = Cl*np.sin(phi) + Cd*np.cos(phi)     # tangential force coefficient
            F_tan = (0.5*rho*V_loc**2)*C_tan*chord      # tangential force [N/m]
           
            dCT = (F_ax*Nb*dr)/(rho*(n**2)*(2*b)**4)        # blade element thrust coefficient                   
            dCQ = (F_tan*Nb*r*dr)/(rho*(n**2)*(2*b)**5)     # blade element torque coefficient
            dCP = (F_ax*Nb*dr*Vinf)/(rho*(n**3)*(2*b)**5)   # blade element power coefficient
            
            a_new = ((1/2)*(-1+np.sqrt(1+(F_ax * Nb / (rho * Vinf**2 * np.pi * r)))))
            a_tan_new = F_tan * Nb / (2*rho*(2*np.pi*r)*Vinf*(1+a_new)*Omega*r)
            
            a_b4_Pr = a_new
            
            F_tot, F_tip, F_root = PrandtlCorrections(Nb, r, b, TSR, a, root_pos_R, tip_pos_R)
            a_new = a_new/F_tot
            a_tan_new=a_tan_new/F_tot
        
            if(np.abs(a-a_new)<tol) and (np.abs(a_tan-a_tan_new)<tol):
                a = a_new
                a_tan = a_tan_new
                flag += 1
                break
            else:
                # introduces relaxation to induction factors a and a' for easier convergence
                a = 0.75*a + 0.25*a_new
                a_tan = 0.75*a_tan + 0.25*a_tan_new
                flag += 1
                continue
    P0_down = P_up + F_ax*dr/(2*np.pi*r)
    Gamma = 0.5*V_ax*Cl*chord
    return a_b4_Pr, a, a_tan, Cl, Cd, F_ax, F_tan, alfa, phi, F_tot, F_tip, F_root, dCT, dCQ, dCP, P0_down, Gamma

#----------------------Lifting Line Model----------------------
def ControlPoint(r_R, b, blade_seg, chord_dist, twist_dist):
    mlt = 0.5       # length normalised distance of control point from origin of blade segment

    CtrlPts = []
    for j in range(blade_seg):
        y = ((r_R[j]*b)+(r_R[j+1]-r_R[j])*mlt*b)
        x = 0.25*(np.interp(y/b, r_R, chord_dist)*b)*np.cos(np.interp(y/b, r_R, twist_dist))
        z = 0.25*(np.interp(y/b, r_R, chord_dist)*b)*np.sin(np.interp(y/b, r_R, twist_dist))

        CtrlPts.append({'CP'+str(j+1): [x, y, z]})

    return CtrlPts

def CoordinateRotation(HS_vortex, blade_seg, Nb):
    HS_base = HS_vortex.copy()

    for i in range(1, Nb):
        theta = ((2*np.pi)/Nb) * i    # angular position of blade element
        # theta = np.deg2rad(np.range(60,300,))
        
        for j in range(blade_seg):
            HS_temp={}
            for key, VF in HS_base[j].items():
                x = [VF['pos1'][0], VF['pos2'][0]]
                y = [VF['pos1'][1], VF['pos2'][1]]
                z = [VF['pos1'][2], VF['pos2'][2]]

                R = np.array([
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta),  np.cos(theta)]
                ])                      # transformation matrix

                pos = np.array([x, y, z])                
                pos = np.dot(R, pos)

                HS_temp[key] = {'pos1': [pos[0][0], pos[1][0], pos[2][0]], 'pos2': [pos[0][1], pos[1][1], pos[2][1]], 'Gamma': VF['Gamma']}

            HS_vortex.append(HS_temp)
    
    
    return HS_vortex

def HorseshoeVortex(l, U_wake, vor_fil, blade_seg, Omega, r_R, Gamma, Nb, chord_dist, twist_dist):
    T = l/U_wake        # total time for wake propagation [s]
    dt = T/vor_fil      # time for propagation of each vortex filament [s]    
    rot = False         # vortex filament rotation flag
    
    HS_vortex = []
    for j in range(blade_seg):
        TR_left = {}
        BVor = {}
        TR_right = {}
        HS_temp = {}
        for i in range(vor_fil):
            # bound vortex coordinates
            if i == 0:
                # x = [(0.25*chord_dist[j]*np.cos(twist_dist[j])), (0.25*chord_dist[j+1]*np.cos(twist_dist[j+1]))]
                x = [-(0.25*chord_dist[j]*np.cos(twist_dist[j])), -(0.25*chord_dist[j+1]*np.cos(twist_dist[j+1]))]
                y = [r_R[j]*b, r_R[j+1]*b]
                z = [-(0.25*chord_dist[j]*np.sin(twist_dist[j])), -(0.25*chord_dist[j+1]*np.sin(twist_dist[j+1]))]
                BVor['VF'+str(vor_fil+1)]={'pos1': [x[0], y[0], z[0]], 'pos2':[x[1], y[1], z[1]], 'Gamma': Gamma[j]}

            # first set of trailing vortices
            if rot == False:
                # x = [[U_wake*dt, (0.25*chord_dist[j]*np.cos(twist_dist[j]))],[(0.25*chord_dist[j+1]*np.cos(twist_dist[j+1])), U_wake*dt]]
                x = [[(0.75*chord_dist[j]*np.cos(twist_dist[j])), -(0.25*chord_dist[j]*np.cos(twist_dist[j]))],
                     [-(0.25*chord_dist[j+1]*np.cos(twist_dist[j+1])), (0.75*chord_dist[j+1]*np.cos(twist_dist[j+1]))]]
                y = [r_R[j]*b, r_R[j+1]*b]
                # z = [[(0.25*chord_dist[j]*np.sin(twist_dist[j])), (0.25*chord_dist[j]*np.sin(twist_dist[j]))], 
                #      [(0.25*chord_dist[j+1]*np.sin(twist_dist[j+1])), (0.25*chord_dist[j+1]*np.sin(twist_dist[j+1]))]]
                z = [[(0.75*chord_dist[j]*np.sin(twist_dist[j])), -(0.25*chord_dist[j]*np.sin(twist_dist[j]))],
                     [-(0.25*chord_dist[j+1]*np.sin(twist_dist[j+1])), (0.75*chord_dist[j+1]*np.sin(twist_dist[j+1]))]]
                TR_left['VF'+str(vor_fil)] = {'pos1': [x[0][0], y[0], z[0][0]], 'pos2':[x[0][1], y[0], z[0][1]], 'Gamma': Gamma[j]}
                TR_right['VF'+str(vor_fil+2)] = {'pos1': [x[1][0], y[1], z[1][0]], 'pos2':[x[1][1], y[1], z[1][1]], 'Gamma': Gamma[j]}
                rot = True
            
            # subsequent set of vortex filaments
            elif rot == True:
                # left side of the trailing vortex
                x = [U_wake*dt*(i+1), U_wake*dt*i]
                y = [r_R[j]*b*np.cos(Omega*dt*(i+1)), r_R[j]*b*np.cos(Omega*dt*i)]
                z = [(-1)*r_R[j]*b*np.sin(Omega*dt*(i+1)), (-1)*r_R[j]*b*np.sin(Omega*dt*i)]

                TR_left['VF'+str(vor_fil-i)] = {'pos1': [x[0], y[0], z[0]], 'pos2':TR_left['VF'+str((vor_fil+1)-i)]['pos1'], 'Gamma': Gamma[j]}

                # right side of the trailing vortex
                x = [U_wake*dt*(i+1), U_wake*dt*i]
                y = [r_R[j+1]*b*np.cos(Omega*dt*(i+1)), r_R[j+1]*b*np.cos(Omega*dt*i)]
                z = [(-1)*r_R[j+1]*b*np.sin(Omega*dt*(i+1)), (-1)*r_R[j+1]*b*np.sin(Omega*dt*i)]

                TR_right['VF'+str((vor_fil+2)+i)] = {'pos1': TR_right['VF'+str((vor_fil+1)+i)]['pos2'], 'pos2':[x[0], y[0], z[0]], 'Gamma': Gamma[j]}
            
        TR_left = dict(reversed(list(TR_left.items())))
        HS_temp = TR_left | BVor | TR_right
        HS_vortex.append(HS_temp)   
        x = y = z = 0
        rot = False
    
    HS_vortex = CoordinateRotation(HS_vortex, blade_seg, Nb)

    return HS_vortex

'''
def CorrectOverlapVelocity(u_infl, v_infl, w_infl):
    u_infl_temp = np.copy(u_infl)
    v_infl_temp = np.copy(v_infl)
    w_infl_temp = np.copy(w_infl)
    for i in range(1, len(u_infl)-1):
        u_infl[:,i] = (u_infl_temp[:,i] + u_infl_temp[:,i-1])
        v_infl[:,i] = (v_infl_temp[:,i] + v_infl_temp[:,i-1])
        w_infl[:,i] = (w_infl_temp[:,i] + w_infl_temp[:,i-1])

    return u_infl, v_infl, w_infl
'''

def InducedVelocities(CtrlPts, pos1, pos2, gamma, tol=1e-4):
    """
    Function to calculate [u,v,w] induced by a vortex filament defined by [pos1, pos2] on a control point defined by CtrlPts.
    Input Arguments:-
        CtrlPts: [xp, yp, zp]; 1D array of control point coordinates
        pos1: [x1, y1, z1]; 1D array of the start position of the vortex filament
        pos1: [x2, y2, z2]; 1D array of the end position of the vortex filament
        gamma: int or float; magnitude of circulation around the filament
    """
    r1 = np.sqrt((CtrlPts[0]-pos1[0])**2 + (CtrlPts[1]-pos1[1])**2 + (CtrlPts[2]-pos1[2])**2)
    r2 = np.sqrt((CtrlPts[0]-pos2[0])**2 + (CtrlPts[1]-pos2[1])**2 + (CtrlPts[2]-pos2[2])**2)

    r12x = (CtrlPts[1]-pos1[1])*(CtrlPts[2]-pos2[2]) - (CtrlPts[2]-pos1[2])*(CtrlPts[1]-pos2[1])
    r12y = -(CtrlPts[0]-pos1[0])*(CtrlPts[2]-pos2[2]) + (CtrlPts[2]-pos1[2])*(CtrlPts[0]-pos2[0])
    r12z = (CtrlPts[0]-pos1[0])*(CtrlPts[1]-pos2[1]) - (CtrlPts[1]-pos1[1])*(CtrlPts[0]-pos2[0])

    r12sq = (r12x**2) + (r12y**2) + (r12z**2)
    if r12sq < tol:
        r12sq = tol
    
    r01 = (pos2[0]-pos1[0])*(CtrlPts[0]-pos1[0]) + (pos2[1]-pos1[1])*(CtrlPts[1]-pos1[1]) + (pos2[2]-pos1[2])*(CtrlPts[2]-pos1[2])
    r02 = (pos2[0]-pos1[0])*(CtrlPts[0]-pos2[0]) + (pos2[1]-pos1[1])*(CtrlPts[1]-pos2[1]) + (pos2[2]-pos1[2])*(CtrlPts[2]-pos2[2])
    
    K = (gamma/(4*np.pi*r12sq))*((r01/r1) - (r02/r2))
    U = K*r12x
    V = K*r12y
    W = K*r12z

    return U, V, W

def InfluenceCoeff(HS_vortex, CtrlPts, vor_fil, Nb):
    N_cp = len(CtrlPts)      #number of control points
    N_hs = len(HS_vortex)    #number of horseshoe vortices

    #Initialising the matrices 
    u_infl = np.zeros((N_cp,N_cp))
    v_infl = np.zeros((N_cp,N_cp))
    w_infl = np.zeros((N_cp,N_cp))

    for i in range(N_cp):   # iterate for i-th collocation point on the base blade
        for j in range(N_cp):   # iterate for effect of j-th HS vortex on i-th collocation point
            u_ind_t = 0
            v_ind_t = 0
            w_ind_t = 0
            
            for k in range(j, N_hs, N_cp):     # iterate for effect of j-th HS vortex existing over the k-th blade
                fil_neg1 = {'pos1': [], 'pos2': [], 'Gamma': 0}
                fil1 = {'pos1': [], 'pos2': [], 'Gamma': 0}
                if j != 0:
                    hs_neg1 = HS_vortex[k-1]
                hs = HS_vortex[k]
                if j != (N_cp-1):
                    hs1 = HS_vortex[k+1]
                vf_num = 0
                for vf_key in hs:
                    if vf_key != ('VF'+str(vor_fil+1)):
                        if j != 0:
                            fil_neg1 = hs_neg1['VF'+str((2*vor_fil)+1-vf_num)]

                        fil = hs[vf_key]

                        if j != (N_cp-1):
                            fil1 = hs1['VF'+str((2*vor_fil)+1-vf_num)]

                        u_fil, v_fil, w_fil = InducedVelocities(CtrlPts[i]['CP'+str(i+1)], fil['pos1'], fil['pos2'], fil['Gamma'])
                        
                        if fil['pos1'] == fil_neg1['pos2'] and fil['pos2'] == fil_neg1['pos1']:
                            # print("hello from behind.")
                            u_fil += InducedVelocities(CtrlPts[i]['CP'+str(i+1)], fil_neg1['pos1'], fil_neg1['pos2'], fil_neg1['Gamma'])[0]
                            v_fil += InducedVelocities(CtrlPts[i]['CP'+str(i+1)], fil_neg1['pos1'], fil_neg1['pos2'], fil_neg1['Gamma'])[1]
                            w_fil += InducedVelocities(CtrlPts[i]['CP'+str(i+1)], fil_neg1['pos1'], fil_neg1['pos2'], fil_neg1['Gamma'])[2]
                        # else:
                        #     print("No overlap trailing vortex filament detected on the left side HS vortex.")
                        
                        if fil['pos1'] == fil1['pos2'] and fil['pos2'] == fil1['pos1']:
                            # print("hello from ahead.")
                            u_fil += InducedVelocities(CtrlPts[i]['CP'+str(i+1)], fil1['pos1'], fil1['pos2'], fil1['Gamma'])[0]
                            v_fil += InducedVelocities(CtrlPts[i]['CP'+str(i+1)], fil1['pos1'], fil1['pos2'], fil1['Gamma'])[1]
                            w_fil += InducedVelocities(CtrlPts[i]['CP'+str(i+1)], fil1['pos1'], fil1['pos2'], fil1['Gamma'])[2]
                        # else:
                        #     print("No overlap trailing vortex filament detected on the right side HS vortex.")
                        
                        u_ind_t += u_fil
                        v_ind_t += v_fil
                        w_ind_t += w_fil

                        vf_num += 1

                    else:    
                        vf_num += 1
                
            u_infl[i][j] = u_ind_t
            v_infl[i][j] = v_ind_t
            w_infl[i][j] = w_ind_t

    # u_infl, v_infl, w_infl = CorrectOverlapVelocity(u_infl, v_infl, w_infl)

    for i in range(N_cp):   # iterate for each collocation point
        for j in range(N_cp):   # iterate for each HS vortex on that collocation point
            u_ind_t = 0
            v_ind_t = 0
            w_ind_t = 0
            
            for k in range(j, N_hs, N_cp):     # iterate over HS vortex of each blade
                hs = HS_vortex[k]
                for vf_key in hs:
                    if vf_key == ('VF'+str(vor_fil+1)):
                        fil = hs[vf_key]
                        u_fil, v_fil, w_fil = InducedVelocities(CtrlPts[i]['CP'+str(i+1)], fil['pos1'], fil['pos2'], fil['Gamma'])
                        u_ind_t += u_fil
                        v_ind_t += v_fil
                        w_ind_t += w_fil
     
            u_infl[i][j] += u_ind_t
            v_infl[i][j] += v_ind_t
            w_infl[i][j] += w_ind_t

    return u_infl, v_infl, w_infl

def LiftingLineModel(HS_vortex, CtrlPts, polar_alfa, polar_cl, polar_cd, Vinf, Omega, rho, b, r_R, chord_dist, twist_dist, Nb, l, U_wake, vor_fil, gamma):
    N_cp = len(CtrlPts)         # number of control points
    
    gamma_new, alfa, phi, a_ax_loc, a_tan_loc, F_ax_l, F_tan_l, r_cp, dCT, dCQ, dCP, seg_len = [np.zeros(N_cp) for i in range(12)]
    results = {'r':[], 'a_ax':[], 'a_tan':[], 'alfa':[], 'phi':[], 'F_ax':[], 'F_tan':[], 'CT': 0.0, 'CQ': 0.0, 'CP': 0.0, 'Gamma':[], 'iterations':0}

    u_infl, v_infl, w_infl = InfluenceCoeff(HS_vortex, CtrlPts, vor_fil, Nb)

    conv = 1e-8
    max_iter = 1000
    relax = 0.1
    iter = 0
    error = 1e8

    while error>conv and iter<=max_iter:
        for i in range(N_cp):
            xp = CtrlPts[i]['CP'+str(i+1)][0]
            yp = CtrlPts[i]['CP'+str(i+1)][1]
            zp = CtrlPts[i]['CP'+str(i+1)][2] 
            r_cp[i] = np.sqrt(yp**2 + zp**2)

            seg_len[i] = (r_R[i+1]-r_R[i])*b

            # Solving the linear system of equations
            u_ind = np.dot(u_infl[i],gamma)
            v_ind = np.dot(v_infl[i],gamma)
            w_ind = np.dot(w_infl[i],gamma)

            # print("--------------TEST-------------", np.array([u_ind, v_ind, w_ind]), iter)

            azim_vec = np.cross([1/r_cp[i],0,0],[xp, yp, zp])

            # Finding the local velocity components at the control points
            V_ax_local = Vinf + u_ind   
            V_tan_local = (Omega*r_cp[i]) + np.dot(azim_vec,[V_ax_local, v_ind, w_ind])

            V_local_mag = np.sqrt((V_ax_local**2) + (V_tan_local**2))
            phi[i] = np.arctan(V_ax_local/V_tan_local)

            # Finding the blade element properties
            r_R_local = r_cp[i]/b
            chord_local = np.interp(r_R_local, r_R, chord_dist) * b
            twist_local = np.interp(r_R_local, r_R, twist_dist)
            alfa[i] = twist_local - np.rad2deg(phi[i])

            Cl_local = np.interp(alfa[i], polar_alfa, polar_cl)
            Cd_local = np.interp(alfa[i], polar_alfa, polar_cd)

            Lift_loc = 0.5 * rho * (V_local_mag**2) * Cl_local * chord_local
            Drag_loc = 0.5 * rho * (V_local_mag**2) * Cd_local * chord_local 

            F_ax_loc = (Lift_loc * np.cos(phi[i])) - (Drag_loc * np.sin(phi[i]))
            F_tan_loc = (Lift_loc * np.sin(phi[i])) + (Drag_loc * np.cos(phi[i]))

            gamma_new[i] = Lift_loc/(rho * V_local_mag) 

            a_ax_loc[i] = (V_ax_local/Vinf) - 1
            a_tan_loc[i] = 1 - ((V_tan_local)/(Omega * r_cp[i]))

            F_ax_l[i] = F_ax_loc
            F_tan_l[i] = F_tan_loc

            dCT[i] = (F_ax_l[i]*Nb*seg_len[i])/(rho*((Omega/(2*np.pi))**2)*(2*b)**4)
            dCQ[i] = (F_tan_l[i]*Nb*r_cp[i]*seg_len[i])/(rho*((Omega/(2*np.pi))**2)*(2*b)**5)
            dCP[i] = (F_ax_l[i]*Nb*seg_len[i]*Vinf)/(rho*((Omega/(2*np.pi))**3)*(2*b)**5)
            
        a_avg = np.dot(a_ax_loc, (2*np.pi*np.multiply(r_cp,seg_len)))/(np.sum(2*np.pi*np.multiply(r_cp,seg_len)))
        # a_avg = np.sqrt(np.mean(np.square(a_ax_loc), axis=0))

        error = np.max(np.abs(gamma_new-gamma))
        if error>conv and iter<=max_iter:
            print(gamma, iter)
            gamma = (gamma_new*relax) + ((1-relax)*gamma)
            HS_vortex = HorseshoeVortex(l, (Vinf*(1+a_avg)), vor_fil, N_cp, -Omega, r_R, np.ones(N_cp), Nb, chord_dist, twist_dist)
            u_infl, v_infl, w_infl = InfluenceCoeff(HS_vortex, CtrlPts, vor_fil, Nb)
            iter += 1
        else:
            results['r'] = r_cp
            results['a_ax'] = a_ax_loc
            results['a_tan'] = a_tan_loc
            results['alfa'] = alfa
            results['phi'] = phi
            results['F_ax'] = F_ax_l
            results['F_tan'] = F_tan_l
            results['CT'] = np.sum(dCT)
            results['CQ'] = np.sum(dCQ)
            results['CP'] = np.sum(dCP)
            results['Gamma'] = gamma
            results['iterations'] = iter
            return results

#--------------------------------- MAIN ---------------------------------
# Read polar data
airfoil = 'ARAD8pct_polar.csv'
data1=pd.read_csv(airfoil, header=0, names = ["alfa", "cl", "cd", "cm"],  sep=',')
polar_alfa = data1['alfa'][:]
polar_cl = data1['cl'][:]
polar_cd = data1['cd'][:]

# Freestream Parameters
rho = 1.007                     # density at h=2000m [kg/m^3]
Pamb = 79495.22                 # static pressure at h=2000m [Pa]
Vinf = 60                       # velocity [m/s]
# J = np.array([1.6, 2.0, 2.4])   # advance ratio
J = np.array([2.0])

# Blade geometry
Nb = 2                  # number of blades
b = 0.7                 # Blade radius [m] (or blade span)
root_pos_R = 0.25       # normalised blade root position (r_root/R)
tip_pos_R = 1           # normalised blade tip position (r_tip/R)
pitch = 46              # blade pitch [deg]

# Discretisation 
blade_seg = 10      # no. of segments for the wing
vor_fil = 100       # no. of vortex filaments
l = 10*(2*b)        # length scale of the trailing vortices [m] (based on blade diameter)
seg_type = 'lin'    # discretisation type- 'lin': linear | 'cos': cosine

# Discretisation into blade elements
r_R, chord_dist, twist_dist = BladeSegment(root_pos_R, tip_pos_R, pitch, (blade_seg+1), seg_type)

# Dependent variables 
n = Vinf/(2*J*b)    # RPS [Hz]
Omega = 2*np.pi*n   # Angular velocity [rad/s]
TSR = np.pi/J       # tip speed ratio

# Iteration inputs
tol = 1e-7  # convergence tolerance

# Variable initialisation
CT, CP, CQ = [np.zeros(len(J)) for i in range(3)]
a_b4_Pr, a = [(np.ones((len(J),len(r_R)-1))*(1/3)) for i in range(2)]
chord, a_tan, Cl, Cd, F_ax, F_tan, dCT, dCQ, dCP, alfa, phi, F_tot, F_tip, F_root, P0_down, Gamma = [np.zeros((len(J),len(r_R)-1)) for i in range(16)]

P_up = np.ones((len(J),len(r_R)-1))*(Pamb + 0.5*rho*(Vinf**2))  # pressure upwind of rotor [Pa]

# Solving Blade Element Method
for j in range(len(J)):
    for i in range(len(r_R)-1):    
        chord[j][i] = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_dist) * b
        twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_dist)
        
        r = (r_R[i+1]+r_R[i])*(b/2)     # radial distance of the blade element
        dr = (r_R[i+1]-r_R[i])*b        # length of the blade element
        
        a_b4_Pr[j][i], a[j][i], a_tan[j][i], Cl[j][i], Cd[j][i], F_ax[j][i], F_tan[j][i], alfa[j][i], phi[j][i], F_tot[j][i], F_tip[j][i], F_root[j][i], dCT[j][i], dCQ[j][i], dCP[j][i], P0_down[j][i], Gamma[j][i] = \
        BladeElementMethod(Vinf, TSR[j], n[j], rho, b, r, root_pos_R, tip_pos_R, dr, Omega[j], Nb, a[j][i], a_tan[j][i], twist, chord[j][i], polar_alfa, polar_cl, polar_cd, tol, P_up[j][i])

        CT[j] += dCT[j][i]    # thrust coefficient for given J
        CP[j] += dCP[j][i]    # power coefficient for given J
        CQ[j] += dCQ[j][i]    # torque coefficient for given J

a_rms = np.sqrt(np.mean(np.square(a), axis=1))  # average induction factor for each advance ratio
U_wake = Vinf*(np.ones(len(a_rms))+a_rms)       # wake velocity [m/s]

CtrlPts, HS_vortex, results = [[] for i in range(3)]

# Solving Lifting Line Model

for i in range(len(U_wake)):
    CtrlPts.append(ControlPoint(r_R, b, blade_seg, chord_dist, np.deg2rad(twist_dist)))
    HS_vortex.append(HorseshoeVortex(l, U_wake[i], vor_fil, blade_seg, -Omega[i], r_R, np.ones(blade_seg), Nb, (chord_dist*b), np.deg2rad(twist_dist)))
    results.append(LiftingLineModel(HS_vortex[i], CtrlPts[i], polar_alfa, polar_cl, polar_cd, Vinf, Omega[i], rho, b, r_R, chord_dist, twist_dist, Nb, l, U_wake[i], vor_fil, Gamma[i]))


# Ensure that HS_vortex has been rotated if needed
# HS_vortex = CoordinateRotation(HS_vortex, blade_seg, Nb)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
colors = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red", 
                "tab:purple", "tab:brown", "tab:pink", "tab:gray", 
                "tab:olive", "tab:cyan"])

# Unpack and plot all vortex segments
for blade_idx in range(Nb * blade_seg):
    hs = HS_vortex[0][blade_idx]  # You probably visualized only one advance ratio's wake
    color = next(colors)
    
    nodes = []
    for vf_key in sorted(hs, key=lambda k: int("".join(filter(str.isdigit, k)))):
        vf = hs[vf_key]
        if not nodes:
            nodes.append(vf["pos1"])
        nodes.append(vf["pos2"])
    
    nodes = np.array(nodes)
    ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], color=color)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("Wake visualization (Horseshoe Vortices)")
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.show()

# Gemini Plotting

#--------------------------------- Plotting routine ---------------------------------
# Ensure you have the results from both BEM and LLM
# For BEM, 'r_R' (radial positions) and 'b' (blade radius) are used to get actual radial positions
radial_positions_bem = (r_R[:-1] + r_R[1:]) / 2 * b # Midpoint of each blade element
radial_positions_llm = results[0]['r'] # Radial positions from LLM control points

# 1. Radial distribution of the angle of attack
plt.figure(figsize=(10, 6))
plt.plot(radial_positions_bem, alfa[0], label='BEM')
plt.plot(radial_positions_llm, results[0]['alfa'], '--', label='LLM') # You'll need to calculate alfa from LLM results or compare circulation
plt.xlabel('Radial position (m)')
plt.ylabel('Angle of attack (degrees)')
plt.title('Radial Distribution of Angle of Attack')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Radial distribution of the inflow angle
plt.figure(figsize=(10, 6))
plt.plot(radial_positions_bem, np.rad2deg(phi[0]), label='BEM')
plt.plot(radial_positions_llm, np.rad2deg(results[0]['phi']), label='LLM')
# You'll need to calculate phi from LLM V_ax_local and V_tan_local (or equivalent)
# For now, let's plot a placeholder or just BEM if LLM inflow angle isn't directly calculated and stored
plt.xlabel('Radial position (m)')
plt.ylabel('Inflow angle (degrees)')
plt.title('Radial Distribution of Inflow Angle')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Radial distribution of the circulation (Gamma)
plt.figure(figsize=(10, 6))
# For BEM, Circulation (Gamma) = 0.5 * V_local * Cl * chord
Gamma_bem = 0.5 * np.sqrt((Vinf*(1+a[0]))*2 + (Omega[0]*radial_positions_bem*(1-a_tan[0]))**2) * Cl[0] * chord[0]
plt.plot(radial_positions_bem, Gamma_bem, label='BEM')
# For LLM, Gamma is directly available in results['Gamma']
plt.plot(radial_positions_llm, results[0]['Gamma'][0:blade_seg], label='LLM') # Assuming first 'blade_seg' elements of gamma correspond to the first blade
plt.xlabel('Radial position (m)')
plt.ylabel('Circulation (m^2/s)')
plt.title('Radial Distribution of Circulation')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Radial distribution of the tangential/azimuthal load
plt.figure(figsize=(10, 6))
# F_tan is directly available from BEM
plt.plot(radial_positions_bem, a_tan[0], label='BEM')
# F_tan is directly available from LLM
plt.plot(radial_positions_llm, results[0]['a_tan'], label='LLM')
plt.xlabel('Radial position (m)')
plt.ylabel('Tangential Load (N/m)')
plt.title('Radial Distribution of Tangential Induction')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 5. Radial distribution of the axial load
plt.figure(figsize=(10, 6))
# F_ax is directly available from BEM
plt.plot(radial_positions_bem, a[0], label='BEM')
# F_ax is directly available from LLM
plt.plot(radial_positions_llm, results[0]['a_ax'], label='LLM')
plt.xlabel('Radial position (m)')
plt.ylabel('Axial Load (N/m)')
plt.title('Radial Distribution of Axial Induction')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 6. Total thrust coefficient (CT) and power coefficient (CP)
# Assuming J is an array and you want to plot CT/CP vs J or for a single J
# For single J:
print(f"Total Thrust Coefficient (CT) for J={J[0]}:")
print(f"BEM: {CT[0]:.4f}")
print(f"LLM: {results[0]['CT']:.4f}")

print(f"\nTotal Power Coefficient (CP) for J={J[0]}:")
print(f"BEM: {CP[0]:.4f}")
print(f"LLM: {results[0]['CP']:.4f}")

# You can create a bar chart to compare CT and CP if you have multiple J values
# For now, let's just print the values. If you want a plot, specify how you'd like to visualize it.

# Non-dimensioning explanation:
# print("\n--- Explanation of Non-Dimensioning ---")
# print("The thrust coefficient (CT) is non-dimensioned by $1/2 \\rho A V_{inf}^2$, where $\\rho$ is air density, $A$ is rotor disk area ($A=\\pi R^2$), and $V_{inf}$ is freestream velocity.")
# print("In this code, for BEM, it's defined as $dCT = (F_{ax} \\cdot Nb \\cdot dr) / (\\rho \\cdot n^2 \\cdot (2R)^4)$[cite: 11].")
# print("For LLM, it's defined as $dCT = (F_{ax,l} \\cdot Nb \\cdot seg_{len}) / (\\rho \\cdot (\\Omega/(2\\pi))^2 \\cdot (2R)^4)$[cite: 11].")
# print("The power coefficient (CP) is non-dimensioned by $1/2 \\rho A V_{inf}^3$.")
# print("In this code, for BEM, it's defined as $dCP = (F_{ax} \\cdot Nb \\cdot dr \\cdot V_{inf}) / (\\rho \\cdot n^3 \\cdot (2R)^5)$[cite: 11].")
# print("For LLM, it's defined as $dCP = (F_{ax,l} \\cdot Nb \\cdot seg_{len} \\cdot V_{inf}) / (\\rho \\cdot (\\Omega/(2\\pi))^3 \\cdot (2R)^5)$[cite: 11].")
# print("The radial position is non-dimensioned by the blade radius $R$.")
# print("Loads (axial and tangential) are presented as force per unit length (N/m) or similar, which can be non-dimensioned by " \
# "dynamic pressure and chord length if needed for comparison across different conditions or airfoils.")
# print("Angle of attack and inflow angle are in degrees for direct interpretation.")
# print("Circulation is typically presented in physical units (m^2/s) but can be non-dimensioned by $V_{inf} R$ for comparison.")


# 7. Plots with explanation of sensitivity of results to choice of:
#    * assumed convection speed for frozen wake
#    * discretization of the blade (constant, cosine)
#    * azimuthal discretization (number of wake segments per rotation)
#    * length of the wake (number of rotations), including convergence of the solution with wake length

# To generate these plots, you would need to run your LiftingLineModel multiple times
# with varying parameters. For example, to show sensitivity to convection speed:

# Sensitivity to assumed convection speed for frozen wake
# (You would need to modify your 'HorseshoeVortex' function or pass U_wake as a variable to LLM)
# Example of how you would set up such a test:
# U_wake_test_values = [0.8 * Vinf, Vinf, 1.2 * Vinf] # Example values
# results_sensitivity_U_wake = {}
# for uw in U_wake_test_values:
#     # Re-run LLM with different U_wake. This might require changes to how U_wake is handled in your main loop.
#     # For this example, we'll assume you already have a mechanism to change it.
#     # HS_vortex_temp = HorseshoeVortex(l, uw, vor_fil, blade_seg, Omega[0], r_R)
#     # HS_vortex_temp = CoordinateRotation(HS_vortex_temp, blade_seg, Nb)
#     # results_sensitivity_U_wake[uw] = LiftingLineModel(HS_vortex_temp, CtrlPts, polar_alfa, polar_cl, polar_cd, Vinf, Omega[0], rho, b, r_R, chord_dist, twist_dist, Nb)

# # Plotting for sensitivity (example for Axial Load)
# plt.figure(figsize=(10, 6))
# for uw, res in results_sensitivity_U_wake.items():
#     plt.plot(res['r'], res['F_ax'], label=f'U_wake = {uw:.2f} m/s')
# plt.xlabel('Radial position (m)')
# plt.ylabel('Axial Load (N/m)')
# plt.title('Sensitivity of Axial Load to Assumed Convection Speed')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
# print("Discussion: (Explain how changing U_wake affects the results, e.g., higher U_wake leads to less induced velocity and thus different loads/angles).")

# Similarly, for discretization of the blade (seg_type):
# You would run your initial BladeSegment and then LLM with 'lin' and 'cos'
# (This is already set up in your code for 'seg_type', so you can just run it twice and plot the comparison.)
# Example:
# r_R_lin, chord_dist_lin, twist_dist_lin = BladeSegment(root_pos_R, tip_pos_R, pitch, (blade_seg+1), 'lin')
# CtrlPts_lin = ControlPoint(r_R_lin, b, blade_seg)
# HS_vortex_lin = HorseshoeVortex(l, U_wake[0], vor_fil, blade_seg, Omega[0], r_R_lin)
# HS_vortex_lin = CoordinateRotation(HS_vortex_lin, blade_seg, Nb)
# results_lin = LiftingLineModel(HS_vortex_lin, CtrlPts_lin, polar_alfa, polar_cl, polar_cd, Vinf, Omega[0], rho, b, r_R_lin, chord_dist_lin, twist_dist_lin, Nb)

# r_R_cos, chord_dist_cos, twist_dist_cos = BladeSegment(root_pos_R, tip_pos_R, pitch, (blade_seg+1), 'cos')
# CtrlPts_cos = ControlPoint(r_R_cos, b, blade_seg)
# HS_vortex_cos = HorseshoeVortex(l, U_wake[0], vor_fil, blade_seg, Omega[0], r_R_cos)
# HS_vortex_cos = CoordinateRotation(HS_vortex_cos, blade_seg, Nb)
# results_cos = LiftingLineModel(HS_vortex_cos, CtrlPts_cos, polar_alfa, polar_cl, polar_cd, Vinf, Omega[0], rho, b, r_R_cos, chord_dist_cos, twist_dist_cos, Nb)

# plt.figure(figsize=(10, 6))
# plt.plot(results_lin['r'], results_lin['Gamma'][0:blade_seg], label='Linear Discretization')
# plt.plot(results_cos['r'], results_cos['Gamma'][0:blade_seg], label='Cosine Discretization')
# plt.xlabel('Radial position (m)')
# plt.ylabel('Circulation (m^2/s)')
# plt.title('Sensitivity of Circulation to Blade Discretization')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
# print("Discussion: (Explain the differences between linear and cosine discretization, typically cosine clusters points near tip/root for better resolution where gradients are higher).")

# For azimuthal discretization (vor_fil - number of wake segments per rotation):
# (You would vary 'vor_fil' in your 'HorseshoeVortex' function)
# Example:
# vor_fil_test_values = [50, 100, 150, 200]
# results_sensitivity_vor_fil = {}
# for vf in vor_fil_test_values:
#     # HS_vortex_temp = HorseshoeVortex(l, U_wake[0], vf, blade_seg, Omega[0], r_R)
#     # HS_vortex_temp = CoordinateRotation(HS_vortex_temp, blade_seg, Nb)
#     # results_sensitivity_vor_fil[vf] = LiftingLineModel(HS_vortex_temp, CtrlPts, polar_alfa, polar_cl, polar_cd, Vinf, Omega[0], rho, b, r_R, chord_dist, twist_dist, Nb)

# plt.figure(figsize=(10, 6))
# for vf, res in results_sensitivity_vor_fil.items():
#     plt.plot(res['r'], res['CT'], 'o', label=f'vor_fil = {vf}') # Plotting total CT as a single point per run
# plt.xlabel('Number of Vortex Filaments (per rotation)')
# plt.ylabel('Total Thrust Coefficient (CT)')
# plt.title('Sensitivity of Total CT to Azimuthal Discretization')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
# print("Discussion: (Explain how increasing azimuthal discretization affects convergence and accuracy, typically more filaments lead to more accurate but computationally expensive results).")


# For length of the wake (l - number of rotations):
# (You would vary 'l' in your 'HorseshoeVortex' function)
# Example:
# wake_length_rotations = [5, 10, 20, 30] # Equivalent to l = N_rotations * (2*b)
# results_sensitivity_wake_length = {}
# for wl in wake_length_rotations:
#     # l_current = wl * (2*b)
#     # HS_vortex_temp = HorseshoeVortex(l_current, U_wake[0], vor_fil, blade_seg, Omega[0], r_R)
#     # HS_vortex_temp = CoordinateRotation(HS_vortex_temp, blade_seg, Nb)
#     # results_sensitivity_wake_length[wl] = LiftingLineModel(HS_vortex_temp, CtrlPts, polar_alfa, polar_cl, polar_cd, Vinf, Omega[0], rho, b, r_R, chord_dist, twist_dist, Nb)

# plt.figure(figsize=(10, 6))
# for wl, res in results_sensitivity_wake_length.items():
#     plt.plot(res['r'], res['CP'], 'o', label=f'Wake Length = {wl} Rotations') # Plotting total CP as a single point per run
# plt.xlabel('Number of Rotations in Wake Length')
# plt.ylabel('Total Power Coefficient (CP)')
# plt.title('Sensitivity of Total CP to Wake Length and Convergence')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
# print("Discussion: (Explain how increasing wake length affects the solution, typically it converges after a certain length as the influence diminishes. Also discuss the convergence of the solution with wake length.)")
