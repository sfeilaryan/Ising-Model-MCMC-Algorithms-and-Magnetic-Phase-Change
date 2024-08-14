#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random as rnd
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import itertools
from copy import deepcopy
from PIL import Image, ImageDraw
import math

K_b = 1

def isingHamiltonian(latticeArray, extFieldArray = None, E = 1, J = 0): # Notice the wrap-around approximation!
    L = latticeArray.shape[0]
    
    energy = 0
    interparticle = 0
    external = 0
    
    for i in range(L):
        for j in range(L):
            interparticle -= latticeArray[i % L][j % L]*latticeArray[(i+1) % L][j % L]
            interparticle -= latticeArray[i % L][j % L]*latticeArray[i % L][(j+1) % L]
    
    interparticle *= E
    
    if extFieldArray is not None:
        for i in range(L):
            for j in range(L):
                external -= extFieldArray[i][j]*latticeArray[i][j]
        external *= J
        
    energy = interparticle + external
    
    return energy

def isingProbabilityNum(config):
    return np.exp(-(config.get_beta()*config.get_energy()))

class Configuration:
    
    def __init__(self, L = 100, spins = None, extFieldArray = None, E = 1, J = 0, temperature = 273.0):
        self.size = L
        if spins is not None:
            self.spins = spins
        else:
            config = np.zeros(shape = (L,L))
            for i in range(L):
                for j in range(L):
                    config[i][j] = rnd.choice([-1,1])
            self.spins = config
        
        self.E = E
        self.J = J
        self.extFieldArray = np.zeros(shape = (L,L)) if extFieldArray is None else extFieldArray
        self.T = temperature
        self.beta = 1/(K_b * self.T)
        self.energy = isingHamiltonian(self.spins, self.extFieldArray, self.E, self.J)
    
    def get_energy(self):
        return self.energy
    
    def get_magnetization(self):
        totalMag = 0
        for i in range(self.size):
            for j in range(self.size):
                totalMag += self.spins[i][j]
        return totalMag / self.size**2
    
    def set_energy(self, energy = None):
        if energy is not None:
            self.energy = energy
        else:
            self.energy = isingHamiltonian(self.spins, self.extFieldArray, self.E, self.J)
        
    def set_externalField(self, extField): self.extFieldArray = extField
        
    def set_temperature(self, temperature): self.T = temperature
    
    def get_temperature(self): return self.T
    
    def get_E_param(self): return self.E
    
    def set_E(self, value): self.E = value
    
    def set_J(self, value): self.J = value
    
    def get_J_param(self): return self.J
    
    def toggle_spin(self, i, j): self.spins[i][j] *= (-1)
        
    def get_spins(self): return self.spins
    
    def get_externalField(self):return self.extFieldArray
    
    def get_beta(self): return self.beta
    
    def get_size(self): return self.size
    
    def reset_beta(self): self.beta = 1/ (K_b * self.T)
    
def config_to_image(config):
    L = config.size
    im = np.zeros([L,L,3])
    for i,j in itertools.product(range(L), repeat=2):
        im[i,j,:] = (1.,0,0) if config.spins[i,j]==1 else (0.,0.,1.)
    return im

def metropolis_move_0(config): #Toggle one spin
    L = config.get_size()
    x, y = (rnd.choice(np.arange(L)), rnd.choice(np.arange(L)))
    lattice = config.get_spins()
    change = lattice[x][y] * (lattice[(x-1)%L][y]  + lattice[(x+1)%L][y] + lattice[x][(y-1)%L] + lattice[x][(y+1)%L])
    change *= config.get_E_param()
    change *= 2
    change += 2 * config.get_J_param() * config.get_externalField()[x][y] * lattice[x][y]
    
    A = np.exp(config.get_beta()*(-(change)))
    if rnd.uniform(0,1) < A:
        config.toggle_spin(x, y)
        config.set_energy(config.get_energy()+change)
        
def metropolis_move_1(config): #Global change
    newConfig = Configuration(L = config.get_size(), extFieldArray = config.get_externalField(), E = config.get_E_param(), J = config.get_J_param(), temperature = config.get_temperature())
    A = isingProbabilityNum(newConfig)/isingProbabilityNum(config)
    if rnd.uniform(0,1) < A:
        config = newConfig
        
def metropolis_move(config, protocol = 0):
    if protocol == 0:
        metropolis_move_0(config)
    elif protocol == 1:
        metropolis_move_1(config)
    else:
        print("Please provide valid protocol.")
        
def getAverageEveryting(config, cycleNumber, warmupNumber, cycleLength = None, protocol = 0):
    
    cycleLength = config.size**2 if cycleLength is None else cycleLength
    
    energyArray = np.zeros(warmupNumber + cycleNumber)
    energyArray2 = np.zeros(warmupNumber + cycleNumber)
    magArray = np.zeros(warmupNumber + cycleNumber)
    mag2array = np.zeros(warmupNumber + cycleNumber)
    
    for i in range(warmupNumber + cycleNumber):
        for _ in range(cycleLength):
            metropolis_move(config, protocol)
        
        energyArray[i] = config.get_energy()
        energyArray2[i] = config.get_energy()**2
        magArray[i] = config.get_magnetization()
        mag2array[i] = config.get_magnetization()**2
    
    return np.sum(energyArray[-cycleNumber:])/len(energyArray[-cycleNumber:]),np.sum(energyArray2[-cycleNumber:])/len(energyArray2[-cycleNumber:]) ,np.sum(magArray[-cycleNumber:])/len(magArray[-cycleNumber:]) , np.sum(mag2array[-cycleNumber:])/len(mag2array[-cycleNumber:])# magArray

def Hysterisis(size, time, protocol = 0):
    B_field_1 = np.linspace(2, -2, time)
    B_field_2 = np.linspace(-2, 2, time)
    
    magnetization_1 = np.zeros(time)
    magnetization_2 = np.zeros(time)
    
    config = Configuration(L=size, E=1, J=1, extFieldArray=2*np.ones(shape=(size, size)), temperature=3)
    
    for _ in range(size**2): metropolis_move(config, protocol)
    
    for i in range(time):
        config.set_externalField(B_field_1[i] * np.ones(shape=(size, size)))
        
        for _ in range(size**2): metropolis_move(config, protocol)
        
        magnetization_1[i] = config.get_magnetization()
        
    for _ in range(size**2): metropolis_move(config, protocol)
            
    for i in range(time):
        config.set_externalField(B_field_2[i] * np.ones(shape=(size, size)))
        
        for _ in range(size**2): metropolis_move(config, protocol)
        
        magnetization_2[i] = config.get_magnetization()
        
    return magnetization_1, magnetization_2

def ACF(magTimeSeries, lagT):   #Assumed warmed up configuration
    
    if lagT >= magTimeSeries.shape[0]:
        print("Please choose a lag smaller than sequence length.")
        return
    
    if lagT <= 0:
        print("Please choose a lag greater or equal to 1.")
        return
    
    N = magTimeSeries.shape[0]
    seriesAvg = np.sum(magTimeSeries)/N
    
    firstFactors = np.zeros(N - lagT)
    secondFactors = np.zeros(N - lagT)
    
    for i in range(N - lagT):
        
        firstFactors[i] = magTimeSeries[i] - seriesAvg
        secondFactors[i] = magTimeSeries[lagT + i] - seriesAvg
    
    stdDevSeries = np.zeros(N)
    
    for j in range(N):
        stdDevSeries[j] = (magTimeSeries[j] - seriesAvg)**2
  
    return np.sum(firstFactors*secondFactors)/np.sum(stdDevSeries)

def compute_ACT(config, protocol = 0, eps = 0.1, maxIterations = 10):     # Assume Warmed Up!
    N = config.get_size() ** 2
    
    n = 0
    
    lagT = 1
    
    while n < maxIterations:
        
        n+=1
        
        #Make a new array if it was too small to find tau.
        
        N*=2
        magTimeSeries = np.zeros(N)
        
        for i in range(N):
            
            metropolis_move(config, protocol)
            magTimeSeries[i] = config.get_magnetization()
            
        while lagT < N:
            
            computedACF = ACF(magTimeSeries, lagT)
            
            if computedACF < eps:
                return lagT
            
            lagT += 1
    return N

### Animation: uncomment to run

# frames = 500

# fig_1 = plt.figure(figsize=(10, 6))
# gs = fig_1.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.5)
# ax_1 = fig_1.add_subplot(gs[:, 0])
# ax_energy = fig_1.add_subplot(gs[0, 1])
# ax_magnet = fig_1.add_subplot(gs[1, 1])

# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.9)

# config = Configuration(L=50, E=1, J=0, extFieldArray=np.ones(shape=(100, 100)), temperature=1)

# lattice = ax_1.imshow(config_to_image(config), animated=True)
# ax_1.set_title('Lattice')

# initial_T = config.get_temperature()
# ax_slider_T = plt.axes([0.25, 0.1, 0.65, 0.03]) 
# T_slider = Slider(ax_slider_T, 'Temperature', valmin=0, valmax=3, valinit=initial_T)

# initial_J = config.get_J_param()
# ax_slider_J = plt.axes([0.25, 0.05, 0.65, 0.03])
# J_slider = Slider(ax_slider_J, 'B field', valmin=-3, valmax=3, valinit=initial_J)

# time = np.array([0])
# energy = np.array([config.get_energy()])
# magnet = np.array([config.get_magnetization()])

# energy_plt, = ax_energy.plot(time, energy, ms=1)
# ax_energy.set_title('Energy')
# ax_energy.set_xlim([0, 500])
# ax_energy.set_ylim([min(energy) - 200, max(energy) + 200])

# magnet_plt, = ax_magnet.plot(time, magnet, ms=1)
# ax_magnet.set_title('Magnetization')
# ax_magnet.set_xlim([0, 500])
# ax_magnet.set_ylim([-1.1, 1.1])

# def update(t):
#     global energy, magnet, time
#     for _ in range(config.size**2):
#         metropolis_move(config, protocol=0)

#     lattice.set_data(config_to_image(config))

#     energy = np.append(energy, config.get_energy())
#     time = np.arange(len(energy))
#     energy_plt.set_data(time, energy)
#     ax_energy.set_xlim([max(0, len(energy) - 500), max(len(energy), 500)])
#     ax_energy.set_ylim([min(energy[max(0, len(energy) - 500):]) - 200, max(energy[max(0, len(energy) - 500):]) + 200])
    
#     magnet = np.append(magnet, config.get_magnetization())
#     magnet_plt.set_data(time, magnet)
#     ax_magnet.set_xlim([max(0, len(magnet) - 500), max(len(magnet), 500)])
#     ax_magnet.set_ylim([min(magnet[max(0, len(magnet) - 500):]) - 0.1, max(magnet[max(0, len(magnet) - 500):]) + 0.1])

#     return lattice, energy_plt, magnet_plt

# def update_slider_T(val):
#     config.set_temperature(val)
#     config.reset_beta()
#     update(0)

# def update_slider_J(val):
#     config.set_J(val)
#     config.set_energy()
#     update(0)

# T_slider.on_changed(update_slider_T)
# J_slider.on_changed(update_slider_J)

# ani = FuncAnimation(fig_1, update, frames=frames, blit=True, interval=20)

# plt.show()

### end of animation section

### "Continuous" Ising model spin values in [-pi, pi] and nearest neighbor interactions

def arrowHamiltonian(latticeArray, extFieldArray = None, E = 1, J = 0): # Notice the wrap-around approximation!
    L = latticeArray.shape[0]
    
    energy = 0
    interparticle = 0
    external = 0
    
    for i in range(L):
        for j in range(L):
            interparticle -= math.cos(latticeArray[i % L][j % L]*math.pi-latticeArray[(i+1) % L][j % L]*math.pi)
            interparticle -= math.cos(latticeArray[i % L][j % L]*math.pi-latticeArray[i % L][(j+1) % L]*math.pi)
    interparticle *= E
    
    if extFieldArray is not None:
        for i in range(L):
            for j in range(L):
                external -= extFieldArray[i][j]*math.cos(latticeArray[i][j]*math.pi)
        external *= J
        
    energy = interparticle + external
    
    return energy

def arrowProbabilityNum(arrow):
    return np.exp(-(arrow.get_beta()*arrow.get_energy()))

class Arrow:
    
    def __init__(self, L = 100, spins = None, extFieldArray = None, E = 1, J = 0, temperature = 273.0):
        self.size = L
        if spins is not None:
            self.spins = spins
        else:
            config = np.zeros(shape = (L,L))
            for i in range(L):
                for j in range(L):
                    config[i][j] = rnd.uniform(-1,1)
            self.spins = config
        
        self.E = E
        self.J = J
        self.extFieldArray = np.zeros(shape = (L,L)) if extFieldArray is None else extFieldArray
        self.T = temperature
        self.beta = 1/(K_b * self.T)
        self.energy = arrowHamiltonian(self.spins, self.extFieldArray, self.E, self.J)
    
    def get_energy(self): return self.energy
    
    def get_magnetization(self):
        totalMag = np.array([0., 0.])
        for i in range(self.size):
            for j in range(self.size):
                totalMag += np.array([- np.sin(self.spins[i][j] * np.pi), np.cos(self.spins[i][j] * np.pi)])
        return totalMag / self.size**2
    
    def get_magnetization_magnitude(self): 
        m = self.get_magnetization()
        return (m[0]**2 + m[1]**2)**0.5
    
    def set_energy(self, energy = None):
        if energy is not None:
            self.energy = energy
        else:
            self.energy = arrowHamiltonian(self.spins, self.extFieldArray, self.E, self.J)
        
    def set_externalField(self, extField): self.extFieldArray = extField
        
    def set_temperature(self, temperature): self.T = temperature
    
    def get_temperature(self): return self.T
    
    def get_E_param(self): return self.E
    
    def set_E(self, value): self.E = value
    
    def set_J(self, value): self.J = value
    
    def get_J_param(self): return self.J
    
    def toggle_spin(self, i, j, orientation): self.spins[i][j] = orientation
        
    def get_spins(self): return self.spins
    
    def get_externalField(self): return self.extFieldArray
    
    def get_beta(self): return self.beta
    
    def get_size(self): return self.size
    
    def reset_beta(self): self.beta = 1/ (K_b * self.T)
    
def get_color(t):
    r, g, b = 0, 0, 0
    
    H = 180 + t * 180
    H_prime = H / 60
    X = 1 - abs(H_prime % 2 - 1)
    if 0 <= H_prime < 1:
        r, g, b = 1, X, 0
    elif 1 <= H_prime < 2:
        r, g, b = X, 1, 0
    elif 2 <= H_prime < 3:
        r, g, b = 0, 1, X
    elif 3 <= H_prime < 4:
        r, g, b = 0, X, 1
    elif 4 <= H_prime < 5:
        r, g, b = X, 0, 1
    elif 5 <= H_prime < 6:
        r, g, b = 1, 0, X
    return (r, g, b)

def arrow_to_image(arrow):
    L = arrow.size
    im = np.zeros([L,L,3])
    for i in range(L):
        for j in range(L):
            im[i,j,:] = get_color(arrow.spins[i][j])
    return im

def arrow_metropolis_move_0(arrow): #Toggle one spin
    L = arrow.get_size()
    x, y = (rnd.choice(np.arange(L)), rnd.choice(np.arange(L)))
    new_orientation = rnd.uniform(-1,1)
    lattice = arrow.get_spins()
    
    old = 0
    old -= math.cos(lattice[(x + 1) % L][y] * math.pi - lattice[x][y] * math.pi ) + math.cos(lattice[(x - 1) % L][y] * math.pi  - lattice[x][y] * math.pi ) + math.cos(lattice[x][(y + 1) % L] * math.pi  - lattice[x][y] * math.pi ) +  math.cos(lattice[x][(y - 1) % L] * math.pi  - lattice[x][y] * math.pi)
    old *= arrow.get_E_param()
    old -= arrow.get_J_param() * arrow.get_externalField()[x][y] * math.cos(lattice[x][y]*math.pi)
    
    new = 0
    new -= math.cos(lattice[(x + 1) % L][y] * math.pi  - new_orientation * math.pi ) + math.cos(lattice[(x - 1) % L][y] * math.pi  - new_orientation * math.pi ) + math.cos(lattice[x][(y + 1) % L] * math.pi  - new_orientation * math.pi ) +  math.cos(lattice[x][(y - 1) % L] * math.pi  - new_orientation * math.pi)
    new *= arrow.get_E_param()
    new -= arrow.get_J_param() * arrow.get_externalField()[x][y] * math.cos(new_orientation*math.pi)
    
    change = new - old
    
    A = np.exp(arrow.get_beta()*(-(change)))
    if rnd.uniform(0,1) < A:
        arrow.toggle_spin(x, y, new_orientation)
        arrow.set_energy(arrow.get_energy()+change)
        
def arrow_metropolis_move(arrow, protocol = 0):
    if protocol == 0:
        arrow_metropolis_move_0(arrow)
    else:
        print("Please provide valid protocol.")
        
def getAverageEveryting_arrow(arrow, cycleNumber, warmupNumber, cycleLength = None, protocol = 0):
    
    cycleLength = arrow.size**2 if cycleLength is None else cycleLength
    
    energyArray = np.zeros(warmupNumber + cycleNumber)
    energyArray2 = np.zeros(warmupNumber + cycleNumber)
    magArray = np.zeros(warmupNumber + cycleNumber)
    mag2array = np.zeros(warmupNumber + cycleNumber)
    
    for i in range(warmupNumber + cycleNumber):
        for _ in range(cycleLength):
            arrow_metropolis_move(arrow, protocol)
        
        energyArray[i] = arrow.get_energy()
        energyArray2[i] = arrow.get_energy()**2
        magArray[i] = arrow.get_magnetization_magnitude()
        mag2array[i] = arrow.get_magnetization_magnitude()**2
    
    return np.sum(energyArray[-cycleNumber:])/len(energyArray[-cycleNumber:]),np.sum(energyArray2[-cycleNumber:])/len(energyArray2[-cycleNumber:]) ,np.sum(magArray[-cycleNumber:])/len(magArray[-cycleNumber:]) , np.sum(mag2array[-cycleNumber:])/len(mag2array[-cycleNumber:])

def Hysterisis_arr(size, time, protocol = 0):
    B_field_1 = np.linspace(2, -2, time)
    B_field_2 = np.linspace(-2, 2, time)
    
    magnetization_1 = np.zeros(time)
    magnetization_2 = np.zeros(time)
    
    arrow = Arrow(L=size, E=1, J=1, extFieldArray=2*np.ones(shape=(size, size)), temperature=0.2)
    
    for _ in range(size**2): arrow_metropolis_move(arrow, protocol)
    
    for i in range(time):
        arrow.set_externalField(B_field_1[i] * np.ones(shape=(size, size)))
        
        for _ in range(size**2): arrow_metropolis_move(arrow, protocol)
        
        magnetization_1[i] = arrow.get_magnetization_magnitude()
        
    for _ in range(size**2): arrow_metropolis_move(arrow, protocol)
            
    for i in range(time):
        arrow.set_externalField(B_field_2[i] * np.ones(shape=(size, size)))
        
        for _ in range(size**2): arrow_metropolis_move(arrow, protocol)
        
        magnetization_2[i] = arrow.get_magnetization_magnitude()
        
    return magnetization_1, magnetization_2

def Hysterisis_arr(size, time, protocol = 0):
    B_field_1 = np.linspace(2, -2, time)
    B_field_2 = np.linspace(-2, 2, time)
    
    magnetization_1 = np.zeros(time)
    magnetization_2 = np.zeros(time)
    
    arrow = Arrow(L=size, E=1, J=1, extFieldArray=2*np.ones(shape=(size, size)), temperature=0.2)
    
    for _ in range(size**3): arrow_metropolis_move(arrow, protocol)
    
    for i in range(time):
        arrow.set_externalField(B_field_1[i] * np.ones(shape=(size, size)))
        
        for _ in range(size**2): arrow_metropolis_move(arrow, protocol)
        
        magnetization_1[i] = arrow.get_magnetization()[1]
        
    for _ in range(size**2): arrow_metropolis_move(arrow, protocol)
            
    for i in range(time):
        arrow.set_externalField(B_field_2[i] * np.ones(shape=(size, size)))
        
        for _ in range(size**2): arrow_metropolis_move(arrow, protocol)
        
        magnetization_2[i] = arrow.get_magnetization()[1]
        
    return magnetization_1, magnetization_2

### Animation (without arrows at latice points): uncomment to run

# frames = 500

# fig_2 = plt.figure(figsize=(10, 6))
# gs = fig_2.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.5)
# ax_2 = fig_2.add_subplot(gs[:, 0])
# ax_energy = fig_2.add_subplot(gs[0, 1])
# ax_magnet = fig_2.add_subplot(gs[1, 1])
# ax_square = fig_2.add_subplot(gs[1, 2])

# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9)

# arrow = Arrow(L=60, E=1, J=0, extFieldArray=np.ones(shape=(100, 100)), temperature=0.5)

# lattice = ax_2.imshow(arrow_to_image(arrow), animated=True)
# ax_2.set_title('Lattice')

# initial_T = arrow.get_temperature()
# ax_slider_T = plt.axes([0.25, 0.1, 0.65, 0.03])
# T_slider = Slider(ax_slider_T, 'Temperature', valmin=0, valmax=3, valinit=initial_T)

# initial_J = arrow.get_J_param()
# ax_slider_J = plt.axes([0.25, 0.05, 0.65, 0.03])
# J_slider = Slider(ax_slider_J, 'B field', valmin=-3, valmax=3, valinit=initial_J)

# energy = np.array([arrow.get_energy()])
# time = np.array([0])
# magnet = np.array([arrow.get_magnetization_magnitude()])
# x_hist = np.array([0])
# y_hist = np.array([0])

# energy_plt, = ax_energy.plot(time, energy, ms=1)
# ax_energy.set_title('Energy')
# ax_energy.set_xlim([0, 500])
# ax_energy.set_ylim([min(energy) - 200, max(energy) + 200])

# magnet_plt, = ax_magnet.plot(time, magnet, ms=1)
# ax_magnet.set_title('Magnetization magnitude')
# ax_magnet.set_xlim([0, 500])
# ax_magnet.set_ylim([-1.1, 1.1])

# direction_plt = ax_square.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color="red")
# dir_hist_plt, = ax_square.plot(x_hist, y_hist, zorder=0)
# circle_x = np.array([np.cos(x) for x in np.linspace(-np.pi / 2, 3 * np.pi / 2, 500)])
# circle_y = np.array([np.sin(x) for x in np.linspace(-np.pi / 2, 3 * np.pi / 2, 500)])
# vals = np.ones((500, 4))
# for i in range(500): vals[i, 0], vals[i, 1], vals[i, 2] = get_color(-1.0001 + 2 * i / 499)
# newcmp = ListedColormap(vals)
# ax_square.scatter(circle_x, circle_y, c=np.linspace(-1, 1, 500), cmap=newcmp, zorder=0)
# ax_square.set_title('Magnetization direction')
# ax_square.set_xlim([-1.1, 1.1])
# ax_square.set_ylim([-1.1, 1.1])
# ax_square.set_aspect('equal')

# def update(t):
#     global energy, magnet, time, x_hist, y_hist

#     for _ in range(arrow.size**2):
#         arrow_metropolis_move(arrow, protocol=0)

#     lattice.set_data(arrow_to_image(arrow))
    
#     energy = np.append(energy, arrow.get_energy())
#     time = np.arange(len(energy))
#     energy_plt.set_data(time, energy)
#     ax_energy.set_xlim([max(0, len(energy) - 500), max(len(energy), 500)])
#     ax_energy.set_ylim([min(energy[max(0, len(energy) - 500):]) - 200, max(energy[max(0, len(energy) - 500):]) + 200])
    
#     magnet = np.append(magnet, arrow.get_magnetization_magnitude())
#     magnet_plt.set_data(time, magnet)
#     ax_magnet.set_xlim([max(0, len(magnet) - 500), max(len(magnet), 500)])
#     ax_magnet.set_ylim([min(magnet[max(0, len(magnet) - 500):]) - 0.1, max(magnet[max(0, len(magnet) - 500):]) + 0.1])
    
#     d = arrow.get_magnetization()
#     direction_plt.set_UVC(d[0], d[1])
    
#     x_hist = np.append(x_hist, d[0])
#     y_hist = np.append(y_hist, d[1])
#     dir_hist_plt.set_data(x_hist, y_hist)
    
#     return lattice, energy_plt, magnet_plt

# def update_slider_T(val):
#     arrow.set_temperature(val)
#     arrow.reset_beta()

# def update_slider_J(val):
#     arrow.set_J(val)
#     arrow.set_energy()

# T_slider.on_changed(update_slider_T)
# J_slider.on_changed(update_slider_J)

# ani = FuncAnimation(fig_2, update, frames=frames, blit=True, interval=20)

# plt.show()

### end of animation

### Animation (with arrows at latice points, it is recomended to run on a smaller grid): uncomment to run

# frames = 500

# fig_2 = plt.figure(figsize=(10, 6))
# gs = fig_2.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.5)
# ax_2 = fig_2.add_subplot(gs[:, 0])
# ax_energy = fig_2.add_subplot(gs[0, 1])
# ax_magnet = fig_2.add_subplot(gs[1, 1])
# ax_square = fig_2.add_subplot(gs[1, 2])

# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9)

# arrow = Arrow(L=30, E=1, J=0, extFieldArray=np.ones(shape=(100, 100)), temperature=1)

# lattice = ax_2.imshow(arrow_to_image(arrow), animated=True)
# ax_2.set_title('Lattice')

# initial_T = arrow.get_temperature()
# ax_slider_T = plt.axes([0.25, 0.1, 0.65, 0.03])
# T_slider = Slider(ax_slider_T, 'Temperature', valmin=0, valmax=3, valinit=initial_T)

# initial_J = arrow.get_J_param()
# ax_slider_J = plt.axes([0.25, 0.05, 0.65, 0.03])
# J_slider = Slider(ax_slider_J, 'B field', valmin=-3, valmax=3, valinit=initial_J)

# energy = np.array([arrow.get_energy()])
# time = np.array([0])
# magnet = np.array([arrow.get_magnetization_magnitude()])
# x_hist = np.array([0])
# y_hist = np.array([0])

# energy_plt, = ax_energy.plot(time, energy, ms=1)
# ax_energy.set_title('Energy')
# ax_energy.set_xlim([0, 500])
# ax_energy.set_ylim([min(energy) - 200, max(energy) + 200])

# magnet_plt, = ax_magnet.plot(time, magnet, ms=1)
# ax_magnet.set_title('Magnetization')
# ax_magnet.set_xlim([0, 500])
# ax_magnet.set_ylim([-1.1, 1.1])

# direction_plt = ax_square.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color="red")
# dir_hist_plt, = ax_square.plot(x_hist, y_hist, zorder=0)
# circle_x = np.array([np.cos(x) for x in np.linspace(-np.pi / 2, 3 * np.pi / 2, 500)])
# circle_y = np.array([np.sin(x) for x in np.linspace(-np.pi / 2, 3 * np.pi / 2, 500)])
# vals = np.ones((500, 4))
# for i in range(500): vals[i, 0], vals[i, 1], vals[i, 2] = get_color(-1.0001 + 2 * i / 499)
# newcmp = ListedColormap(vals)
# ax_square.scatter(circle_x, circle_y, c=np.linspace(-1, 1, 500), cmap=newcmp, zorder=0)
# ax_square.set_title('Magnetization direction')
# ax_square.set_xlim([-1.1, 1.1])
# ax_square.set_ylim([-1.1, 1.1])
# ax_square.set_aspect('equal')

# def direction(x, arrow):
#     u = - 10 * np.sin(arrow.spins[x[1]][x[0]] * np.pi)
#     v = 10 * np.cos(arrow.spins[x[1]][x[0]] * np.pi)
    
#     return u, v

# X_ar, Y_ar = np.mgrid[0:arrow.size, 0:arrow.size]
# U_ar = np.zeros_like(X_ar)
# V_ar = np.zeros_like(Y_ar)
# for i in range(arrow.size):
#     for j in range(arrow.size):
#         U_ar[i, j], V_ar[i, j] = direction([i, j], arrow)

# q = ax_2.quiver(X_ar, Y_ar, U_ar, V_ar)

# def update_3(t):
#     global energy, magnet, time, X_ar, Y_ar

#     for _ in range(arrow.size):
#         arrow_metropolis_move(arrow, protocol=0)
    
#     U_ar = np.zeros_like(X_ar)
#     V_ar = np.zeros_like(Y_ar)
#     for i in range(arrow.size):
#         for j in range(arrow.size):
#             U_ar[i, j], V_ar[i, j] = direction([i, j], arrow)
    
#     lattice.set_data(arrow_to_image(arrow))
#     q.set_UVC(U_ar, V_ar)
    
#     energy = np.append(energy, arrow.get_energy())
#     time = np.arange(len(energy))
#     energy_plt.set_data(time, energy)
#     ax_energy.set_xlim([max(0, len(energy) - 500), max(len(energy), 500)])
#     ax_energy.set_ylim([min(energy[max(0, len(energy) - 500):]) - 200, max(energy[max(0, len(energy) - 500):]) + 200])
    
#     magnet = np.append(magnet, arrow.get_magnetization_magnitude())
#     magnet_plt.set_data(time, magnet)
#     ax_magnet.set_xlim([max(0, len(magnet) - 500), max(len(magnet), 500)])
#     ax_magnet.set_ylim([min(magnet[max(0, len(magnet) - 500):]) - 0.1, max(magnet[max(0, len(magnet) - 500):]) + 0.1])
    
#     d = arrow.get_magnetization()
#     direction_plt.set_UVC(d[0], d[1])
    
#     # x_hist = np.append(x_hist, d[0])
#     # y_hist = np.append(y_hist, d[1])
#     # dir_hist_plt.set_data(x_hist, y_hist)

#     return lattice, # energy_plt, magnet_plt

# def update_slider_T(val):
#     arrow.set_temperature(val)
#     arrow.reset_beta()

# def update_slider_J(val):
#     arrow.set_J(val)
#     arrow.set_energy()

# T_slider.on_changed(update_slider_T)
# J_slider.on_changed(update_slider_J)

# ani = FuncAnimation(fig_2, update_3, frames=frames, blit=True, interval=20)

# plt.show()

### end of animation 