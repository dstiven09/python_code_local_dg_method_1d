from __future__ import annotations
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from visualization_settings import *


class SimulationResult:

    def __init__(self, grid_x, element_indicies, element_derivative,
                 bathymetry, still_water_depth) -> None:
        self.number_of_corrected_degress_of_freedom = 0
        self.q_in_time = []
        self.grid_x = grid_x
        self.element_indicies = element_indicies
        self.corrected_element_indicies = []
        self.element_derivative = element_derivative
        self.still_water_depth = still_water_depth
        self.bathymetry = bathymetry
        self.criteria = None

    def plot_water_hight_at_index(self, step_index: int):
        min_element = min(self.corrected_element_indicies[step_index])
        max_element = max(self.corrected_element_indicies[step_index])
        min_plot_element = np.where(self.element_indicies[:, :] == min_element)[0][0]
        max_plot_element = np.where(self.element_indicies[:, :] == max_element)[0][0]
        element_range = max_plot_element - min_plot_element
        i_min = min_plot_element - int(element_range)*0.2
        i_max = max_plot_element + int(element_range)*0.2
        water_height = self.q_in_time[step_index][:, 0] + self.bathymetry
        fig, ax = plt.subplots()
        for i in range(len(self.element_indicies)):
            if i_min <= i <= i_max:
                if any(self.element_indicies[i, 0] ==
                       self.corrected_element_indicies[step_index]):
                    ax.plot(self.grid_x[i], water_height[self.element_indicies[i, 0]], 'ro')
                else:
                    ax.plot(self.grid_x[i], water_height[self.element_indicies[i, 0]], 'bo')
        ax.set_ylabel('$h+b$')
        ax.set_xlabel('X-axis')
        plt.show()

    def plot_solution_at_index(self, step_index: int):
        water_height = self.q_in_time[step_index][:, 0] + self.bathymetry
        fig, axs = plt.subplots(2, 2)
        for i in range(len(self.element_indicies)):
            if any(self.element_indicies[i, 0] ==
                   self.corrected_element_indicies[step_index]):
                axs[0, 0].plot(self.grid_x[i],
                               water_height[self.element_indicies[i, 0]], 'ro')
            else:
                axs[0, 0].plot(self.grid_x[i],
                               water_height[self.element_indicies[i, 0]], 'bo')
        axs[0, 0].set_ylabel('$h+b$')

        for i in range(len(self.element_indicies)):
            if any(self.element_indicies[i, 0] ==
                   self.corrected_element_indicies[step_index]):
                axs[0, 1].plot(
                    self.grid_x[i],
                    self.q_in_time[step_index][self.element_indicies[i, 0],
                                               1], 'ro')
            else:
                axs[0, 1].plot(
                    self.grid_x[i],
                    self.q_in_time[step_index][self.element_indicies[i, 0],
                                               1], 'bo')
        axs[0, 1].set_ylabel('$hu$')

        for i in range(len(self.element_indicies)):
            if any(self.element_indicies[i, 0] ==
                   self.corrected_element_indicies[step_index]):
                axs[1, 0].plot(
                    self.grid_x[i],
                    self.q_in_time[step_index][self.element_indicies[i, 0],
                                               2], 'ro')
            else:
                axs[1, 0].plot(
                    self.grid_x[i],
                    self.q_in_time[step_index][self.element_indicies[i, 0],
                                               2], 'bo')
        axs[1, 0].set_ylabel('$hw$')

        for i in range(len(self.element_indicies)):
            if any(self.element_indicies[i, 0] ==
                   self.corrected_element_indicies[step_index]):
                axs[1, 1].plot(
                    self.grid_x[i],
                    self.q_in_time[step_index][self.element_indicies[i, 0],
                                               3], 'ro')
            else:
                axs[1, 1].plot(
                    self.grid_x[i],
                    self.q_in_time[step_index][self.element_indicies[i, 0],
                                               3], 'bo')
        axs[1, 1].set_ylabel('$pnh$')

        plt.show()

    def append_time_step(self, number_of_corrected_degrees_of_freedom: int,
                         q: np.array,
                         corrected_element_indicies: np.array) -> None:
        self.number_of_corrected_degress_of_freedom += number_of_corrected_degrees_of_freedom
        self.q_in_time.append(q)
        self.corrected_element_indicies.append(corrected_element_indicies)

    def calculate_criteria(self) -> dict:
        if self.criteria is None:
            time_steps = len(self.q_in_time)
            number_of_grid_points, _ = self.q_in_time[0].shape
            criteria = {
                'H_PLUS_B_MINUS_D_DIV_H': np.zeros(
                    (time_steps, number_of_grid_points)),
                'HU': np.zeros((time_steps, number_of_grid_points)),
                'U': np.zeros((time_steps, number_of_grid_points)),
                'HW': np.zeros((time_steps, number_of_grid_points)),
                'W': np.zeros((time_steps, number_of_grid_points)),
                'HW_X': np.zeros((time_steps, number_of_grid_points)),
                'PNH': np.zeros((time_steps, number_of_grid_points)),
                'PNH_X': np.zeros((time_steps, number_of_grid_points)),
                'W_X': np.zeros((time_steps, number_of_grid_points)),
                'U_X': np.zeros((time_steps, number_of_grid_points)),
                'H_X': np.zeros((time_steps, number_of_grid_points)),
                'HU_X': np.zeros((time_steps, number_of_grid_points)),
            }
            for time_index, current_q in enumerate(self.q_in_time):
                h_plus_b_minus_d_div_h = (current_q[:, 0] + self.bathymetry[:] -
                                        self.still_water_depth) / current_q[:, 0]
                hu = current_q[:, 1]
                u = current_q[:, 1] / current_q[:, 0]
                hw = current_q[:, 2]
                w = current_q[:, 2] / current_q[:, 0]
                h = current_q[:, 0]
                pnh = current_q[:, 3]

                pnh_x = (self.element_derivative.transpose(0, 2, 1)
                        @ pnh[self.element_indicies, np.newaxis]).squeeze()
                hw_x = (self.element_derivative.transpose(0, 2, 1)
                        @ hw[self.element_indicies, np.newaxis]).squeeze()

                w_x = (self.element_derivative.transpose(0, 2, 1)
                    @ w[self.element_indicies, np.newaxis]).squeeze()

                u_x = (self.element_derivative.transpose(0, 2, 1)
                    @ u[self.element_indicies, np.newaxis]).squeeze()

                hu_x = (self.element_derivative.transpose(0, 2, 1)
                        @ hu[self.element_indicies, np.newaxis]).squeeze()

                h_x = (self.element_derivative.transpose(0, 2, 1)
                    @ h[self.element_indicies, np.newaxis]).squeeze()

                criteria['H_PLUS_B_MINUS_D_DIV_H'][
                    time_index, :] = h_plus_b_minus_d_div_h
                criteria['HU'][time_index, :] = hu
                criteria['U'][time_index, :] = u
                criteria['HW'][time_index, :] = hw
                criteria['W'][time_index, :] = w
                criteria['HW_X'][time_index, :] = hw_x.reshape(
                    number_of_grid_points)
                criteria['PNH'][time_index, :] = pnh
                criteria['PNH_X'][time_index, :] = pnh_x.reshape(
                    number_of_grid_points)
                criteria['W_X'][time_index, :] = w_x.reshape(number_of_grid_points)
                criteria['U_X'][time_index, :] = u_x.reshape(number_of_grid_points)
                criteria['H_X'][time_index, :] = h_x.reshape(number_of_grid_points)
                criteria['HU_X'][time_index, :] = hu_x.reshape(
                    number_of_grid_points)
                self.criteria = criteria
        return self.criteria

    def plot_criteria_at_time_index(self, step_index: int):
        criteria_latex = {
            'H_PLUS_B_MINUS_D_DIV_H': '$\\frac{h+b-d}{h}$',
            'HU': '$hu$',
            'U': '$u$',
            'HW': '$hw$',
            'W': '$w$',
            'HW_X': '$hw_x$',
            'PNH': '$p^{nh}$',
            'PNH_X': '$p^{nh}_x$',
            'W_X': '$w_x$',
            'U_X': '$u_x$',
            'H_X': '$h_x$',
            'HU_X': '$hu_x$',
        }

        criteria = self.calculate_criteria()

        fig, axs = plt.subplots(6, 2)
        fig.suptitle(f'Criteria at time step {step_index}', fontsize=16)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        plt.subplots_adjust(hspace=0.3)
        for index, criterion_name in enumerate(criteria):
            water_height = criteria[criterion_name][step_index, :]
            for i in range(len(self.element_indicies)):
                if any(self.element_indicies[i, 0] ==
                       self.corrected_element_indicies[step_index]):
                    axs[index % 6, index // 6].plot(
                        self.grid_x[i],
                        water_height[self.element_indicies[i, 0]], 'r.')
                else:
                    axs[index % 6, index // 6].plot(
                        self.grid_x[i],
                        water_height[self.element_indicies[i, 0]], 'b.')
            axs[index % 6,
                index // 6].set_ylabel(criteria_latex[criterion_name],
                                       rotation=0,
                                       fontsize='small')
            axs[index % 6, index // 6].get_yaxis().set_label_coords(-0.15, 0.5)

        plt.show()

    def plot_criteria_norm(self):

        criteria = self.calculate_criteria()
        time_steps = len(self.q_in_time)
        output_criteria = {
            'H_PLUS_B_MINUS_D_DIV_H': np.zeros(time_steps),
            'HU': np.zeros(time_steps),
            'U': np.zeros(time_steps),
            'HW': np.zeros(time_steps),
            'W': np.zeros(time_steps),
            'HW_X': np.zeros(time_steps),
            'PNH': np.zeros(time_steps),
            'PNH_X': np.zeros(time_steps),
            'W_X': np.zeros(time_steps),
            'U_X': np.zeros(time_steps),
            'H_X': np.zeros(time_steps),
            'HU_X': np.zeros(time_steps),
        }
        for time_index in range(time_steps):
            output_criteria['H_PLUS_B_MINUS_D_DIV_H'][time_index] = np.max(
                abs(criteria['H_PLUS_B_MINUS_D_DIV_H'][time_index, :]))
            output_criteria['HU'][time_index] = np.max(
                abs(criteria['HU'][time_index, :]))
            output_criteria['U'][time_index] = np.max(
                abs(criteria['U'][time_index, :]))
            output_criteria['HW'][time_index] = np.max(
                abs(criteria['HW'][time_index, :]))
            output_criteria['W'][time_index] = np.max(
                abs(criteria['W'][time_index, :]))
            output_criteria['HW_X'][time_index] = np.max(
                abs(criteria['HW_X'][time_index, :]))
            output_criteria['PNH'][time_index] = np.max(
                abs(criteria['PNH'][time_index, :]))
            output_criteria['PNH_X'][time_index] = np.max(
                abs(criteria['PNH_X'][time_index, :]))
            output_criteria['W_X'][time_index] = np.max(
                abs(criteria['W_X'][time_index, :]))
            output_criteria['U_X'][time_index] = np.max(
                abs(criteria['U_X'][time_index, :]))
            output_criteria['H_X'][time_index] = np.max(
                abs(criteria['H_X'][time_index, :]))
            output_criteria['HU_X'][time_index] = np.max(
                abs(criteria['HU_X'][time_index, :]))

        criteria_latex = {
            'H_PLUS_B_MINUS_D_DIV_H': '$\\frac{h+b-d}{h}$',
            'HU': '$hu$',
            'U': '$u$',
            'HW': '$hw$',
            'W': '$w$',
            'HW_X': '$hw_x$',
            'PNH': '$p^{nh}$',
            'PNH_X': '$p^{nh}_x$',
            'W_X': '$w_x$',
            'U_X': '$u_x$',
            'H_X': '$h_x$',
            'HU_X': '$hu_x$',
        }

        for criterion_name in output_criteria:
            plt.plot(output_criteria[criterion_name],
                     label=criteria_latex[criterion_name])
            plt.legend()
        plt.show()

    def save(self, output_file: str) -> None:
        #output_file_path = os.path.abspath(output_file)
        #os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(self, f)

        print("Data saved.")

    @classmethod
    def load(cls, input_file: str) -> SimulationResult:
        with open(input_file, 'rb') as f:
            obj = pickle.load(f)
            obj.criteria = None
            return obj

    def get_number_of_time_steps(self):
        return len(self.q_in_time)
