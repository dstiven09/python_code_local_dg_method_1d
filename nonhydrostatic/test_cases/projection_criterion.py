from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt


class ProjectionCriterionType(Enum):
    GLOBAL = auto()
    H_PLUS_B_MINUS_D_DIV_H = auto()
    HU = auto()
    U = auto()
    HW = auto()
    W = auto()
    HW_X = auto()
    HW_X_PROTECTED = auto()
    PNH = auto()
    PNH_X = auto()
    W_X = auto()
    W_X_PROTECTED = auto()
    U_X = auto()
    U_X_PROTECTED = auto()
    H_X = auto()
    H_X_PROTECTED = auto()
    HU_X = auto()
    HU_X_PROTECTED = auto()

    def __str__(self) -> str:
        return self.name


class ProjectionCriterion:

    def __init__(self, type: ProjectionCriterionType,
                 threshold: float) -> None:
        self.type = type
        self.threshold = threshold

    def __str__(self) -> str:
        return self.type.__str__()


def calculate_criteria(dg_element, elliptic_factors, flux_divergence, q):

    b = np.zeros(dg_element.doflength)
    b[:] = elliptic_factors.Eqell.btopo[:]
    h_plus_b_minus_d_div_h = (q[:, 0] + b[:] - flux_divergence.Eq.d) / q[:, 0]
    hu = q[:, 1]
    u = q[:, 1] / q[:, 0]
    hw = q[:, 2]
    w = q[:, 2] / q[:, 0]
    pnh = q[:, 3]
    pnh_x = np.zeros(dg_element.doflength)
    for ielmt in range(flux_divergence.Gr.elength):
        pnh_x[dg_element.elementdofs[ielmt]] = np.dot(
            (dg_element.ddx[ielmt]).T, q[dg_element.elementdofs[ielmt], 3])

    hw_x = np.zeros(dg_element.doflength)
    for ielmt in range(flux_divergence.Gr.elength):
        hw_x[dg_element.elementdofs[ielmt]] = np.dot(
            (dg_element.ddx[ielmt]).T, hw[dg_element.elementdofs[ielmt]])

    w_x = np.zeros(dg_element.doflength)
    for ielmt in range(flux_divergence.Gr.elength):
        w_x[dg_element.elementdofs[ielmt]] = np.dot(
            (dg_element.ddx[ielmt]).T, w[dg_element.elementdofs[ielmt]])

    u_x = np.zeros(dg_element.doflength)
    for ielmt in range(flux_divergence.Gr.elength):
        u_x[dg_element.elementdofs[ielmt]] = np.dot(
            (dg_element.ddx[ielmt]).T, u[dg_element.elementdofs[ielmt]])

    hu_x = np.zeros(dg_element.doflength)
    for ielmt in range(flux_divergence.Gr.elength):
        hu_x[dg_element.elementdofs[ielmt]] = np.dot(
            (dg_element.ddx[ielmt]).T, hu[dg_element.elementdofs[ielmt]])

    h = q[:, 0]
    h_x = np.zeros(dg_element.doflength)
    for ielmt in range(flux_divergence.Gr.elength):
        h_x[dg_element.elementdofs[ielmt]] = np.dot(
            (dg_element.ddx[ielmt]).T, h[dg_element.elementdofs[ielmt]])

    criteria = {
        'H_PLUS_B_MINUS_D_DIV_H': h_plus_b_minus_d_div_h,
        'HU': hu,
        'U': u,
        'HW': hw,
        'W': w,
        'HW_X': hw_x,
        'PNH': pnh,
        'PNH_X': pnh_x,
        'W_X': w_x,
        'U_X': u_x,
        'H_X': h_x,
        'HU_X': hu_x
    }
    for criterion in criteria:
        criteria[criterion] = np.max(abs(criteria[criterion]))

    return criteria


def plot_criteria(criteria_in_time):
    plot_outputs = []
    for criterion in criteria_in_time[0]:
        output = np.zeros(len(criteria_in_time))
        for i in range(len(criteria_in_time)):
            output[i] = criteria_in_time[i][criterion]
        plot_outputs.append(output)
    criteria_names = [criterion_name for criterion_name in criteria_in_time[0]]

    for criterion_name, plot_output in zip(criteria_names, plot_outputs):
        # if criterion_name != 'HU':
        plt.plot(plot_output, label=criterion_name)
        plt.legend()
    plt.savefig('results/original_1000_iter.png')