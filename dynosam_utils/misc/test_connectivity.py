import gtsam
import gtsam.utils.plot as gtsam_plot
from typing import List, Optional
import numpy as np

def custom_error_function(measurement: np.ndarray, this: gtsam.CustomFactor,
                          values: gtsam.Values,
                          jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    key_m, key_h, key_x = this.keys()
    m = values.atPoint3(key_m)
    h = values.atPoint3(key_h)
    x = values.atPoint3(key_x)

    predicted_m = x + h + m
    error = predicted_m - measurement

    if jacobians is not None:
        # Fake Jacobians (identity for simplicity)
        jacobians[0] = np.eye(3)
        jacobians[1] = np.eye(3)
        jacobians[2] = np.eye(3)

    return error

def custom_smooth_brain_Z(hm2, hm1, h):
    return hm2 + hm1 + h

def custom_smooth_brain_function(measurement: np.ndarray, this: gtsam.CustomFactor,
                          values: gtsam.Values,
                          jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    key_hm2, key_hm1, key_h = this.keys()
    hm2 = values.atPoint3(key_hm2)
    hm1 = values.atPoint3(key_hm1)
    h = values.atPoint3(key_h)

    predicted_m = custom_smooth_brain_Z(hm2, hm1, h)
    error = predicted_m - measurement

    if jacobians is not None:
        # Fake Jacobians (identity for simplicity)
        jacobians[0] = np.eye(3)
        jacobians[1] = np.eye(3)
        jacobians[2] = np.eye(3)

    return error

def main():
    graph = gtsam.NonlinearFactorGraph()
    noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 4)

    # Keys
    # key_m1 = gtsam.symbol('m', 1)
    key_x1 = gtsam.symbol('x', 1)
    key_h1 = gtsam.symbol('h', 1)

    key_x2 = gtsam.symbol('x', 2)
    key_h2 = gtsam.symbol('h', 2)

    key_x3 = gtsam.symbol('x', 3)
    key_h3 = gtsam.symbol('h', 3)

    key_x4 = gtsam.symbol('x', 4)
    key_h4 = gtsam.symbol('h', 4)


    key_m1 = gtsam.symbol('m', 1)
    key_m2 = gtsam.symbol('m', 2)
    key_m3 = gtsam.symbol('m', 3)

    x1 =  gtsam.Point3(1, 2, 3)
    x2 =  gtsam.Point3(2, 2, 2)
    h1 = gtsam.Point3(1, 1, 1)
    h2 = gtsam.Point3(1, 1, 4)

    x3 = gtsam.Point3(1, 1, 1)
    h3 = gtsam.Point3(1, 1, 4)

    x4 = gtsam.Point3(1, 1, 1)
    h4 = gtsam.Point3(1, 1, 4)

    m1 = gtsam.Point3(1, 2, 3)
    m2 = gtsam.Point3(2, 3, 4)
    m3 = gtsam.Point3(1, 5, 1)
    # m4 = gtsam.Point3(3, 6, 1)
    # m5 = gtsam.Point3(6, 1, 3)


    def make_measurement(x, h, m):
        return x + h + m

    prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

    graph.add(gtsam.PriorFactorPoint3(key_x1, x1, prior_noise))
    graph.add(gtsam.PriorFactorPoint3(key_x2, x2, prior_noise))
    graph.add(gtsam.PriorFactorPoint3(key_x3, x3, prior_noise))
    graph.add(gtsam.PriorFactorPoint3(key_x4, x4, prior_noise))

    graph.add(gtsam.PriorFactorPoint3(key_h1, h1, prior_noise))

    # connect m1 to all x's, h's
    graph.add(gtsam.CustomFactor(noise_model, [key_m1, key_x1, key_h1],
                                 lambda *args: custom_error_function(make_measurement(x1, h1, m1), *args)))
    graph.add(gtsam.CustomFactor(noise_model, [key_m1, key_x2, key_h2],
                                 lambda *args: custom_error_function(make_measurement(x2, h2, m1), *args)))
    graph.add(gtsam.CustomFactor(noise_model, [key_m1, key_x3, key_h3],
                                 lambda *args: custom_error_function(make_measurement(x3, h3, m1), *args)))
    graph.add(gtsam.CustomFactor(noise_model, [key_m1, key_x4, key_h4],
                                 lambda *args: custom_error_function(make_measurement(x4, h4, m1), *args)))

    # connext m2 to 1,2,3
    graph.add(gtsam.CustomFactor(noise_model, [key_m2, key_x1, key_h1],
                                 lambda *args: custom_error_function(make_measurement(x1, h1, m2), *args)))
    graph.add(gtsam.CustomFactor(noise_model, [key_m2, key_x2, key_h2],
                                 lambda *args: custom_error_function(make_measurement(x2, h2, m2), *args)))
    graph.add(gtsam.CustomFactor(noise_model, [key_m2, key_x3, key_h3],
                                 lambda *args: custom_error_function(make_measurement(x3, h3, m2), *args)))
    # connect m3 to 3,4
    graph.add(gtsam.CustomFactor(noise_model, [key_m3, key_x3, key_h3],
                                 lambda *args: custom_error_function(make_measurement(x3, h3, m3), *args)))
    graph.add(gtsam.CustomFactor(noise_model, [key_m3, key_x4, key_h4],
                                 lambda *args: custom_error_function(make_measurement(x4, h4, m3), *args)))

    # add smoothing
    # h1, h2, h3
    graph.add(gtsam.CustomFactor(noise_model, [key_h1, key_h2, key_h3],
                                 lambda *args: custom_smooth_brain_function(custom_smooth_brain_Z(h1, h2, h3), *args)))
    # h2, h3, h4
    graph.add(gtsam.CustomFactor(noise_model, [key_h2, key_h3, key_h4],
                                 lambda *args: custom_smooth_brain_function(custom_smooth_brain_Z(h2, h3, h4), *args)))

    # separation on m2
    # Add custom factors (ensuring at least 3 on H and X)
    # graph.add(gtsam.CustomFactor(noise_model, [key_m1, key_x1, key_h1],
    #                              lambda *args: custom_error_function(z1, *args)))

    # graph.add(gtsam.CustomFactor(noise_model, [key_m2, key_x1, key_h1],
    #                              lambda *args: custom_error_function(z2, *args)))
    # graph.add(gtsam.CustomFactor(noise_model, [key_m2, key_x2, key_h2],
    #                              lambda *args: custom_error_function(z3, *args)))
    # graph.add(gtsam.CustomFactor(noise_model, [key_m3, key_x2, key_h2],
    #                              lambda *args: custom_error_function(z4, *args)))
    # graph.add(gtsam.CustomFactor(noise_model, [key_m4, key_x2, key_h2],
    #                              lambda *args: custom_error_function(z5, *args)))

    # no separation - connect m3 to x1
    # graph.add(gtsam.CustomFactor(noise_model, [key_m1, key_x1, key_h1],
    #                              lambda *args: custom_error_function(make_measurement(x1, h1, m1), *args)))
    # # connect m5 to x1
    # graph.add(gtsam.CustomFactor(noise_model, [key_m5, key_x1, key_h1],
    #                              lambda *args: custom_error_function(make_measurement(x1, h1, m5), *args)))

    # graph.add(gtsam.CustomFactor(noise_model, [key_m5, key_x2, key_h2],
    #                              lambda *args: custom_error_function(make_measurement(x2, h2, m5), *args)))

    # graph.add(gtsam.CustomFactor(noise_model, [key_m1, key_x2, key_h2],
    #                              lambda *args: custom_error_function(make_measurement(x2, h2, m1), *args)))

    # # graph.add(gtsam.CustomFactor(noise_model, [key_m5, key_x2, key_h2],
    # #                              lambda *args: custom_error_function(make_measurement(x2, h2, m5), *args)))

    # graph.add(gtsam.CustomFactor(noise_model, [key_m2, key_x1, key_h1],
    #                              lambda *args: custom_error_function(make_measurement(x1, h1, m2), *args)))
    # # graph.add(gtsam.CustomFactor(noise_model, [key_m2, key_x2, key_h2],
    # #                              lambda *args: custom_error_function(make_measurement(x2, h2, m2), *args)))
    # graph.add(gtsam.CustomFactor(noise_model, [key_m3, key_x2, key_h2],
    #                              lambda *args: custom_error_function(make_measurement(x2, h2, m3), *args)))
    # graph.add(gtsam.CustomFactor(noise_model, [key_m4, key_x2, key_h2],
    #                              lambda *args: custom_error_function(make_measurement(x2, h2, m4), *args)))


    # Initial values
    values = gtsam.Values()
    values.insert(key_m1, m1)
    values.insert(key_x1, x1)
    values.insert(key_h1, h1)
    values.insert(key_m2, m2)
    values.insert(key_x2, x2)
    values.insert(key_h2, h2)
    values.insert(key_x3, x3)
    values.insert(key_x4, x4)
    values.insert(key_h3, h3)
    values.insert(key_h4, h4)
    values.insert(key_m3,m3)
    # values.insert(key_m5,m5)
    # values.insert(key_m4,m4)

    graph.saveGraph("test.dot", values)

    # print(f"Error {graph.error(values)}")

    gaussian_fg = graph.linearize(values)
    # print(type(gaussian_fg))
    colamd_ordering = graph.orderingCOLAMD()
    print(colamd_ordering)
    # print(dir(gtsam.Ordering))
    constrain_last = [key_m1, key_m3, key_h4]
    # print(type(constrain_last))
    # colamd_ordering = gtsam.Ordering.ColamdConstrainedLastGaussianFactorGraph(gaussian_fg, constrain_last)
    # # colamd_ordering = gtsam.Ordering.ColamdConstrainedLastNonlinearFactorGraph(graph=graph, constrainLast=[x2, h2])

    print(colamd_ordering)

    # # print(gaussian_fg.augmentedJacobian(colamd_ordering))
    gaussian_elimination_tree = gaussian_fg.eliminateMultifrontal(colamd_ordering)
    gaussian_elimination_tree.saveGraph("test_bayes_tree1.dot")
    # # Convert to Bayes Tree
    # bayes_tree = graph.eliminateSequential()

    # # Log Bayes Tree to a file
    # with open("bayes_tree_log.txt", "w") as f:
    #     f.write(str(bayes_tree))

    # print("Bayes tree logged to bayes_tree_log.txt")

if __name__ == "__main__":
    main()
