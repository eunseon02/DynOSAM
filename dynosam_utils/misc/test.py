import evo.tools
import numpy as np
import evo.core.geometry as evo_geo
import evo.core.lie_algebra as evo_lie

def so3_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    return R.from_quat(quaternion, scalar_first=False).as_matrix()

# q1 = np.array([0.00500072,0.997234,0.00314333,0.0740967])
# t1 = np.array([-17.0628,1.62373,222.847])

# t2 = np.array([-16.8142,1.63817,221.344])
# q2 = np.array([0.00838996,0.996871,0.00213994,0.0785667])


# mt1 = np.array([1.1535,0.410892,-1.37047])
# mq1 = np.array([0.00165347,-0.00168394,-0.0055539,0.999982])

# L1 = evo_lie.se3(so3_from_xyzw(q1), t1)
# L2 = evo_lie.se3(so3_from_xyzw(q2), t2)

# M_hat = evo_lie.se3(so3_from_xyzw(mq1), mt1)

# print(L1)
# print(L2)

# M = L2 @ np.linalg.inv(L1)

# print(M)
# print(M_hat)

# q1 = np.array([-0.00059185,0.997772,0.00530414,0.0665062])
# t1 = np.array([-20.7722,1.48931,249.727])

# # q2 = np.array([-0.00121539,0.997179,0.0055074,0.0748511])
# # t2 = np.array([-20.5208,1.50215,248.1])

# q2 = np.array([0.000354644,0.997746,0.00650005,0.0667937])
# t2 = np.array([-20.5619,1.49354,248.306])


# mt1 = np.array([7.15295,-0.199602,-0.473675])
# mq1 = np.array([-0.000953273,-0.0134832,0.00324328,0.999903])

#vdo
# q1 = np.array([0.00329204,0.997774,0.00364192,0.0664999])
# t1 = np.array([-20.7121,1.52677,249.433])


# q2 = np.array([0.00365954,0.997753,0.00568971,0.0666623])
# t2 = np.array([-20.5097,1.47777,248.133])



# mt1 = np.array([0.317726,0.146492,-1.59026])
# mq1 = np.array([0.000667651,-0.000134309,-0.00421709,0.999991])

q1 = np.array([-0.00059184986110275407487,0.99777174079081132341,0.0053041357745424117312,0.066506158345209592797])
t1 = np.array([-20.772209657395009685,1.4893054588712060227,249.72652270068567759])


q2 = np.array([-0.00031966793536338113753,0.99726106119501400915,0.0073348669301711759763,0.07359669370824332979])
t2 = np.array([-20.51207370079401926,1.4981205045950394261,248.08904595942675542])



mt1 = np.array([3.8067835945868044867,1.0248405011477057514,-1.3210742259054710779])
mq1 = np.array([0.0020512131770778524627,-0.0071060535736166899334,-0.00017382621319170121657,0.9999726327812971105])

L1 = evo_lie.se3(so3_from_xyzw(q1), t1)
L2 = evo_lie.se3(so3_from_xyzw(q2), t2)

M_hat = evo_lie.se3(so3_from_xyzw(mq1), mt1)

# print(L1)
# print(L2)

M = L2 @ np.linalg.inv(L1)

print(M)
print(M_hat)
