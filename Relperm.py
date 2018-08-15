import numpy as np

def sws(s_w, swc, sgr):
    s_w = s_w.reshape([len(s_w),1])
    s_ws = np.ndarray(shape=(len(s_w),1),
                        buffer = np.zeros([len(s_w),1]),
                        dtype = float)
    b = ((s_w > swc) &  (s_w < (1-sgr))).astype(int)
    c = (s_w>=(1-sgr)).astype(int)
    s_ws = b * ((s_w-swc)/(1-sgr-swc)) + c * 1

    return s_ws


def krg(s_w, swc, sgr, krg0, ng):
    s_w = s_w.reshape([len(s_w),1])
    kr_g = np.ndarray(shape=(len(s_w),1),
                        buffer = np.zeros([len(s_w),1]),
                        dtype = float)
    b = (s_w >= swc).astype(int)
    c = (s_w<swc).astype(int)
    kr_g = b * (krg0 * (1-sws(s_w, swc, sgr)) ** ng) + c *(1+(krg0-1)/swc * s_w)
    return kr_g



def krw(s_w, swc, sgr, krw0, nw):
    s_w = s_w.reshape([len(s_w),1])
    kr_w = np.ndarray(shape=(len(s_w),1),
                        buffer = np.zeros([len(s_w),1]),
                        dtype = float)
    b = (s_w <= (1 - sgr)).astype(int)
    c = (s_w > (1 - sgr)).astype(int)
    kr_w = b * (krw0 * sws(s_w, swc, sgr) ** nw) + c * ((-(1 - krw0) / sgr * (1.0 - s_w) + 1.0))
    return kr_w
