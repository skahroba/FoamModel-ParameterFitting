import numpy as np
from Relperm import *

def dryout(x, sw):
    return 0.5 + np.arctan(x[1]*(sw-x[2]))/np.pi

########################################################################
########################################################################
# three parameter function (fmmob,epdry,fmdry)
def fm_mod_f2(s_w, f2):
    s_w = s_w.reshape([len(s_w),1])
    fm = 1 + f2[0] * (0.5 + np.arctan(f2[1] * (s_w - f2[2])) / np.pi)
    return fm


def fg_mod_f2(s_w, swc, sgr, krw0, krg0, nw, ng, mug, muw, f2):
    krf = (krg(s_w, swc, sgr, krg0, ng) / fm_mod_f2(s_w, f2))
    fg = ((krf / mug) / (krw(s_w, swc, sgr, krw0, nw) / muw + krf / mug))
    return fg


def muf_mod_f2(s_w, swc, sgr, krw0, krg0, nw, ng, mug, muw, f2):
    krf = (krg(s_w, swc, sgr, krg0, ng) / fm_mod_f2(s_w, f2))
    mu_foam = (1 / (krw(s_w, swc, sgr, krw0, nw) / muw + krf / mug))
    return mu_foam

########################################################################
########################################################################
# five parameter function (fmmob,epdry,fmdry,fmcap,epcap)
def fm_mod_f2f5(s_w, f2, f5, mu_f, u_t, sigma_wg):
    s_w = s_w.reshape([len(s_w),1])
    mu_f = mu_f.reshape([len(mu_f),1])
    fm = (1 + f2[0] * (0.5 + np.arctan(f2[1] * (s_w - f2[2])) / np.pi) *
          (f5[0] / (mu_f * u_t / sigma_wg)) ** f5[1])
    return fm


def fg_mod_f2f5(s_w, swc, sgr, krw0, krg0, nw, ng, mug, muw, sigma_wg, f2, f5, mu_f, u_t):
    krf = (krg(s_w, swc, sgr, krg0, ng) / fm_mod_f2f5(s_w, f2, f5, mu_f, u_t, sigma_wg))
    fg = ((krf / mug) / (krw(s_w, swc, sgr, krw0, nw) / muw + krf / mug))
    return fg


def muf_mod_f2f5(s_w, swc, sgr, krw0, krg0, nw, ng, mug, muw, sigma_wg, f2, f5, mu_f, u_t):
    krf = (krg(s_w, swc, sgr, krg0, ng) / fm_mod_f2f5(s_w, f2, f5, mu_f, u_t, sigma_wg))
    mu_foam = (1 / (krw(s_w, swc, sgr, krw0, nw) / muw + krf / mug))
    return mu_foam
########################################################################
########################################################################
# seven parameter function (fmmob,epdry,fmdry,fmcap,epcap,csurf,fmsurf,epsurf)

def fm_mod_f1f2f5(s_w, f1,f2, f5, mu_f, u_t, sigma_wg, csurf):
    s_w = s_w.reshape([len(s_w),1])
    mu_f = np.reshape(mu_f, (np.size(mu_f), 1))
    fm = (1 + f2[0] * (0.5 + np.arctan(f2[1] * (s_w - f2[2])) / np.pi) *
          ((f5[0] / (mu_f * u_t / sigma_wg)) ** f5[1])*(csurf/f1[0])**f1[1])
    return fm

def fg_mod_f1f2f5(s_w, swc, sgr, krw0, krg0, nw, ng, mug, muw, sigma_wg, csurf, f1, f2, f5, mu_f, u_t):
    krf = (krg(s_w, swc, sgr, krg0, ng) / fm_mod_f1f2f5(s_w, f1, f2, f5, mu_f, u_t, sigma_wg, csurf))
    fg = ((krf / mug) / (krw(s_w, swc, sgr, krw0, nw) / muw + krf / mug))
    return fg


def muf_mod_f1f2f5(s_w, swc, sgr, krw0, krg0, nw, ng, mug, muw, sigma_wg, csurf, f1, f2, f5, mu_f, u_t):
    krf = (krg(s_w, swc, sgr, krg0, ng) / fm_mod_f1f2f5(s_w, f1, f2, f5, mu_f, u_t, sigma_wg, csurf))
    mu_foam = (1 / (krw(s_w, swc, sgr, krw0, nw) / muw + krf / mug))
    return mu_foam