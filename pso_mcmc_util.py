# =================
# log prob PSO taken from William Sheu's H_0 constant code
# creates negative log likelihood function for minimization
# =================

def log_prob_pso(xl, model, p_params_inits, fix_params):
    resp = []
    for x in xl:
        counter = 0
        inputx = []
        for i, p_v in enumerate(p_params_inits):
            if fix_params[i]:
                inputx += [p_v]
            else:
                inputx += [x[counter]]
                counter += 1
        resp += [-1*model.ll_after_pso(inputx, get_plot = False)]
    return resp


def placeholder(x, use = True):
    if use:
        return 4
    if not use:
        return x