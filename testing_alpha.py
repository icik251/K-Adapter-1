def get_curr_alpha(step, t_total):
    return max(0.0, float(t_total - step) / float(max(1.0, t_total)))


print(get_curr_alpha(810, 810))
