def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)

def _adjust_learning_rate(optimizer, i_iter, MAX_ITERS, POWER, learning_rate):
    lr = lr_poly(learning_rate, i_iter, MAX_ITERS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate(optimizer, i_iter, MAX_ITERS, POWER, learning_rate):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, MAX_ITERS, POWER, learning_rate)
