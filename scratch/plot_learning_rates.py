# caffe learning rates:
# Return the current learning rate. The currently implemented learning rate
# policies are as follows:
#    - fixed: always return base_lr.
#    - step: return base_lr * gamma ^ (floor(iter / step))
#    - exp: return base_lr * gamma ^ iter
#    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
#    - multistep: similar to step but it allows non uniform steps defined by
#      stepvalue
#    - poly: the effective learning rate follows a polynomial decay, to be
#      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
#    - sigmoid: the effective learning rate follows a sigmod decay
#      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
#
# where base_lr, max_iter, gamma, step, stepvalue and power are defined
# in the solver parameter protocol buffer, and iter is the current iteration.


# https://cs231n.github.io/neural-networks-3/#anneal
#   
# Step decay:
#    Reduce the learning rate by some factor every few epochs. Typical values
#    might be reducing the learning rate by a half every 5 epochs, or by 0.1
#    every 20 epochs. These numbers depend heavily on the type of problem and
#    the model. One heuristic you may see in practice is to watch the validation
#    error while training with a fixed learning rate, and reduce the learning
#    rate by a constant(e.g. 0.5) whenever the validation error stops improving.
# Exponential decay:
#    has the mathematical form α = α_0 e^{−kt}, where α0, k are hyperparameters
#    and t is the iteration number(but you can also use units of epochs).
# 1 / t decay:
#    has the mathematical form α = α_0 / (1 + kt) where a0, k are hyperparameters
#    and t is the iteration number.
