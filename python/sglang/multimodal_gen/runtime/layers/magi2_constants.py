# SPDX-License-Identifier: Apache-2.0

# gate * sigmoid(alpha * gate) * (up + 1), with the same clipping used by the
# routed MoE epilogue and the MAGI-2 dense/shared experts.
SWIGLU7_ALPHA = 1.702
SWIGLU7_LIMIT = 7.0
