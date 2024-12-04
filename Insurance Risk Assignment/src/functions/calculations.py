import sympy as sy


def question2():
    t, r = sy.symbols("theta r")
    c1, c2, alpha1, alpha2 = sy.symbols("c_1 c_2 alpha_1 alpha_2")

    # Nominator coefficient equations
    eq1 = sy.Eq(4000 * t, c1 * alpha1 + c2 * alpha2)
    eq2 = sy.Eq(t, c1 * alpha1 * alpha2 + c2 * alpha1 * alpha2)

    # quadratic formula a,b,c
    a = 64000000 * (1 + t) ** 2
    b = -4000 * (4 * t + 3) * (1 + t)
    c = t**2 + t

    # quadratic formula result roots
    alpha1_val = (-b - sy.sqrt(b**2 - 4 * a * c)) / (2 * a)
    alpha2_val = (-b + sy.sqrt(b**2 - 4 * a * c)) / (2 * a)

    # use eq1 to get c1 in terms of c2 
    c1_val = sy.solve(eq1.subs({alpha1: alpha1_val, alpha2: alpha2_val}), c1)[0]

    # Substitute all of these values into eq2 to get c2 in terms of theta 
    c2_val = sy.solve(eq2.subs({alpha1: alpha1_val, alpha2: alpha2_val, c1:c1_val}), c2)[0]

    # Substitute c2 back into the equation of c1 to get a formula in theta only 
    c1_val = c1_val.subs({c2: c2_val})

    # Simplify the equations 
    c1_val = sy.simplify(c1_val)
    c2_val = sy.simplify(c2_val)
    alpha1_val = sy.simplify(alpha1_val)
    alpha2_val = sy.simplify(alpha2_val)


    # I simplified them all by hand and the results I got are the following: (bunlarin usttekilere esit oldugunu kontrol ettim ama siz de kontrol edebilirsiniz isterseniz)
    alpha1_simplified = (4 * t + 3 - sy.sqrt(8*t+9)) / (32000 * (1 + t))
    alpha2_simplified = (4 * t + 3 - sy.sqrt(8*t+9)) / (32000 * (1 + t))
    c1_simplified = (64000000 * (t**2 + t) * (4 * t + 5 + sy.sqrt(8 * t + 9))) / (-(8*t+9)+((4 * t + 3) * sy.sqrt(8 * t + 9)))
    c2_simplified = (64000000 * (t + 1) * (sy.sqrt(8 * t + 9) -2*t-3)) / (sy.sqrt(8 * t + 9))

    u = sy.symbols("u")

    ruin_prob = c1 * sy.exp(-alpha1 * u) + c2 * sy.exp(-alpha2 * u)

    ruin_prob = ruin_prob.subs({alpha1: alpha1_simplified, alpha2: alpha2_simplified, c1:c1_simplified, c2:c2_simplified, u:16000})
    ruin_prob.subs({t:1}).evalf() # bu absurt bi deger veriyo
