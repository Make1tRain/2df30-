from models.model import model, model_gamma_4_16, model_gamma_02_08, model_question_6
import sys 

# This method is to run the simulation on multiple terminals. 
def run_multiple_terminals(model1:model, i): 
    u0List = ((16000*i),)
    thetaList = [0.01, 0.1, 0.5, 0.9, 1 ,2]

    model1.set_name(f"u={16000*i}")
    df = model1.simulate(u0List, thetaList)
    return df

def run_on_single_terminal(model, n): 
    u0List = [16000 * i for i in [0,1,2,3,4,5]]
    thetaList = [0.01, 0.1, 0.5, 0.9, 1 ,2]
    df = model.simulate_multiprocessing(u0List, thetaList,n=n)
    print(df)
    return df

def run_question(i:int): 
    """_summary_

    Args:
        i (int): Question Number
    """

    LAMBDA_ARRIVAL= 4
    CLAIMSIZE_MEAN = 16000 
    TIME_HORIZON = 2000 

    df = None 

    if i == 1: 
        m = model(TIME_HORIZON, LAMBDA_ARRIVAL, CLAIMSIZE_MEAN) 
        m.set_name("q1_model")
        df = run_on_single_terminal(m, n=1000)

    elif i == 5.1: 
        m = model_gamma_4_16(TIME_HORIZON, LAMBDA_ARRIVAL, CLAIMSIZE_MEAN) 
        m.set_name("q5_gamma_4_16_model")
        df = run_on_single_terminal(m, n=1000)

    elif i == 5.2: 
        m = model_gamma_02_08(TIME_HORIZON, LAMBDA_ARRIVAL, CLAIMSIZE_MEAN) 
        m.set_name("q5_gamma_02_08_model")
        df = run_on_single_terminal(m, n=1000)

    elif i == 6: 
        m = model_question_6(TIME_HORIZON, LAMBDA_ARRIVAL, CLAIMSIZE_MEAN)
        df = run_on_single_terminal(m, n=1000)

    return df

def main(): 
    """To run the model on multiple terminals for question 1, 
    run this script with the parameters [0,1,2,3,4,5] on different terminals.

    To run the model on a single terminal for question 1, 
    run this script with the parameter 'x' on one terminal. 

    """

    LAMBDA_ARRIVAL= 4
    CLAIMSIZE_MEAN = 16000 
    TIME_HORIZON = 2000  

    if sys.argv[1] == "q1": 
        m = model(TIME_HORIZON, LAMBDA_ARRIVAL, CLAIMSIZE_MEAN) 
        m.set_name("q1_model")
        run_on_single_terminal(m)

    elif sys.argv[1] == "q5.1": 
        m = model_gamma_4_16(TIME_HORIZON, LAMBDA_ARRIVAL, CLAIMSIZE_MEAN) 
        m.set_name("q5_gamma_4_16_model")
        run_on_single_terminal(m)

    elif sys.argv[1] == "q5.2": 
        m = model_gamma_02_08(TIME_HORIZON, LAMBDA_ARRIVAL, CLAIMSIZE_MEAN) 
        m.set_name("q5_gamma_02_08_model")
        run_on_single_terminal(m)

    else:
        pass 

if __name__ == "__main__":
    main()
