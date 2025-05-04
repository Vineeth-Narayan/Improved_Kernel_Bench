def gen_for_correctness(LLM_server, max_iterations, for_optimization:bool):
    for i in range (max_iterations):
        if i == 0:
	        prompt = gen_first_prompt(problem, for_optimization)
        else:
            prompt = gen_fix_*(custom_cuda,error_metadata)
        custom_cuda = LLM_server(prompt)

        print(compilation_start) #> temp_path
        kernel_exec_result = eval(custom_cuda) #> temp_path
        print(compilation_end) #> temp_path

        if compiled and correct:
            return custom_cuda, iterations
        get_error_metadata()
    
    return None, iterations

def gen_for_optimization(LLM_server, recommendation:str):


def main():
    LLM_servers = []
    for t in range(min_temp, max_temp, stride):
        LLM_servers.append(create_LLM_server(t))

    con = []  
    for l in LLM_servers:
        correct_cuda = gen_for_correctness(l, max_iteration, False)
        if correct_cuda: con.append(correct_cuda)

"""
temps = [0, 0.5, 1, 1.5]
for temp in temps
    for i in max_iteration:
        cuda = generate (feedback, temp, cuda)
        feedback = compile_eval_profile(cuda)
    
"""


temps = [0, 0.5, 1, 1.5]
recommendations = ["tensor core", None]

best_cuda = gen_for_correct()
profiling = get_profile(best_cuda)

for rec in recommendations:
    for temp in temps:
        while i < max_iteration:
            cuda, iterations_used = gen_for_opt(temp, rec, best_cuda, profiling, max_ierations-i)
            i += iterations_used
            if (cuda):
                best_cuda = cuda if cuda runs faster
                profiling = get_profile(best_cuda)

def gen_for_opt(temp, rec, cuda, profiling, max_ierations):
    first_prompt = gen_prompt(rec, cuda, profiling)
    return gen_for_correctness(cuda, first_prompt)

def gen_for_correctness(cuda=None, first_prompt=None, max_iterations):
    prompt = first_prompt (or gen_first_prompt_for_correctness())
    for i in max_iterations:
        cuda = LLM(prompt)
        feedback = compile_eval(cuda)
        if compiled and correct:
            return custom_cuda, i
        prompt = gen_fix_prompt(cuda, feedback)    
    return None, i

    

    
    

    



    



        
