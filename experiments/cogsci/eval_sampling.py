import os
import pandas as pd
import time
from math import ceil
from battleship.grammar import BattleshipGrammar
from battleship.scoring import compute_score_parallel
from battleship.board import Board
from tqdm import tqdm

grammar = BattleshipGrammar(include_lambdas=False)

RESULTS_FILENAME = "results.csv"
COMMAND_FILENAME = "command.txt"

def repeated_generation(samples: int = 10000, min_depth: int = 1, max_depth: int = 16):
    generations = []
    while len(generations) != samples:
        for _ in range(samples - len(generations)):
            prog = grammar.sample(min_depth=min_depth, max_depth=max_depth)
            generations.append(prog)
        generations = [i for i in generations if i != None]
    return generations


def sample_baseline(
    cores=int(os.cpu_count() / 2),
    samples: int = 10000,
    min_depth: int = 1,
    max_depth: int = 16,
    sample_size: int = 50,
    output_dir: str = "results_official"
):
    time_start = time.time()
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    experiment_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        output_dir,
        f"sampling-depths-{min_depth}-{max_depth}-{timestamp}",
    )
    os.makedirs(experiment_dir, exist_ok=True)
    results_filepath = os.path.join(experiment_dir, RESULTS_FILENAME)
    print(f"Results will be saved to: {results_filepath}")


    results = []
    acceptable_programs = repeated_generation(samples, min_depth, max_depth)
    for id in range(1, 18 + 1):
        board_start_time = time.time()
        cache = {}
        print(f"board {id}", end=" | ")
        if cores > 1:
            for sample_index in tqdm(range(ceil(samples / sample_size))):
                unique_programs = []
                program_tuples = acceptable_programs[
                    sample_index * sample_size : ((sample_index + 1) * sample_size)
                ]
                program_selection = [item[0] for item in program_tuples]
                program_selection_depths = [item[1] for item in program_tuples]
                for program in program_selection:
                    key = (program, id)
                    if key in list(cache.keys()):
                        result = {
                            "program": program,
                            "board_id": id,
                            "score": cache[key],
                            "depth": program_selection_depths[index],
                            "min_depth": min_depth,
                            "max_depth": max_depth,
                        }
                        results.append(result)
                    else:
                        unique_programs.append(program)

                program_scores = compute_score_parallel(
                    programs=unique_programs,
                    board=Board.from_trial_id(id),
                    processes=cores,
                    show_progress=False,
                )
                for index in range(len(program_scores)):
                    cache[(unique_programs[index], id)] = program_scores[index]
                    result = {
                        "program": program_selection[index],
                        "board_id": id,
                        "score": program_scores[index],
                        "depth": program_selection_depths[index],
                        "min_depth": min_depth,
                        "max_depth": max_depth,
                    }
                    results.append(result)
        print(f"finished scoring in {round(time.time()-board_start_time,2)}s from the start")

    df = pd.DataFrame(results)
    df.to_csv(results_filepath, index=False)
    #df.to_csv(f"sampling_data_depths_{min_depth}_{max_depth}.csv", mode="a", header=False)
    print(f"finished {samples}-shot sampling at depth {(min_depth,max_depth)} in time {time.time() - time_start}")
    return df

df = sample_baseline(cores=os.cpu_count() - 1, samples=10, min_depth=2, max_depth=16, sample_size=2)
