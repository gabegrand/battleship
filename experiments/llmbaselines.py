import numpy as np
from battleship.board import *
from openai import OpenAI
from battleship.translator import *
from eig import *
from eig.battleship.program import ProgramSyntaxError
import matplotlib.pyplot as plt
import standard_prompts
import io
from base64 import b64encode

client = OpenAI(api_key="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")

prompt_base_translator = standard_prompts.constructPrompt(standard_prompts.translator_constant,["examples.csv","additional_examples.csv"])

def callOpenAI(model_used,system_prompt,inputBoard,n_completions):
  if model_used != "gpt-4-vision-preview":
    completion = client.chat.completions.create(
      model=model_used,
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(inputBoard)}
      ], n=n_completions
    )
  else:
    completion = client.chat.completions.create(
      model=model_used,
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{str(inputBoard)}"
            }
        }
        ]}
      ], n=n_completions
    )
  return completion

def eigFromLLM(context_file,mode, n_completion):
  questions = []
  lines = []
  board_path = os.path.join(f"{os.getcwd()}/battleship/question_dataset", "contexts", f"board_{context_file}.txt")
  contextBoard = Board.from_text_file(board_path).board

  if mode == "ascii":
    inputBoard = Board(contextBoard) #ASCII representation
    prompt = standard_prompts.ascii_prompt
  if mode == "serial":
    inputBoard = str(Board.serialized_representation(Board(contextBoard), conv=True)) #Serialized representation
    prompt = standard_prompts.serial_prompt
  if mode == "vision": #Image representation
    inputBoard = Board(contextBoard).to_figure()
    IObytes = io.BytesIO()
    inputBoard.savefig(IObytes,format="jpg") #Saves figure to buffer as to avoid unnecessary I/O on disk
    IObytes.seek(0)
    inputBoard = b64encode(IObytes.read()).decode('utf-8')
    prompt = standard_prompts.vision_prompt

  if mode != "vision":
    completion = callOpenAI("gpt-4",prompt,inputBoard,n_completion)
  else:
    completion = callOpenAI("gpt-4-vision-preview",prompt,inputBoard,n_completion)

  for i in range(0,n_completion):
    question = str(completion.choices[i].message.content)
    questions.append(question)

    completion_translation = callOpenAI("gpt-4",prompt_base_translator,questions[i],1)
    question_code = str(completion_translation.choices[0].message.content).strip().replace('"',"")

    try:
        score = compute_eig_fast(question_code, contextBoard, grid_size=6, ship_labels=[1, 2, 3], ship_sizes=[2, 3, 4], orientations=['V', 'H'])
    except ProgramSyntaxError:
        score = -1
    except RuntimeError:
        score = 0
    lines.append([questions[i],question_code,str(score)])
    
  return lines

def generateResponses(mode,max_context,n_completion):
  with open(f"responses_{mode}.txt", "w+") as responseFile:
    for i in range(1,max_context+1):
      response = [f"Context {i}"]
      llm_response = eigFromLLM(i,mode,n_completion)
      for i in range(0,n_completion):
        response_write = response + llm_response[i]
        responseFile.write("\t".join(response_write)+"\n")

scores = []
mode, max_context, completions = "serial", 18, 10
generateResponses(mode,max_context,completions)
with open(f"responses_{mode}.txt", "r") as responseFile:
  for line in responseFile:
    score = float(line.split("\t")[3].strip())
    scores.append(score)
plt.hist(scores,[-1,-0.5,-0.1,0,0.25,0.5,0.75,1,1.25,2,3,4,5])
plt.savefig(f"responses_{mode}.svg")